import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error

# ---------------- USER CONFIG ----------------
# Use ONE of the two modes below:

# MODE A: single merged file (e.g., MergedChanged.csv containing 85/86/87/88)
MERGED_CSV = r" "
USE_MERGED_ONLY = False

#For MODE B (two files), set USE_MERGED_ONLY = False and fill these:
TRAIN_CSV = r"/Users/tylerong/PycharmProjects/PythonProject/MergedChanged.csv"   # SS85–SS87 merged
SS88_CSV  = r"/Users/tylerong/PycharmProjects/PythonProject/ss88.csv"           # SS88 only

#Seasons to train vs predict (numbers)
TRAIN_SEASONS = {85, 86, 87}
PRED_SEASON_NUM = 88

#Rolling window: None (expanding) or an int (e.g., 5 for the last 5)
ROLLING_WINDOW = None

# History gates for SS88 predictions
MIN_PRIOR_GAMES_TEAM = 3
MIN_PRIOR_GAMES_OPP  = 3

#Outputs
OUTPUT_PRED = "ss88_predictions_from_8587_model.csv"
OUTPUT_SUMMARY = "ss88_team_summary_from_8587_model.csv"

#Single matchup (optional): predict TEAM_TO_PRED vs OPP_TO_PRED in SS88 (latest pre-game averages)
ENABLE_SINGLE_MATCHUP = True
TEAM_TO_PRED = "UP"
OPP_TO_PRED  = "UE"

"""#########################################"""

TEAM_OFF_FEATURES = [
    'FGA','3PA','FTA',
    'eFG %','TS %','FTR (FTA/FGA) %',
    'TOV %','AST','ORB %'
]

OPP_DEF_ALLOWED_FEATURES_MAP = {
    'eFG % opp':           'opp_def_eFG_allowed',
    'TS % opp':            'opp_def_TS_allowed',
    'FTR (FTA/FGA) % opp': 'opp_def_FTR_allowed',
    'TOV % opp':           'opp_def_TOV_forced',
    'ORB % opp':           'opp_def_ORB_allowed',
}

POSSESSIONS_FORMULA_COLS = ['FGA','FTA','TO','Rebounds OR']


def season_to_num(val):
    try:
        return int(val)
    except Exception:
        m = re.search(r'\d+', str(val))
        return int(m.group(0)) if m else np.nan


def rolling_mean(series, window=ROLLING_WINDOW):
    """Past-only average: expanding if window is None; else last-N (min_periods=1)."""
    s = series.shift()
    if window is None:
        return s.expanding().mean()
    return s.rolling(window=window, min_periods=1).mean()


def add_possessions_and_ppp(df):
    if all(col in df.columns for col in POSSESSIONS_FORMULA_COLS):
        for c in POSSESSIONS_FORMULA_COLS:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['Poss'] = df['FGA'] + 0.44*df['FTA'] + df['TO'] - df['Rebounds OR']
    else:
        df['Poss'] = np.nan
    df['PPP'] = np.where(df['Poss'] > 0, pd.to_numeric(df['PTS'], errors='coerce') / df['Poss'], np.nan)
    return df


def build_feature_frame(df):
    """Compute pre-game *_off_avg and *_opp_avg features; return enriched df and predictor column names."""
    #required columns
    required = ['season','team','team opp','game_id','PTS']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    #normalize season numeric
    df = df.copy()
    df['season_num'] = df['season'].apply(season_to_num)

    #coerce numeric team offense inputs
    for c in TEAM_OFF_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    #possessions & PPP
    df = add_possessions_and_ppp(df)

    #choose grouping key (prefer numeric if any)
    use_season_key = 'season_num' if df['season_num'].notna().any() else 'season'
    grp_key = [use_season_key, 'team']

    #sort chronologically
    df = df.sort_values([use_season_key, 'team', 'game_id']).copy()

    #ream offense pre-game averages
    off_avg_cols = []
    for col in TEAM_OFF_FEATURES + ['Poss']:
        if col in df.columns:
            avg_col = f"{col}_off_avg"
            df[avg_col] = df.groupby(grp_key)[col].transform(rolling_mean)
            off_avg_cols.append(avg_col)

    #team prior games
    df['team_prior_games'] = df.groupby(grp_key).cumcount()

    #opponent defensive-allowed pre-game averages (compute on team rows, merge as *_opp_avg)
    def_src_cols = [c for c in OPP_DEF_ALLOWED_FEATURES_MAP.keys() if c in df.columns]
    def_avg_cols_own = []
    for src in def_src_cols:
        outname = OPP_DEF_ALLOWED_FEATURES_MAP[src]
        avg_col = f"{outname}_avg"
        df[avg_col] = df.groupby(grp_key)[src].transform(rolling_mean)
        def_avg_cols_own.append(avg_col)

    if def_avg_cols_own:
        opp_def_table = df[[use_season_key, 'team', 'game_id'] + def_avg_cols_own].copy()
        opp_def_table = opp_def_table.rename(columns={'team': 'opp_team'})
        rename_to_opp = {c: c.replace('_avg','_opp_avg') for c in def_avg_cols_own}
        opp_def_table = opp_def_table.rename(columns=rename_to_opp)

        df = df.merge(
            opp_def_table,
            how='left',
            left_on=[use_season_key, 'team opp', 'game_id'],
            right_on=[use_season_key, 'opp_team', 'game_id']
        )

    #opponent prior games (for gating)
    team_prior_tmp = (
        df[[use_season_key,'team','game_id']]
        .sort_values([use_season_key,'team','game_id'])
        .assign(prior_games=lambda x: x.groupby([use_season_key,'team']).cumcount())
    )
    opp_prior_table = team_prior_tmp.rename(columns={'team':'opp_team','prior_games':'opp_prior_games'})
    df = df.merge(
        opp_prior_table[[use_season_key,'opp_team','game_id','opp_prior_games']],
        how='left',
        left_on=[use_season_key,'team opp','game_id'],
        right_on=[use_season_key,'opp_team','game_id']
    )

    def_avg_cols_opp = [c for c in df.columns if c.endswith('_opp_avg')]
    predictor_cols = off_avg_cols + def_avg_cols_opp
    return df, predictor_cols, use_season_key


def fit_ridge_ppp(train_df, predictor_cols):
    needed = predictor_cols + ['PPP']
    tdf = train_df.dropna(subset=needed).copy()
    X = tdf[predictor_cols].apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(tdf['PPP'], errors='coerce')

    valid = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()
    if len(X) == 0:
        raise ValueError("No valid rows to train on after dropping NaNs.")

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5))
    ])
    pipe.fit(X, y)

    y_hat = pipe.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    r2 = r2_score(y, y_hat)
    print(f"[TRAIN PPP] rows={len(y)} | alpha={pipe.named_steps['ridge'].alpha_:.6f} | R^2={r2:.4f} | RMSE={rmse:.5f}")

    return pipe, rmse


def predict_ss88_table(ss88_df, predictor_cols, pipe, season_key):
    # history gate
    mask = (ss88_df['team_prior_games'] >= MIN_PRIOR_GAMES_TEAM)
    if 'opp_prior_games' in ss88_df.columns:
        mask &= (ss88_df['opp_prior_games'] >= MIN_PRIOR_GAMES_OPP)
    use_df = ss88_df.loc[mask].copy()

    if 'Poss_off_avg' not in use_df.columns:
        use_df['Poss_off_avg'] = np.nan

    X = use_df[predictor_cols].apply(pd.to_numeric, errors='coerce')
    valid = X.dropna().index
    X = X.loc[valid]
    use_df = use_df.loc[valid].copy()
    if len(X) == 0:
        raise ValueError("No SS88 rows meet history requirements and have complete features.")

    y_hat_ppp = pipe.predict(X)
    use_df['PPP_pred'] = y_hat_ppp
    use_df['Poss_for_pred'] = np.where(use_df['Poss_off_avg'].notna(), use_df['Poss_off_avg'], use_df['Poss'])
    use_df['PTS_pred'] = use_df['PPP_pred'] * use_df['Poss_for_pred']
    use_df['PTS_actual'] = use_df['PPP'] * use_df['Poss']
    use_df['err_PTS'] = use_df['PTS_pred'] - use_df['PTS_actual']

    cols_out = ['season','season_num','team','team opp','game_id','PTS_actual','PTS_pred','err_PTS']
    cols_out = [c for c in cols_out if c in use_df.columns]
    pred_out = use_df[cols_out].sort_values([season_key, 'team', 'game_id']).reset_index(drop=True)

    if {'PTS_actual','PTS_pred'}.issubset(pred_out.columns):
        team_summary = pred_out.groupby(['season','team']).agg(
            n=('PTS_actual','size'),
            rmse=('err_PTS', lambda e: np.sqrt(np.mean(e**2))),
            mae=('err_PTS',  lambda e: np.mean(np.abs(e)))
        ).reset_index().sort_values(['season','team'])
    else:
        team_summary = pd.DataFrame()

    return pred_out, team_summary


def latest_team_vector_from_ss88(ss88_enriched, predictor_cols, team_name, opp_name, season_key):
    team_rows = ss88_enriched[ss88_enriched['team'] == team_name].copy()
    opp_rows  = ss88_enriched[ss88_enriched['team'] == opp_name].copy()
    if team_rows.empty:
        raise ValueError(f"No SS88 rows found for team '{team_name}'.")
    if opp_rows.empty:
        raise ValueError(f"No SS88 rows found for opponent '{opp_name}'.")

    team_latest = team_rows.sort_values([season_key,'game_id']).iloc[-1]
    opp_latest  = opp_rows.sort_values([season_key,'game_id']).iloc[-1]

    if team_latest.get('team_prior_games', 0) < MIN_PRIOR_GAMES_TEAM:
        raise ValueError(f"Team '{team_name}' has only {int(team_latest.get('team_prior_games',0))} prior SS88 games (need ≥ {MIN_PRIOR_GAMES_TEAM}).")
    if opp_latest.get('team_prior_games', 0) < MIN_PRIOR_GAMES_OPP:
        raise ValueError(f"Opponent '{opp_name}' has only {int(opp_latest.get('team_prior_games',0))} prior SS88 games (need ≥ {MIN_PRIOR_GAMES_OPP}).")

    x = {}
    for col in predictor_cols:
        if col.endswith('_off_avg'):
            x[col] = float(team_latest.get(col, np.nan))
        elif col.endswith('_opp_avg'):
            own_name = col.replace('_opp_avg', '_avg')
            x[col] = float(opp_latest.get(own_name, np.nan))
        else:
            x[col] = float(team_latest.get(col, np.nan))

    poss_off_avg = float(team_latest.get('Poss_off_avg', np.nan))
    poss_fallback = float(team_latest.get('Poss', np.nan))

    # get a display season value
    season_raw = team_latest.get('season')
    try:
        season_val = int(season_raw)
    except Exception:
        m = re.search(r'\d+', str(season_raw))
        season_val = int(m.group(0)) if m else str(season_raw)

    return x, poss_off_avg, poss_fallback, season_val


def main():
    # ---- Load & split data (merged or two files) ----
    if USE_MERGED_ONLY:
        df_all = pd.read_csv(MERGED_CSV)
        # Normalize numeric season for filtering
        df_all['season_num'] = df_all['season'].apply(season_to_num)

        train_raw = df_all[df_all['season_num'].isin(TRAIN_SEASONS)].copy()
        ss88_raw  = df_all[df_all['season_num'] == PRED_SEASON_NUM].copy()
    else:
        train_raw = pd.read_csv(TRAIN_CSV)
        ss88_raw  = pd.read_csv(SS88_CSV)

    #_____Build features for train and SS88_____
    train_enriched, predictor_cols, season_key_train = build_feature_frame(train_raw)
    print(f"Predictor columns ({len(predictor_cols)}): {predictor_cols}")

    pipe, train_ppp_rmse = fit_ridge_ppp(train_enriched, predictor_cols)

    ss88_enriched, predictor_cols_pred, season_key_pred = build_feature_frame(ss88_raw)

    for c in predictor_cols:
        if c not in ss88_enriched.columns:
            ss88_enriched[c] = np.nan

    #_____Predict SS88 (full table)_____
    pred_out, team_summary = predict_ss88_table(ss88_enriched, predictor_cols, pipe, season_key_pred)
    pred_out.to_csv(OUTPUT_PRED, index=False)
    print(f"\n✅ SS88 predictions written to: {OUTPUT_PRED}")

    if not team_summary.empty and {'PTS_actual','PTS_pred'}.issubset(pred_out.columns):
        team_summary.to_csv(OUTPUT_SUMMARY, index=False)
        print(f"✅ SS88 team summary written to: {OUTPUT_SUMMARY}")
        rmse_all = np.sqrt(mean_squared_error(pred_out['PTS_actual'], pred_out['PTS_pred']))
        r2_all = r2_score(pred_out['PTS_actual'], pred_out['PTS_pred'])
        print(f"[SS88 EVAL] rows={len(pred_out)} | R^2={r2_all:.4f} | RMSE={rmse_all:.4f}")
    else:
        print("(i) SS88 actuals not fully available or no rows passed the history filter.")

    #_____Single Matchup_____
    if ENABLE_SINGLE_MATCHUP:
        x_dict, poss_off_avg, poss_fallback, season_val = latest_team_vector_from_ss88(
            ss88_enriched, predictor_cols, TEAM_TO_PRED, OPP_TO_PRED, season_key_pred
        )
        X_vec = pd.DataFrame([{c: x_dict.get(c, np.nan) for c in predictor_cols}])
        ppp_pred = float(pipe.predict(X_vec)[0])
        poss_for_pred = poss_off_avg if not np.isnan(poss_off_avg) else poss_fallback
        pts_pred = ppp_pred * poss_for_pred if not np.isnan(poss_for_pred) else np.nan
        pts_rmse_est = train_ppp_rmse * poss_for_pred if not np.isnan(poss_for_pred) else np.nan

        print("\n===== SINGLE MATCHUP PREDICTION =====")
        print(f"Season: {season_val} | Team: {TEAM_TO_PRED} vs Opp: {OPP_TO_PRED}")
        print(f"PPP_pred: {ppp_pred:.4f}")
        if not np.isnan(pts_pred):
            lo = pts_pred - pts_rmse_est
            hi = pts_pred + pts_rmse_est
            print(f"PTS_pred: {pts_pred:.2f}  (± ~{pts_rmse_est:.2f})  ~ [{lo:.1f}, {hi:.1f}]")
        else:
            print("PTS_pred: NaN (missing possession info)")


if __name__ == "__main__":
    main()
