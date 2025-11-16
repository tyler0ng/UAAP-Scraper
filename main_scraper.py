from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import numpy as np

#Season year
theseason=88

#Input file saving path
pathsave = r" "


lsheaders = ['Name', 'Minutes', 'PTS', 'FG M/A', 'FG %', '2 Points M/A', '2 Points %', '3 Points M/A', '3 Points %',
             'Free Throws M/A', 'Free Throws %', 'Rebounds OR', 'Rebounds DR', 'Rebounds TOT',
             'AST', 'TO', 'STL', 'BLK', 'FOULS PF', 'FOULS FD', '+/-']
lsheaderupdated = ['Name', 'Minutes', 'FG M/A', 'FG %', '2 Points M/A', '2 Points %', '3 Points M/A', '3 Points %',
             'Free Throws M/A', 'Free Throws %', 'Rebounds OR', 'Rebounds DR', 'Rebounds TOT',
             'AST', 'TO', 'STL', 'BLK', 'FOULS PF', 'FOULS FD', '+/-', 'PTS']
rows = []


lsurl = list(range(1, 45))
for gonum in lsurl:
    print(gonum)
    i = 0  #team index
    url = "https://uaap.livestats.ph/tournaments/uaap-season-"+str(theseason)+"-men-s-basketball?game_id=" +str(gonum)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    teams = [el.get_text(strip=True) for el in soup.select("th.team-0 .team-code, th.team-1 .team-code")]
    time.sleep(3) #to prevent rate limit
    for tr in soup.find_all("tr"):
        if tr.find("td", class_="text-left totals-title"):   #only TOTALS rows
            tds = [td.get_text(strip=True) for td in tr.find_all("td") if td.get_text(strip=True) != ""]

            if tds and tds[0].upper().startswith("TEAM TOTALS"):
                tds[0] = f"{teams[i]} Totals" + " " + str(gonum)
                i += 1

            rows.append(tds)



#Turn into DataFrame
print(rows)
df = pd.DataFrame(rows)
df = df[~df.apply(lambda s: s.astype(str).str.contains(r"Team\s*/\s*Coach", case=False, na=False)).any(axis=1)].reset_index(drop=True)

df.columns=(lsheaders)
df = df[lsheaderupdated]
print(df.to_string())
df['season'] = 'SS'+str(theseason)




#Sample names: "UP Totals 37", "LA SALLE Totals 12"
df['game_id'] = (
    df['Name'].astype(str).str.extract(r'(\d+)').iloc[:, 0].astype('Int64')
)
df['team'] = (
    df['Name'].astype(str)
      .str.replace(r'\s*Totals.*$', '', regex=True)  # drop ' Totals ...'
      .str.strip()
)

df['ASTtoTO'] = df['AST'].astype('Int64')/df['TO'].astype('Int64')
df[['FGM', 'FGA']] = df['FG M/A'].str.split(r'[/-]', expand=True)
df[['3PM', '3PA']] = df['3 Points M/A'].str.split(r'[/-]', expand=True)
df[['FTM', 'FTA']] = df['Free Throws M/A'].str.split(r'[/-]', expand=True)
df = df.astype({'FGA':'float',
                'FGM':'float',
                '3PA':'float',
                '3PM':'float',
                'FTM': 'float',
                'FTA':'float',
                'PTS':'float',
                'TO':'float',
                'Rebounds OR':'float',
                'Rebounds DR': 'float',
                'Rebounds TOT': 'float'
                })


# Shooting metrics express as %
df['eFG %'] = np.where(
    df['FGA'] > 0,
    (df['FGM'] + 0.5*df['3PM']) / df['FGA'] * 100,
    np.nan
)

df['TS %'] = np.where(
    (df['FGA'] + 0.44*df['FTA']) > 0,
    df['PTS'] / (2*(df['FGA'] + 0.44*df['FTA'])) * 100,
    np.nan
)

df['TOV %'] = np.where(
    (df['FGA'] + 0.44*df['FTA'] + df['TO']) > 0,
    df['TO'] / (df['FGA'] + 0.44*df['FTA'] + df['TO']) * 100,
    np.nan
)
#Free throw rate
df['FTR (FTA/FGA) %'] = np.where(
    df['FGA'] > 0,
    df['FTA'] / df['FGA'] * 100,
    np.nan
)

res = df.merge(
    df,
    on=['season','game_id'],
    how='inner',
    suffixes=('', ' opp')
)

#Drop same rows that match
res = res[res['team'] != res['team opp']].copy()

#____Opponent-based metrics____
#Rebound rates
res['ORB %'] = np.where(
    (res['Rebounds OR'] + res['Rebounds DR opp']) > 0,
    res['Rebounds OR'] / (res['Rebounds OR'] + res['Rebounds DR opp']) * 100,
    np.nan
)
res['DRB %'] = np.where(
    (res['Rebounds DR'] + res['Rebounds OR opp']) > 0,
    res['Rebounds DR'] / (res['Rebounds DR'] + res['Rebounds OR opp']) * 100,
    np.nan
)

#Print and save to path as csv
print(res.to_string())
res.to_csv(pathsave, index=False)

