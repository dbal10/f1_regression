import pandas as pd
races = pd.read_csv('../races.csv', header = 0)
results = pd.read_csv('../results.csv', header = 0)

racestest = races[(races['date'].str.match(r'2001-.*')) | (races['date'].str.match(r'2001-.*')) | (races['date'].str.match(r'2002-.*')) | (races['date'].str.match(r'2003-.*')) | (races['date'].str.match(r'2004-.*')) | (races['date'].str.match(r'2005-.*')) |(races['date'].str.match(r'2006-.*')) | (races['date'].str.match(r'2007-.*')) | (races['date'].str.match(r'2008-.*')) | (races['date'].str.match(r'2009-.*'))]
racestest.to_csv('./racestest.csv', index= False)

resultstest = results[results.raceId.isin(racestest['raceId'].values)]
resultstest.to_csv('./resultstest.csv', index=False)

