import pandas as pd
import sys
import csv
from pandasql import sqldf
pysqldf = lambda q: sqldf (q, globals())

#allow import of csv of large file size while preventing interger overflow error
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

ratehike = pd.read_csv("../Data/Preprocessed Data/ratehikes_process.csv",engine='python',converters={"Date":pd.to_datetime,"rate_hike":float})
minutes = pd.read_csv("../Data/Preprocessed Data/minutes_preprocessed.csv",engine='python',converters={"index":pd.to_datetime,"Federal_Reserve_Mins":str})
statements = pd.read_csv("../Data/Raw Data/FOMCStatement.csv",engine='python',converters={"Unnamed: 0":pd.to_datetime,"FOMC_Statements":str})

# rename columns to Date
statements.rename(columns={'Unnamed: 0':'Date'},inplace=True)
minutes.rename(columns={'index':'Date'},inplace=True)

#change datatype of date to string for sql
ratehike['Date'] = ratehike['Date'].astype(str)
minutes['Date'] = minutes['Date'].astype(str)
statements['Date'] = statements['Date'].astype(str)

#add a column for the search range
ratehike["From_Date"] = ratehike['Date'].shift(1)
ratehike.dropna(inplace=True)

#change the units of rate hike for easier understanding
ratehike['rate_hike'] = ratehike['rate_hike']*10000
ratehike['Interest'] = ratehike['Interest']*100

#sql query for merging data using subquery and concatenating the textual data
df = pysqldf("SELECT r.Date, r.From_Date, r.rate_hike, \
            (SELECT s.FOMC_Statements FROM statements s WHERE s.Date >= r.From_Date AND s.Date < r.Date) || ' ' ||\
            (SELECT m.Federal_Reserve_Mins FROM minutes m WHERE m.Date >= r.From_Date AND m.Date < r.Date) \
            AS data FROM ratehike as r WHERE data <> ' '")

df.to_csv('../Data/Merged Data/rate_FOMC.csv',index=False)