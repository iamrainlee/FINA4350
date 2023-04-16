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

ratehike = pd.read_csv("../Data/Preprocessed Data/ratehikes_preprocessed.csv",engine='python',converters={"Date":pd.to_datetime,"rate_hike":float})
speeches = pd.read_csv("../Data/Preprocessed Data/speeches_preprocessed.csv",engine='python',converters={"index":pd.to_datetime,"Content":str})
testimony = pd.read_csv("../Data/Preprocessed Data/testimony_preprocessed.csv",engine='python',converters={"date":pd.to_datetime,"content":str})

# rename columns to Date
speeches.rename(columns={'index':'Date'},inplace=True)
testimony.rename(columns={'date':'Date'},inplace=True)

#change datatype of date to string for sql
ratehike['Date'] = ratehike['Date'].astype(str)
speeches['Date'] = speeches['Date'].astype(str)
testimony['Date'] = testimony['Date'].astype(str)

#add a column for the search range
ratehike["From_Date"] = ratehike['Date'].shift(1)
ratehike.dropna(inplace=True)

#change the units of rate hike for easier understanding
ratehike['rate_hike'] = ratehike['rate_hike']*10000
#ratehike['Interest'] = ratehike['Interest']*100

#sql query for merging data using subquery and concatenating the textual data
df = pysqldf("SELECT r.Date, r.From_Date, r.rate_hike, \
            IFNULL((SELECT s.Content FROM speeches s WHERE s.Date >= r.From_Date AND s.Date < r.Date),'') || ' ' ||\
            IFNULL((SELECT t.content FROM testimony t WHERE t.Date >= r.From_Date AND t.Date < r.Date),'') \
            AS data FROM ratehike as r WHERE data <> ' '")

df.to_csv('../Data/Merged Data/rate_speeches_testimony.csv',index=False)