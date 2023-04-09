import csv
import sys
import pandas as pd

# load the csv


csv.field_size_limit(sys.maxsize)

#allow import of csv of large file size while preventing interger overflow error
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


df = pd.read_csv("../Data/Raw Data/ratehike.csv")
df['rate_hike'] = df['Interest'].diff().round(4)

#print(df)

df.to_csv("../Data/Preprocessed Data/ratehikes_preprocessed.csv", index=False)


