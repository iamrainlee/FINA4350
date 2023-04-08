import pandas as pd
# load the csv
#df = pd.read_csv("ratehikes.csv", engine='python', converters={"FEDFUNDS": int})

df = pd.read_csv('ratehikes.csv')
df['rate_hike'] = df['FEDFUNDS'].diff().round(2)

print(df)

df.to_csv("ratehikes_process.csv", index=False)

