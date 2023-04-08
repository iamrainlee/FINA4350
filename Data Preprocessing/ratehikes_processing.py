import pandas as pd
import csv
# load the csv

df = pd.read_csv("../Data/Raw Data/ratehikes.csv")

df['rate_hike'] = df['FEDFUNDS'].diff()

df.to_csv("../Data/Preprocessed Data/ratehikes_processed.csv")

