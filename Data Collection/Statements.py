# import relevant package and collect dataset:
from FedTools import MonetaryPolicyCommittee
dataset = MonetaryPolicyCommittee().find_statements()

# for data manipulation:
import numpy as np
import pandas as pd
import datetime as dt

# silence warnings:
import warnings 
warnings.filterwarnings("ignore")

for i in range(len(dataset)):
  dataset.iloc[i,0] = dataset.iloc[i,0].replace('\n','. ')
  dataset.iloc[i,0] = dataset.iloc[i,0].replace('\n',' ')
  dataset.iloc[i,0] = dataset.iloc[i,0].replace('\r',' ')
  dataset.iloc[i,0] = dataset.iloc[i,0].replace('\xa0',' ')

dataset.to_csv("FOMCStatement.csv")
