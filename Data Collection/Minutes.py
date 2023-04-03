from FedTools import MonetaryPolicyCommittee
from FedTools import BeigeBooks
from FedTools import FederalReserveMins
from FedTools import MonetaryPolicyCommittee
from FedTools import BeigeBooks
from FedTools import FederalReserveMins

#use the find minutes function to return the dataset containing Federal Reserve Minutes since 1993
dataset = FederalReserveMins().find_minutes()
dataset.reset_index(inplace=True)
dataset.rename({"index":"date"},inplace=True)
# save the dataset into csv format
dataset.to_csv("../Data/Raw Data/FOMC meeting minutes.csv",index=False)
