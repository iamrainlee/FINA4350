import pandas as pd
import csv
import sys
maxInt = sys.maxsize
import re

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english') #remove stopwords

from nltk.stem import PorterStemmer
ps = PorterStemmer()

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs, we catch the error
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
df = pd.read_csv('../Data/Raw Data/FOMC meeting minutes.csv',engine="python",converters={"index":pd.to_datetime,"Federal_Reserve_Mins":str})

def removeUnrelatedText(x):
    # remove words before the line
    if x.find('By unanimous vote, the minutes of the meeting of the Federal Open Market Committee held on')>0:
        x = x.split('By unanimous vote, the minutes of the meeting of the Federal Open Market Committee held on')
        if len(x)>=1:
            x = " ".join(x[1:])
            x = x[x.find('were approved.')+14:]
            return x
    else:
        return x
    return x

def str_processing(s):
    """
    Preprocess the string by removing symbols, short words and stop words
    """
    global ps #stemming
    s = re.sub(r"[^a-zA-Z.]", " ",str(s)) #remove symbols except full stop.
    s = re.sub(r'\s+',' ',str(s))
    s = ' '.join([ps.stem(w) for w in s.split() if len(w)>3 and w not in stop_words])
    s = s.lower()
    return s
df['Federal_Reserve_Mins']=df.Federal_Reserve_Mins.apply(lambda x: re.sub(r'\s+',r' ',str(x)))
df['Federal_Reserve_Mins']=df.Federal_Reserve_Mins.apply(removeUnrelatedText) #remove unrelated text such as participants of the FOMC meeting and voting result that is considered irrelevant
df['Federal_Reserve_Mins'] = df.Federal_Reserve_Mins.apply(str_processing)

df.to_csv("../Data/Preprocessed Data/minutes_preprocessed.csv",index=False)
