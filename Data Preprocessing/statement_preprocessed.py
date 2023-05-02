import sys
import csv
import pandas as pd
#allow import of csv of large file size while preventing interger overflow error
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

df = pd.read_csv("../Data/Raw Data/FOMCStatement.csv",index_col=0)

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english') #remove stopwords

from nltk.stem import PorterStemmer
ps = PorterStemmer()

import re

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

pattern = re.compile('The Federal Reserve.*?Share')
df['FOMC_Statements'] = df['FOMC_Statements'].apply(lambda x: pattern.sub('', x))

pattern2= re.compile('Home | News.*?Last update:')
df['FOMC_Statements'] = df['FOMC_Statements'].apply(lambda x: pattern2.sub('', x))

pattern3= r'\s*Home\s*\|\s*News\s+and\s+events\s+Accessibility\s+Last\s+update:\s*\w+\s+\d+,\s+\d+\s*'
# Use re.sub() to remove the text and date
df['FOMC_Statements'] = df['FOMC_Statements'].apply(lambda x: re.sub(pattern3, '', x))

# Define the regular expression pattern
pattern4 = r'For immediate release'

# Use re.sub() to remove the text
df['FOMC_Statements'] = df['FOMC_Statements'].apply(lambda x: re.sub(pattern4, '', x))

df["FOMC_Statements"] = df["FOMC_Statements"].apply(str_processing)
df.to_csv("../Data/Preprocessed Data/statement_preprocessed.csv")



