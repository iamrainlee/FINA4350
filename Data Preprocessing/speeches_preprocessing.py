# This doc preprocesses speeches.

import pandas as pd
import sys
import csv
from ast import literal_eval
import re

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from nltk.stem import PorterStemmer
ps = PorterStemmer()

#allow import of csv of large file size while preventing interger overflow error
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

#load the csv
df = pd.read_csv("../Data/Raw Data/speeches.csv",engine='python',converters={"Date":pd.to_datetime,"Content":str})

def strToList(x):
    """
    This function converts string-representation of list to list and skip lines if the format is incorrect
    """
    try:
        return literal_eval(str(x))   
    except Exception as e:
        print(e)
        return []

def removeUnrelated(l):
    """
    This function removes paragraphs that are irrelevant.
    """

    #removes lines teaching keys to control the video
    if l[0] == "Accessible Keys for Video":
        l = l[7:]
    
    # remove None from the list
    l = list(filter(lambda item: item is not None, l))

    # skip references and footnotes
    for i,x in enumerate(l):
        if re.match('References\r\n',x):
            return l[:i]
        if re.match(r'\d+. \w+',x):
            return l[:i]
    return l

def str_preprocess(s):
    """
    Preprocess the string by removing symbols, short words and stop words
    """
    global ps #stemming
    s = re.sub(r"[^a-zA-Z.]", " ",s) #remove symbols except full stop.
    s = re.sub(r'\s+',' ',s)
    s = ' '.join([ps.stem(w) for w in s.split() if len(w)>3 and w not in stop_words])
    s = s.lower()
    return s

#Apply preprocessing functions to Content
df["Content"] = df.Content.apply(strToList)
df["Content"] = df.Content.apply(removeUnrelated)
df["Content"] = df.Content.apply(lambda x: " ".join(x))
df["Content"] = df.Content.apply(str_preprocess)

df = df[df["Content"].str.len()>300] #remove empty/ failed to retrieve content

df.sort_values('Date',inplace=True) #sort by date for merging

df.to_csv("../Data/Preprocessed Data/speeches_preprocessed.csv",index=False) #save