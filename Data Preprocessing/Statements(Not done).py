import pandas as pd
import csv
import sys
csv.field_size_limit(sys.maxsize) #allow import of csv of large file size

df=pd.read_csv('FOMCStatement.csv')


#remove whitespace
def remove_whitespace(text):
    return  " ".join(text.split())

#spelling correction
from spellchecker import SpellChecker
def spell_check(text):
    
    result = []
    spell = SpellChecker()
    for word in text:
        correct_word = spell.correction(word)
        result.append(correct_word)
    
    return result

#tokenization
from nltk import word_tokenize
#df.apply(lambda X: word_tokenize(X))


#Remove punctuation
from nltk.tokenize import RegexpTokenizer

def remove_punct(text):
    
    tokenizer = RegexpTokenizer(r"\w+")
    lst=tokenizer.tokenize(' '.join(text))
    return lst

#remove frequent words
from nltk import FreqDist

def frequent_words(df):
    
    lst=[]
    for text in df.values:
        lst+=text[0]
    fdist=FreqDist(lst)
    return fdist.most_common(10)

def remove_freq_words(text):
    
    result=[]
    for item in text:
        if item not in lst:
            result.append(item)
    
    return result

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize,pos_tag

#lemmatization
def lemmatization(text):
    
    result=[]
    wordnet = WordNetLemmatizer()
    for token,tag in pos_tag(text):
        pos=tag[0].lower()
        
        if pos not in ['a', 'r', 'n', 'v']:
            pos='n'
            
        result.append(wordnet.lemmatize(token,pos))
    
    return result

#stemming
from nltk.stem import PorterStemmer

def stemming(text):
    porter = PorterStemmer()
    
    result=[]
    for word in text:
        result.append(porter.stem(word))
    return result

#remove tags
import re
def remove_tag(text):
    
    text=' '.join(text)
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

