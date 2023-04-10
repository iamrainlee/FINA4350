import sys
import csv
import pandas as pd
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

def remove_unrelated_text(x): #x = df.content
  if x[0] == ' ':
    x = x[1:]
  if x[-1] == ' ':
    x = x[:-1]
  if 'Return to top' in x:
    x = x[:x.find('Return to top')]
  if 'Note:' in x:
    x = x[:x.rfind(' Note:')]
  if 'Notes:' in x:
    x = x[:x.rfind(' Notes:')]
  if 'footnotes' in x:
    x = x[:x.find('footnotes')]
  if 'Footnotes' in x:
    x = x[:x.find('Footnotes')]
  x = x.replace('       ', ' ')
  x = x.replace('     ', ' ')
  x = x.replace('  ', ' ')
  return x

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

total_testimony = pd.read_csv('../Data/Raw Data/FOMC_Testimony.csv')

total_testimony.content = total_testimony.content.apply(remove_unrelated_text)

total_testimony.content = total_testimony.content.apply(str_preprocess)

total_testimony_preprocessed = total_testimony

total_testimony_preprocessed.drop('Unnamed: 0', axis = 1, inplace = True)

total_testimony_preprocessed.to_csv('../Data/Preprocessed Data/testimony_preprocessed.csv', index = False)
