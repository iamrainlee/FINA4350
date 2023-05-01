from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import sys
import csv

#allow import of csv of large file size while preventing interger overflow error
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

data_list = ['rate_FOMC_speeches_testimony', 'rate_speeches_testimony', 'rate_FOMC'] #data available
data_chosen = input("Please choose the data :")

while (data_chosen not in data_list): #proceeds only if the data is inputted correctly
  data_chosen = input("Please choose again :")

data = pd.read_csv('../Data/Merged Data/' + data_chosen + '.csv')

data['data'] = data['data'].str.replace('.','') #remove . 

X = data['data']
y = data['rate_hike']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

cnt_vect = CountVectorizer() #bag-of-word
X_train_count = cnt_vect.fit_transform(X_train)
X_test_count = cnt_vect.transform(X_test)

tfidf_transformer = TfidfTransformer() #TFIDF Vectorization
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)
X_test_tfidf = tfidf_transformer.transform(X_test_count)

clf_tfidf = KNeighborsClassifier() #knn classifier

clf_tfidf.fit(X_train_tfidf, y_train)

pred1_tfidf = clf_tfidf.predict(X_test_tfidf)

accu1_tfidf = accuracy_score(y_test, pred1_tfidf)

score_tfidf = cross_val_score(clf_tfidf, X_train_tfidf, y_train, cv = 5) #5-fold cross validation

print("mean score for 5-fold cross validation :", score_tfidf.mean())

k_range = list(range(1, 16))

param_grid_tfidf = dict(n_neighbors=k_range)

grid_search_tfidf = GridSearchCV(estimator = clf_tfidf, param_grid = param_grid_tfidf,  #gridcv
                          cv = 5, n_jobs = -1, verbose = 2)

grid_search_tfidf.fit(X_train_tfidf, y_train)

grid_search_tfidf.best_params_

pred2_tfidf = grid_search_tfidf.predict(X_test_tfidf) #prediction using the best parameter derived from gridcv

accu2_tfidf = accuracy_score(y_test, pred2_tfidf)

print(" Data     : {0}\n TFIDF\n initial  : accuracy : {1}\n gridcv   : accuracy : {2}".format(data_chosen, accu1_tfidf, accu2_tfidf))