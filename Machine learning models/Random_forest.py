from sklearn.ensemble import RandomForestClassifier
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

data = pd.read_csv('../Data/Merged Data/rate_FOMC_speeches_testimony.csv')

#1. data
data['data'] = data['data'].str.replace('.','')

X = data['data']
y = data['rate_hike']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

cnt_vect = CountVectorizer()
X_train_count = cnt_vect.fit_transform(X_train)
X_test_count = cnt_vect.transform(X_test)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)
X_test_tfidf = tfidf_transformer.transform(X_test_count)

clf_tfidf = RandomForestClassifier() #tfidf
clf_count = RandomForestClassifier() #count_vectorizer

clf_tfidf.fit(X_train_tfidf, y_train)
clf_count.fit(X_train_count, y_train)

pred_tfidf = clf_tfidf.predict(X_test_tfidf)
pred_count = clf_count.predict(X_test_count)

accu1_tfidf = accuracy_score(y_test, pred_tfidf)

accu1_count = accuracy_score(y_test, pred_count)

mse1_tfidf = mean_squared_error(y_test, pred_tfidf)

mse1_count = mean_squared_error(y_test, pred_count)

mae1_tfidf = mean_absolute_error(y_test, pred_tfidf)

mae1_count = mean_absolute_error(y_test, pred_count)

score_tfidf = cross_val_score(clf_tfidf, X_train_tfidf, y_train, cv = 5) #5-fold cross validation
score_count = cross_val_score(clf_count, X_train_count, y_train, cv = 5)

score_tfidf.mean()

score_count.mean()

grid_random_tfidf = {'n_estimators' : [10, 100, 200, 500, 1000, 1200],
                     'bootstrap' : [True, False],
                     'max_depth' : [None, 10, 20, 30, 40, 50],
                     'min_samples_split' : [2, 4, 6],
                     'min_samples_leaf' : [1, 2, 4]}

grid_random_count = {'n_estimators' : [10, 100, 200, 500, 1000, 1200],
                     'bootstrap' : [True, False],
                     'max_depth' : [None, 10, 20, 30, 40, 50],
                     'min_samples_split' : [2, 4, 6],
                     'min_samples_leaf' : [1, 2, 4]}

rs_clf_tfidf = RandomizedSearchCV(estimator = clf_tfidf,
                            param_distributions = grid_random_tfidf,
                            n_iter = 50,
                            cv = 5)

rs_clf_count = RandomizedSearchCV(estimator = clf_count,
                            param_distributions = grid_random_count,
                            n_iter = 50,
                            cv = 5)

rs_clf_tfidf.fit(X_train_tfidf, y_train)
rs_clf_count.fit(X_train_count, y_train)

rs_clf_tfidf.best_params_

rs_clf_count.best_params_

new_pred_tfidf = rs_clf_tfidf.predict(X_test_tfidf)
new_pred_count = rs_clf_count.predict(X_test_count)

accu2_tfidf = accuracy_score(y_test, new_pred_tfidf)

accu2_count = accuracy_score(y_test, new_pred_count)

mse2_tfidf = mean_squared_error(y_test, new_pred_tfidf)

mse2_count = mean_squared_error(y_test, new_pred_count)

mae2_tfidf = mean_absolute_error(y_test, new_pred_tfidf)

mae2_count = mean_absolute_error(y_test, new_pred_count)

param_grid_tfidf = {'n_estimators' : [500, 1000, 1200],
                    'bootstrap' : [False],
                    'max_depth' : [30, 40, 50],
                    'min_samples_split' : [2,4],
                    'min_samples_leaf' : [2,4]} 

param_grid_count = {'n_estimators' : [500, 1000, 1200],
                    'bootstrap' : [False],
                    'max_depth' : [20, 30, 40],
                    'min_samples_split' : [2, 4],
                    'min_samples_leaf' : [2, 4]}

grid_search_tfidf = GridSearchCV(estimator = clf_tfidf, param_grid = param_grid_tfidf, 
                          cv = 5, n_jobs = -1, verbose = 2)

grid_search_count = GridSearchCV(estimator = clf_count, param_grid = param_grid_count, 
                          cv = 5, n_jobs = -1, verbose = 2)

grid_search_tfidf.fit(X_train_tfidf, y_train)
grid_search_count.fit(X_train_count, y_train)

grid_search_tfidf.best_params_

grid_search_count.best_params_

pred2_tfidf = grid_search_tfidf.predict(X_test_tfidf)
pred2_count = grid_search_count.predict(X_test_count)

accu3_tfidf = accuracy_score(y_test, pred2_tfidf)

accu3_count = accuracy_score(y_test, pred2_count)

mse3_tfidf = mean_squared_error(y_test, pred2_tfidf)

mse3_count = mean_squared_error(y_test, pred2_count)

mae3_tfidf = mean_absolute_error(y_test, pred2_tfidf)

mae3_count = mean_absolute_error(y_test, pred2_count)

print(" TFIDF\n initial  : accuracy : {0}, mse : {1}, mae : {2}\n \
randomcv : accuracy : {3}, mse : {4}, mae : {5}\n \
gridcv   : accuracy : {6}, mse : {7}, mae : {8}".format(accu1_tfidf, mse1_tfidf, mae1_tfidf, accu2_tfidf, mse2_tfidf, mae2_tfidf, accu3_tfidf, mse3_tfidf, mae3_tfidf))

print(" COUNT\n initial  : accuracy : {0}, mse : {1}, mae : {2}\n \
randomcv : accuracy : {3}, mse : {4}, mae : {5}\n \
gridcv   : accuracy : {6}, mse : {7}, mae : {8}".format(accu1_count, mse1_count, mae1_count, accu2_count, mse2_count, mae2_count, accu3_count, mse3_count, mae3_count))
