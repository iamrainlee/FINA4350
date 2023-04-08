import pandas as pd
import sys
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import collections

csv.field_size_limit(sys.maxsize)

# load in preprocessed data
df = pd.read_csv("../Data/Preprocessed Data/speeches_preprocessed.csv",engine='python',converters={"Date":pd.to_datetime,"Content":str})

doc = df.copy()

#remove fullstops
doc['Content'] = doc['Content'].str.replace('.','')

#vectorize the content for topic clustering
vectorizer = TfidfVectorizer(stop_words='english',\
                             max_features= 1000,
                             max_df = 0.5, 
                             smooth_idf=True)

X = vectorizer.fit_transform(doc['Content'])
 
 # fit SVD model for topic clustering
svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)
lsd = svd_model.fit_transform(X)

terms = vectorizer.get_feature_names_out()

#print out terms to check which topics are relevant
for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:15]
    print("Topic "+str(i)+": ")
    for t in sorted_terms:
        print(t[0])
    print(" ")

#classify topics for each row
topics = lsd.argmax(axis=1)

print(collections.Counter(topics))

#save the clustering results
df['topic'] = topics.tolist()
df.to_csv("../Data/Preprocessed Data/speeches_with_topics.csv",index=False)

#Topic 0 and 1 are most relevant topics
df = df[(df['topic']==0)|(df['topic']==1)] 
df.to_csv("../Data/Preprocessed Data/speeches_topics_0_1.csv",index=False)