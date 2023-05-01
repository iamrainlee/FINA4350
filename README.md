# FINA4350
## Predicting Rate Hikes with FOMC and FED data - a NLP and ML approach

## About
This project is a project for FINA4350 to predict rate hike with textual data from official sources. The sources include FOMC statements, FOMC minutes, FED speeches and FED testimonies. We apply NLP (Natural Language Processing) techniques and ML (Machine Learning) models on the data to train models for prediction.

As our aim is to predict rate hike, all of the textual data for prediction are released before the rate hike is announced. During merging, we separated into 3 datasets:
- Rate Hike with FOMC statements and minutes
- Rate Hike with FED speeches and testimonies
- Rate Hike with FOMC statements, FOMC minutes, FED speeches and FED testimonies (Full)

The data here are collected until April 2023, if you wish to train the model with more data, you could run the Data Collection, Preprocessing and Merging programs to collect new data.

For NLP and ML, we have applied various techniques and trained various models. For machine learning models, we applied count vectoriser and term frequency–inverse document frequency (tf-idf) to transform data before machine learning. Then, we train 3 ML models, namely Support Vector Machine (SVM), Random Forest Classifier (RF) and k Nearest Neighbour (kNN). For Deep Learning, we apply pre-trained GloVe (Global Vector for Word Representation) and combine with a 3-layer LSTM model. We also trained another model by applying pre-trained BERT (Bidirectional Encoder Representations from Transformers) and combine with CNN model.

## Results
The full dataset has the highest accuracy across the 3 datasets in most of the model, so the following results correspond to the full dataset.

| Model        | Accuracy (Testing Data) |
|--------------|-------------------------|
| SVM          | 71.74%                  |
| kNN          | 71.74%                  |
| RF           | 73.91%                  |
| LSTM - GLoVe | 71.74%                  |
| CNN - BERT   | 71.74%                  |

## How to use
First download the repository or clone the repository. Then, install the required libraries by

    pip install -r requirements.txt
  
#### Data Collection:

    cd "Data Collection"
    python3 Minutes.py
    python3 fomc_testimony.py
    python3 speeches.py
    python3 Statements.py

Please note that rate hike data are downloaded rather than collected.

#### Data Preprocessing and Topic Modelling:
Please return to the base directory (FINA4350) before running following codes.
Please note that speeches_precessing.py must be run before speeches_topic_classification.py

    cd "Data Collection"
    python3 fomc_testimony_preprocessing.py
    python3 minutes.py
    python3 ratehikes_processing.py
    python3 speeches_preprocessing.py
    python3 speeches_topic_classification.py
    python3 statement_preprocessed.py
    
#### Data Merging:

    cd "Data Merging"
    python3 merging_rate_FOMC.py
    python3 merging_rate_speeches_testimony.py
    python3 merging_rate_FOMC_speeches_testimony.py

#### Machine Learning:
Please return to the base directory (FINA4350) before running following codes.

    cd "Machine learning models"
    
For SVM, RF or kNN models:

SVM: **SVM.py**, RF: **Random_forest.py**, kNN: **knn.py**

E.g.

    python3 SVM.py
    
>Please choose the data : **merging_rate_FOMC**
>#You can choose from merging_rate_FOMC, merging_rate_speeches_testimony or merging_rate_FOMC_speeches_testimony

For Deep Learning models (LSTM with Glove, CNN with BERT):

LSTM with Glove: **LSTM_Glove.py**, CNN with BERT: **CNN.py**

    python3 [file] [data]
    
[data] could be "../Data/Merged Data/rate_FOMC.csv", "../Data/Merged Data/rate_speeches_testimony.csv" or "../Data/Merged Data/rate_FOMC_speeches_testimony.csv"

E.g.

    python3 LSTM_Glove.py "../Data/Merged Data/rate_FOMC.csv"
    
