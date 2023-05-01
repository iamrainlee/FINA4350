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
The full dataset has the highest accuracy, so the following results correspond to the full dataset.

| Model        | Accuracy (Testing Data) |
|--------------|-------------------------|
| SVM          | 66.18%                  |
| kNN          | 68.61%                  |
| RF           | 66.07%                  |
| LSTM - GLoVe | 65.27%                  |
| CNN - BERT   | 63.80%                  |
