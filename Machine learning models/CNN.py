import sys
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow_text as text
import tensorflow as tf
import tensorflow_hub as hub

#allow import of csv of large file size while preventing interger overflow error
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

def get_sentence_embeding(sentences):
    preprocessed_text = bert_preprocess(sentences)
    return bert_encoder(preprocessed_text)['pooled_output']

if __name__ == "__main__":

    #This part is for checking the argument input
    if len(sys.argv) != 2:
        print("""Usage of LSTM_Glove.py:
    python3 LSTM_Glove.py [data]
    [data] can be:
    ../Data/Merged Data/rate_FOMC.csv
    ../Data/Merged Data/rate_speeches_testimony.csv
    ../Data/Merged Data/rate_FOMC_speeches_testimony.csv
    """)
        exit()
    else:
        if "rate_FOMC" not in sys.argv[1] and "rate_speeches_testimony" not in sys.argv[1] and "rate_FOMC_speeches_testimony" not in sys.argv[1]:
            print("""Usage of LSTM_Glove.py:
python3 LSTM_Glove.py [data]
[data] can be:
../Data/Merged Data/rate_FOMC.csv
../Data/Merged Data/rate_speeches_testimony.csv
../Data/Merged Data/rate_FOMC_speeches_testimony.csv
""")
            exit()
        try:
            data = pd.read_csv(sys.argv[1])
        except:
            print("The path to data is invalid. Please check")
            exit()

    
    print("using data:",sys.argv[1])

    #remove full stop
    data['data'] = data.data.str.replace('.','')
    
    #separate data into x,y and apply one-hot encoding to y
    X = np.array(data['data'])
    y = pd.get_dummies(data['rate_hike']).values

    X_train, X_test,Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state = 43)

    # Download pretrained BERT models
    bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

    # Bert layers
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)

    outputs = outputs['pooled_output'][...,None]

    # Neural network layers
    conv = tf.keras.layers.Conv1D(32, 10, activation='relu')(outputs)
    conv_pool = tf.keras.layers.GlobalMaxPooling1D()(conv)
    l = tf.keras.layers.Dropout(0.1, name="dropout")(conv_pool)
    l = tf.keras.layers.Dense(20, activation='relu')(l)
    l = tf.keras.layers.Dense(Y_train.shape[1], activation='sigmoid', name="output")(l)

    # Use inputs and outputs to construct a final model
    model = tf.keras.Model(inputs=[text_input], outputs = [l])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=10, batch_size = 32, validation_split=0.1)

    print("Training completed")
    print("Training accuracy:",model.evaluate(X_train, Y_train)[1])

    #evaluate model
    print('Testing accuracy:',model.evaluate(X_test, Y_test)[1])