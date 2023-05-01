import pandas as pd
import numpy as np
from keras.layers import LSTM, Activation, Dropout, Dense, Input
from keras.layers import Embedding
from keras.models import Model
import string
import re
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from keras.utils import pad_sequences
import keras
from sklearn.model_selection import train_test_split
import sys
import csv
import os
import wget

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #remove debug info

#allow import of csv of large file size while preventing interger overflow error
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

def read_glove_vector(glove_vec):
    """
    This function is to read the pretrained GloVe and convert into a vector for later transformation
    """
    with open(glove_vec, 'r', encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            w_line = line.split()
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
    return word_to_vec_map

def rate_predict(input_shape,output_shape):
    """
    This is to generate the LSTM model.
    There are 3 LSTM layers with dropout layer in between, and a fully connected layer at last for classification
    """

    X_indices = Input(input_shape)

    embeddings = embedding_layer(X_indices)

    X = LSTM(256, return_sequences=True)(embeddings)

    X = Dropout(0.6)(X)

    X = LSTM(128, return_sequences=True)(X)

    X = Dropout(0.6)(X)

    X = LSTM(128)(X)

    X = Dense(output_shape, activation='sigmoid')(X)

    model = Model(inputs=X_indices, outputs=X)

    return model

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
    
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    words_to_index = tokenizer.word_index

    #check if Glove library is downloaded
    if not os.path.exists('Library/glove.6B.50d.txt'):
        if not os.path.exists("Library"):
            os.mkdir('Library')
        url = 'https://www.dropbox.com/s/net9dyagiwskupp/glove.6B.50d.txt?dl=1'
        wget.download(url, out='Library/glove.6B.50d.txt')
    
    word_to_vec_map = read_glove_vector('Library/glove.6B.50d.txt') #read the glove vector

    maxLen = 300

    vocab_len = len(words_to_index)
    embed_vector_len = word_to_vec_map['moon'].shape[0]

    #create embedding of Glove for the data set
    emb_matrix = np.zeros((vocab_len, embed_vector_len))

    for word, index in words_to_index.items():
        embedding_vector = word_to_vec_map.get(word)
        if embedding_vector is not None:
            index -= 1
            emb_matrix[index, :] = embedding_vector

    embedding_layer = Embedding(input_dim=vocab_len, output_dim=embed_vector_len, input_length=maxLen, weights = [emb_matrix], trainable=False)
    
    X_train_indices = tokenizer.texts_to_sequences(X_train)

    X_train_indices = pad_sequences(X_train_indices, maxlen=maxLen, padding='post')

    model = rate_predict(maxLen,Y_train.shape[1]) #create machine learning model

    adam = keras.optimizers.Adam(learning_rate = 0.00005) #add optimizer for model
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    #train model
    model.fit(X_train_indices, Y_train, batch_size=32, epochs=10, validation_split=0.1)
    
    print("Training completed")
    print("Training accuracy:",model.evaluate(X_train_indices, Y_train)[1])

    #transform testing data and evaluate model
    X_test_indices = tokenizer.texts_to_sequences(X_test)

    X_test_indices = pad_sequences(X_test_indices, maxlen=maxLen, padding='post')

    print('Testing accuracy:',model.evaluate(X_test_indices, Y_test)[1])