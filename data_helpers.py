import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence

def load_data(dataset):
    if dataset == 'imdb':
        return load_imdb()
    elif dataset == 'polaritydata':
        return load_polaritydata()
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)

def load_imdb(max_features=5000, max_len=400):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path='imdb.npz',num_words=max_features)

    x_train = sequence.pad_sequences(x_train, maxlen=max_len, padding='post')
    x_test = sequence.pad_sequences(x_test, maxlen=max_len, padding='post')

    y_train = [[1, 0] if y == 0 else [0, 1] for y in y_train]
    y_test = [[1, 0] if y == 0 else [0, 1] for y in y_train]

    X = np.concatenate((np.array(x_train), np.array(x_test)))
    Y = np.concatenate((np.array(y_train), np.array(y_test)))

    return (X, Y), (X[len(X)*9/10:], Y[len(X)*9/10:])

def load_polaritydata():
    return ''