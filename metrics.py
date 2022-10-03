import numpy as np

def accuracy(y, predictions):
    return np.sum(y == np.argmax(predictions, axis=1)) / y.shape[0]