import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
import math


EMBEDDING_DIM = 768


class DataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size):
        'Initialization'
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        'Denotes the number of batches'
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        y = self.y[index * self.batch_size:(index + 1) * self.batch_size]

        x = pad_sequences(x, dtype='object', padding='post',
                          value=np.zeros(EMBEDDING_DIM)).astype(np.float32)

        return x, y
