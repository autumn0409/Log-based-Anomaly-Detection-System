import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Masking
from keras.preprocessing.sequence import pad_sequences

from Attention import Attention


# hyper-parameters
batch_size = 64
epochs = 20
rnn_units = 32


# load data
training_data = np.load(
    'preprocessed_data/training_data.npz', allow_pickle=True)
x_train = training_data['x_train']
y_train = training_data['y_train']


# padding
x_train = [i.tolist() for i in x_train]
x_train = pad_sequences(x_train, padding='post', dtype='float32')
x_train = np.array(x_train)


# model
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(None, x_train.shape[2])))
model.add(Bidirectional(LSTM(rnn_units, return_sequences=True)))
model.add(Attention(bias=False))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
print(model.summary())


# train
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)


# save model
model.save('my_model.h5')
