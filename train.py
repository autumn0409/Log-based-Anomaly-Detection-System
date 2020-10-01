import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Masking

from Attention import Attention
from dataloader import DataGenerator


# hyper-parameters
EMBEDDING_DIM = 768
batch_size = 32
epochs = 20
rnn_units = 256


# load data
training_data = np.load(
    'preprocessed_data/training_data.npz', allow_pickle=True)
x_train = training_data['x']
y_train = training_data['y']
del training_data


# model
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(None, EMBEDDING_DIM)))
model.add(Bidirectional(LSTM(rnn_units, return_sequences=True)))
model.add(Attention(bias=False))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop', metrics=['acc'])
print(model.summary())

# train
train_generator = DataGenerator(x_train, y_train, batch_size=batch_size)
model.fit(train_generator, epochs=epochs)

# save model
model.save('my_model.h5')
