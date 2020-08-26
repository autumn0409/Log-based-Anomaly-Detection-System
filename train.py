from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from Attention import Attention


# hyper-parameters
batch_size = 16
epochs = 20

timesteps = 20
vec_len = 300
rnn_units = 32

# model
model = Sequential()
model.add(Bidirectional(LSTM(rnn_units, return_sequences=True),
                        input_shape=(timesteps, vec_len)))
model.add(Attention(bias=False))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())


# train
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

model.save('my_model.h5')
