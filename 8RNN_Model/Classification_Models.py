from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout, Flatten

class RNNModel:
    def rnn_model(self, input_shape, num_classes, optimizer='adam'):
        model = Sequential()
        model.add(SimpleRNN(64, input_shape=input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
