from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Reshape

class LSTMModel:
    def lstm_model(self, input_shape=(None, 128, 128, 3), units=100, optimizer='adam'):
        model = Sequential()
        model.add(Reshape((-1, 128, 128, 3), input_shape=input_shape))
        model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(units, return_sequences=True))  # Add return_sequences=True
        model.add(LSTM(units))  # Add another LSTM layer
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
