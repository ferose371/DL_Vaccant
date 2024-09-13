from keras import Sequential
from keras.layers import Flatten, Dense, BatchNormalization

class DeepANN():
    def normalized_model(self, input_shape=(128, 128, 3), optimizer='sgd'):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))

        model.add(Dense(128, activation="relu"))
        model.add(BatchNormalization())

        model.add(Dense(64, activation="relu"))
        model.add(BatchNormalization())

        model.add(Dense(2, activation="softmax"))

        model.compile(loss="binary_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])
        return model

# Example usage
vacant_parking_model = DeepANN().normalized_model()
