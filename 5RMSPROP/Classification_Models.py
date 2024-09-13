from keras import Sequential
from keras.layers import Flatten, Dense


class DeepANN():
    def optimized_model(self, input_shape=(128, 128, 3), optimizer='rmsprop'):  # Change the function name

        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(2, activation="softmax"))

        # Use the optimizer passed as a parameter
        model.compile(loss="categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])
        return model
