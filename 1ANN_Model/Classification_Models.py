from keras import Sequential
from keras.layers import Flatten, Dense


class DeepANN():
    def simple_model(self):
       model = Sequential()
       model.add(Flatten(input_shape=(128, 128, 3)))
       model.add(Dense(128, activation="relu"))
       model.add(Dense(64, activation="relu"))
       model.add(Dense(2, activation="softmax"))
       model.compile(loss="categorical_crossentropy",
                     optimizer="sgd",
                     metrics=["accuracy"])
       return model
