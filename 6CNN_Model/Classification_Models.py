from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class VacantParkingSpotDetector():
    def create_model(self, input_shape=(128, 128, 3), num_classes=2):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))  # Adjusted dense layer size
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='sigmoid'))  # Using sigmoid for binary classification

        # Compile the model with appropriate loss and metrics
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',  # Using Adam optimizer for better convergence
                      metrics=['accuracy'])
        return model

# Create an instance of the vacant parking spot detector
detector = VacantParkingSpotDetector()

# Create the vacant parking spot detection model
vacant_spot_detection_model = detector.create_model()

# Summary of the model architecture
vacant_spot_detection_model.summary()
