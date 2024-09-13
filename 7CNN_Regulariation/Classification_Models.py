from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class VacantParkingSpotDetector():
    def create_model(self, input_shape=(128, 128, 3), num_classes=2):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        # Compile the model with appropriate loss and metrics
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

# Create an instance of the vacant parking spot detector
detector = VacantParkingSpotDetector()

# Create the vacant parking spot detection model
vacant_spot_detection_model = detector.create_model()

# Summary of the model architecture
vacant_spot_detection_model.summary()
