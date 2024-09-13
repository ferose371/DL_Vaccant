from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Conv2DTranspose, UpSampling2D, Add

class ResidualAutoencoder():
    def __init__(self, input_shape=(128, 128, 3), latent_dim=64, optimizer='adam'):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.optimizer = optimizer

    def build_model(self):
        model = Sequential()

        # Encoder
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))

        # Flatten before latent space
        model.add(Flatten())

        # Latent space
        model.add(Dense(self.latent_dim, activation='relu'))

        # Decoder
        model.add(Dense(8 * 8 * 128, activation='relu'))
        model.add(Reshape((8, 8, 128)))

        # Decoder with residual connections
        model.add(Conv2DTranspose(64, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))

        # Add the residual connection here
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same'))
        model.add(UpSampling2D((2, 2)))

        # Adding skip connections
        model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))

        # Another convolutional layer to match the number of filters in the last decoder layer
        model.add(Conv2DTranspose(3, (3, 3), activation='relu', padding='same'))

        model.compile(loss='mse', optimizer=self.optimizer)
        return model

# Instantiate the ResidualAutoencoder
residual_autoencoder = ResidualAutoencoder()

# Build the model
residual_autoencoder_model = residual_autoencoder.build_model()

# Print the summary
residual_autoencoder_model.summary()
