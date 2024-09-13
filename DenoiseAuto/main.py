import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error

# Path to your dataset
dataset_path = "D:\\3rd YEAR\\2nd SEM\\Deep Lerning\\DL Project\\dataset\\train"

# Using ImageDataGenerator to load and preprocess the images
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)  # Assuming 20% validation split

# Training dataset
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),  # Change target input size to 128x128
    batch_size=32,
    class_mode='binary',  # Use 'binary' as class_mode for two classes
    subset='training')  # Use subset 'training' for training data

# Validation dataset
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),  # Change target input size to 128x128
    batch_size=32,
    class_mode='binary',  # Use 'binary' as class_mode for two classes
    subset='validation')  # Use subset 'validation' for validation data


# Add Gaussian noise to the images
def add_noise(images, noise_factor=0.5):
    noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    noisy_images = np.clip(noisy_images, 0., 1.)
    return noisy_images


# Build the denoising autoencoder model
def build_denoising_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    encoded = Conv2D(32, 3, activation='relu', padding='same')(x)

    # Decoder
    x = Conv2D(32, 3, activation='relu', padding='same')(encoded)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    decoded = Conv2D(3, 3, activation='sigmoid', padding='same')(x)  # Output image has 3 channels (RGB)

    model = Model(inputs, decoded)
    return model


# Instantiate the model
input_shape = (128, 128, 3)  # Assuming input images are 128x128 RGB images
denoising_autoencoder = build_denoising_autoencoder(input_shape)

# Compile the model
denoising_autoencoder.compile(optimizer='adam', loss='mse')

# Print model summary
denoising_autoencoder.summary()

# Train the model
noise_factor = 0.5  # Adjust noise factor as needed
noisy_train_data = add_noise(train_generator.next()[0], noise_factor)
noisy_val_data = add_noise(validation_generator.next()[0], noise_factor)

# Train for 2 epochs
history = denoising_autoencoder.fit(noisy_train_data, train_generator.next()[0], epochs=100,
                                    validation_data=(noisy_val_data, validation_generator.next()[0]))

# Save the model
denoising_autoencoder.save("denoising_autoencoder_model.h5")

# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on validation data
reconstructed_images = denoising_autoencoder.predict(noisy_val_data)

# Calculate Mean Squared Error (MSE) between original clean images and reconstructed images
mse = mean_squared_error(validation_generator.next()[0], reconstructed_images)

# Calculate accuracy (lower MSE means higher accuracy)
accuracy = 1.0 / (1.0 + mse)

print("Total Accuracy:", accuracy)
