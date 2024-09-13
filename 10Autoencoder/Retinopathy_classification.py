import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.utils import plot_model
import Data_preprocess as dp
import Classification_Models as am

if __name__ == "__main__":
    images_folder_path = 'D:\\3rd YEAR\\2nd SEM\\Deep Lerning\\DL Project\\dataset\\train'
    imdata = dp.PreProcess_Data()
    train, label, image_df = imdata.preprocess(images_folder_path)
    print(label)
    train_generator, test_generator, validate_generator = imdata.generate_train_test_image(train, label)

    image_df.to_csv("image_df.csv")
    AutoencoderModel = am.SimpleAutoencoder()  # Instantiate your autoencoder model class
    Model1 = AutoencoderModel.build_model()
    print("train generator", train_generator)
    autoencoder_history = Model1.fit(train_generator, epochs=1, validation_data=validate_generator)
    autoencoder_test_loss = Model1.evaluate(test_generator)
    print(f'Test Loss:{autoencoder_test_loss}')
    Model1.save("my_autoencoder_model.h5")

    model_save_path = "autoencoder_model.h5"
    Model1.save(model_save_path)

    print(f"Model saved to {model_save_path}")

    # Plot training history
    plt.figure(figsize=(12, 6))

    # Plot training & validation loss values
    plt.subplot(1, 1, 1)
    plt.plot(autoencoder_history.history['loss'])
    plt.plot(autoencoder_history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.show()

    print("The Autoencoder architecture is")
    print(Model1.summary())
