# RetinopathyClassification.py
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from Data_preprocess import PreProcess_Data
from Classification_Models import LSTMModel

def main():
    images_folder_path = 'D:\\3rd YEAR\\2nd SEM\\Deep Lerning\\DL Project\\dataset\\train'
    imdata = PreProcess_Data()

    # Visualize sample images
    imdata.display_random_images(images_folder_path, 3, 3)

    # Preprocess data
    train, label, image_df = imdata.preprocess(images_folder_path)

    # Generate train and test generators
    train_generator, test_generator, validation_generator = imdata.generate_train_test_image(train, label)

    # Save the preprocessed data frame
    image_df.to_csv("image_df.csv")

    # Create an instance of the LSTM model
    LSTMModelInstance = LSTMModel()
    LSTM_Model = LSTMModelInstance.lstm_model()

    # Train the LSTM model
    LSTM_history = LSTM_Model.fit(train_generator, epochs=2, validation_data=validation_generator)

    # Evaluate the LSTM model on the test data
    LSTM_test_loss, LSTM_test_acc = LSTM_Model.evaluate(test_generator)
    print(f'Test Accuracy: {LSTM_test_acc}')

    # # Save the LSTM model
    # model_save_path = "D:\\3rd YEAR\\2nd SEM\\Deep Lerning\\DL Project\\VaccantParking\\VaccantParking\\Vaccant-Parking-GUI-main\\lstm_model.h5"
    # LSTM_Model.save(model_save_path)
    # print(f"Model saved to {model_save_path}")

    # Print model summary
    print("The LSTM architecture is:")
    print(LSTM_Model.summary())

    # Plot the training history
    imdata.plot_history(LSTM_history)

if __name__ == "__main__":
    main()
