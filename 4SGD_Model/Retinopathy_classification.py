import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import Data_preprocess as dp
import Classification_Models as cm
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":
    images_folder_path = 'D:\\3rd YEAR\\2nd SEM\\Deep Lerning\\DL Project\\dataset\\train'
    imdata = dp.PreProcess_Data()
    # Remove the following line since visualization_images method is not defined
    # imdata.visualization_images(images_folder_path, 3)
    train, label, image_df = imdata.preprocess(images_folder_path)
    print(label)
    train_generator, test_generator, validate_generator = imdata.generate_train_test_image(train, label)

    image_df.to_csv("image_df.csv")
    AnnModel = cm.DeepANN()

    # Use the optimized_model function with Adam optimizer
    Model1 = AnnModel.optimized_model(optimizer=Adam())

    print("train generator", train_generator)
    ANN_history = Model1.fit(train_generator, epochs=2, validation_data=validate_generator)
    Ann_test_loss, Ann_test_acc = Model1.evaluate(test_generator)
    print(f'Test Accuracy: {Ann_test_acc}')

    # Save the model as a .h5 file
    model_filename = "D:\\3rd YEAR\\2nd SEM\\Deep Lerning\\DL Project\\VaccantParking\\VaccantParking\\Vaccant-Parking-GUI-main\\sgd_model.h5"
    Model1.save(model_filename)
    print(f"Model saved as {model_filename}")

    print("The optimized rmsprop architecture is")
    print(Model1.summary())
    print("plot the graph")
    imdata.plot_history(ANN_history)
