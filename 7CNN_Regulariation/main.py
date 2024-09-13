import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Data_preprocess import PreProcess_Data
import Data_preprocess as dp
from Classification_Models import VacantParkingSpotDetector

def visualize_image_counts(main):
    data = dict()

    for i in os.listdir(main):
        sub_dir = os.path.join(main, i)
        count = len(os.listdir(sub_dir))
        data[i] = count

    keys = data.keys()
    values = data.values()

    colors = ["red" if x <= 150 else "green" for x in values]

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(values))
    plt.barh(y_pos, values, align='center', color=colors)
    for i, v in enumerate(values):
        ax.text(v + 1.4, i - 0.25, str(v), color=colors[i])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(keys)
    ax.set_xlabel('Images', fontsize=16)
    plt.xticks(color='black', fontsize=13)
    plt.yticks(fontsize=13)
    plt.show()

if __name__ == "__main__":
    images_folder_path = 'D:\\3rd YEAR\\2nd SEM\\Deep Lerning\\DL Project\\dataset\\train'
    imdata = PreProcess_Data()
    # imdata.visualization_images(images_folder_path, 2)
    train, label, image_df = imdata.preprocess(images_folder_path)
    print(label)
    train_generator, test_generator, validate_generator = imdata.generate_train_test_image(train, label)

    image_df.to_csv("image_df.csv")
    CNNModel = VacantParkingSpotDetector()
    Model1 = CNNModel.create_model()
    print("train generator", train_generator)
    CNN_history = Model1.fit(train_generator, epochs=2, validation_data=validate_generator)
    CNN_test_loss, CNN_test_acc = Model1.evaluate(test_generator)
    print(f'Test Accuracy: {CNN_test_acc}')
    Model1.save(f"my_cnn_model.keras")  # Saving the model

    # Visualize image counts
    visualize_image_counts(images_folder_path)

    # Save the model to a specified path
    model_save_path = "D:\\3rd YEAR\\2nd SEM\\Deep Lerning\\DL Project\\VaccantParking\\VaccantParking\\Vaccant-Parking-GUI-main\\my_cnn-regularization_model.h5"
    Model1.save(model_save_path)

    print(f"Model saved to {model_save_path}")

    print("The CNN Regularization architecture is")
    print(Model1.summary())
    print("Plot the graph")
    imdata.plot_history(CNN_history)
