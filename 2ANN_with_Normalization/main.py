import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import Data_preprocess as dp
import Classification_Models as cm

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
    imdata = dp.PreProcess_Data()
    imdata.display_random_images(images_folder_path, 3, 3)
    train, label, image_df = imdata.preprocess(images_folder_path)
    print(label)
    train_generator, test_generator, validate_generator = imdata.generate_train_test_image(train, label)

    # Visualize image counts
    visualize_image_counts(images_folder_path)

    image_df.to_csv("image_df.csv")
    AnnModel = cm.DeepANN()
    Model1 = AnnModel.normalized_model()
    print("train generator", train_generator)
    ANN_history=Model1.fit(train_generator,epochs=2,validation_data=validate_generator)
    Ann_test_loss, Ann_test_acc = Model1.evaluate(test_generator)
    print(f'Test Accuracy:{Ann_test_acc}')

    # Save the model
    model_save_path = "D:\\3rd YEAR\\2nd SEM\\Deep Lerning\\DL Project\\VaccantParking\\VaccantParking\\Vaccant-Parking-GUI-main\\ann_normalized_model.h5"
    Model1.save(model_save_path)

    print(f"Model saved to {model_save_path}")

    print("The ann architecture is")
    print(Model1.summary())
    print("plot the graph")
    imdata.plot_history(ANN_history)
