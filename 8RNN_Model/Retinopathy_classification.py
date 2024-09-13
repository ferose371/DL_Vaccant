# RetinopathyClassification.py
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from Data_preprocess import DataPreprocess
from Classification_Models import RNNModel

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

def main():
    images_folder_path = 'D:\\3rd YEAR\\2nd SEM\\Deep Lerning\\DL Project\\dataset\\train'
    imdata = DataPreprocess()

    # Visualize sample images
    # imdata.visualization_images(images_folder_path,2)

    # Preprocess data
    train, label, image_df = imdata.preprocess(images_folder_path)

    # Determine the number of classes
    num_classes = len(np.unique(label))

    # Generate train and test sequences
    train_sequences, test_sequences, train_labels, test_labels = imdata.generate_train_test_sequence(train, label)

    # Save the preprocessed dataframe
    image_df.to_csv("image_df.csv")

    # Create an instance of the RNN model with the correct number of classes
    RNNModelInstance = RNNModel()
    RNN_Model = RNNModelInstance.rnn_model(input_shape=(128, 128), num_classes=num_classes)

    print("Train sequences shape:", train_sequences.shape)

    # Train the RNN model
    RNN_history = RNN_Model.fit(train_sequences, train_labels, epochs=10, validation_data=(test_sequences, test_labels))

    # Evaluate the RNN model on the test data
    RNN_test_loss, RNN_test_acc = RNN_Model.evaluate(test_sequences, test_labels)
    print(f'Test Accuracy: {RNN_test_acc}')

    # Save the RNN model
    RNN_Model.save("my_rnn_model.keras")

    # # Save the model in .h5 format
    # model_save_path = "D:\\3rd YEAR\\2nd SEM\\Deep Lerning\\DL Project\\VaccantParking\\VaccantParking\\Vaccant-Parking-GUI-main\\rnn_model.h5"
    # RNN_Model.save(model_save_path)
    # print(f"Model saved to {model_save_path}")

    # Visualize image counts
    visualize_image_counts(images_folder_path)

    print("The RNN architecture is:")
    print(RNN_Model.summary())

    # Plot the training history
    imdata.plot_history(RNN_history)




if __name__ == "__main__":
    main()
