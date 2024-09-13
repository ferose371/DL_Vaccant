# DataPreprocess.py
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataPreprocess:
    # def visualization_images(self, dir_path, nimages):
    #     fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    #     dpath = dir_path
    #     count = 0
    #     for i in os.listdir(dpath):
    #         train_class = os.listdir(os.path.join(dpath, i))
    #         for j in range(nimages):
    #             img = os.path.join(dpath, i, train_class[j])
    #             img = cv2.imread(img)
    #             axs[count][j].title.set_text(i)
    #             axs[count][j].imshow(img)
    #         count += 1
    #     fig.tight_layout()
    #     plt.show(block=True)

    def preprocess(self, dir_path):
        dpath = dir_path
        train = []
        label = []
        for i in os.listdir(dpath):
            train_class = os.listdir(os.path.join(dpath, i))
            for j in train_class:
                img = os.path.join(dpath, i, j)
                train.append(img)
                label.append(i)
        print('Number of train images : {} \n'.format(len(train)))
        print('Number of train images labels: {} \n'.format(len(label)))
        retina_df = pd.DataFrame({'Image': train, 'Labels': label})
        return train, label, retina_df

    def generate_train_test_sequence(self, train, label):
        retina_df = pd.DataFrame({'Image': train, 'Labels': label})
        train_data, test_data = train_test_split(retina_df, test_size=0.2)

        train_sequences = []
        for img_path in train_data['Image']:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
            img = cv2.resize(img, (128, 128))  # Resize to (128, 128)
            train_sequences.append(img)

        test_sequences = []
        for img_path in test_data['Image']:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))  # Resize to (128, 128)
            test_sequences.append(img)

        # Convert lists to numpy arrays
        train_sequences = np.array(train_sequences)
        test_sequences = np.array(test_sequences)

        train_labels = pd.get_dummies(train_data['Labels'])
        test_labels = pd.get_dummies(test_data['Labels'])

        print(f"Train sequences shape: {train_sequences.shape}")
        print(f"Test sequences shape: {test_sequences.shape}")

        return train_sequences, test_sequences, train_labels, test_labels

    def plot_history(self, history):
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show(block=True)

        plt.plot(history.history['accuracy'], label='train_accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show(block=True)

    def display_random_images(self, dir_path, n_rows, n_cols, image_size=(256, 256)):
        images = []

        for food_folder in sorted(os.listdir(dir_path)):
            food_items = os.listdir(os.path.join(dir_path, food_folder))
            if food_items:  # Check if there are items in the folder
                food_selected = np.random.choice(food_items)
                images.append(os.path.join(dir_path, food_folder, food_selected))

        fig = plt.figure(figsize=(20, 20))

        for i, image_path in enumerate(images):
            img = plt.imread(image_path)
            if img is not None:
                a, b, c = img.shape
                ax = fig.add_subplot(n_rows, n_cols, i + 1)
                ax.imshow(img, extent=[0, b, 0, a])
                category = image_path.split(os.path.sep)[-2]
                ax.text(0.5, -0.4, category, size=15, ha="center", transform=ax.transAxes)

        plt.tight_layout()
        plt.show()

        # Example usage

train_folder = "D:\\3rd YEAR\\2nd SEM\\Deep Lerning\\DL Project\\dataset\\train"
data_processor = DataPreprocess()
data_processor.display_random_images(train_folder, n_rows=5, n_cols=5)
