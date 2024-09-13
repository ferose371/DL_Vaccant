import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import random

class PreProcess_Data:
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
        print('Number of train images : {}\n'.format(len(train)))
        print('Number of train images labels: {}\n'.format(len(label)))
        retina_df = pd.DataFrame({'Image': train, 'Labels': label})
        return train, label, retina_df

    def generate_train_test_image(self, train, label):
        retina_df = pd.DataFrame({'Image': train, 'Labels': label})
        train_data, test_data = train_test_split(retina_df, test_size=0.2)
        print(test_data)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            validation_split=0.15
        )
        test_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_dataframe(
            train_data,
            directory='./',
            x_col="Image",
            y_col="Image",  # Use the images themselves as labels
            target_size=(128, 128),
            color_mode="rgb",
            class_mode="input",  # Set class_mode to 'input'
            batch_size=32,
            subset='training'
        )
        validation_generator = train_datagen.flow_from_dataframe(
            train_data,
            directory='./',
            x_col="Image",
            y_col="Image",  # Use the images themselves as labels
            target_size=(128, 128),
            color_mode="rgb",
            class_mode="input",  # Set class_mode to 'input'
            batch_size=32,
            subset='validation'
        )
        test_generator = test_datagen.flow_from_dataframe(
            test_data,
            directory='./',
            x_col="Image",
            y_col="Image",  # Use the images themselves as labels
            target_size=(128, 128),
            color_mode="rgb",
            class_mode="input",  # Set class_mode to 'input'
            batch_size=32
        )
        return train_generator, test_generator, validation_generator


    def plot_history(self, history):
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show(block=True)

        plt.plot(history.history['accuracy'], label='train_loss')
        plt.plot(history.history['val_accuracy'], label='val_loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
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
data_processor = PreProcess_Data()
data_processor.display_random_images(train_folder, n_rows=5, n_cols=5)



# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from keras.preprocessing.image import ImageDataGenerator
#
# class PreProcessData:
#     def __init__(self, image_size=(128, 128)):
#         self.image_size = image_size
#
#     def preprocess(self, dir_path):
#         train = []
#         label = []
#         for i, class_name in enumerate(os.listdir(dir_path)):
#             class_path = os.path.join(dir_path, class_name)
#             for img_name in os.listdir(class_path):
#                 img_path = os.path.join(class_path, img_name)
#                 train.append(img_path)
#                 label.append(i)
#         print('Number of train images:', len(train))
#         print('Number of train image labels:', len(label))
#         return train, label
#
#     def generate_train_test_data(self, train, label, test_size=0.2):
#         X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=test_size, random_state=42)
#         return (X_train, y_train), (X_test, y_test)
#
#     def create_image_data_generators(self, train_data, test_data, train_df, test_df, batch_size=32):
#         train_datagen = ImageDataGenerator(rescale=1. / 255)
#         test_datagen = ImageDataGenerator(rescale=1. / 255)
#
#         train_generator = train_datagen.flow_from_dataframe(
#             dataframe=train_df,
#             x_col='Image',
#             y_col='Labels',
#             target_size=self.image_size,
#             batch_size=batch_size,
#             class_mode='input'
#         )
#
#         test_generator = test_datagen.flow_from_dataframe(
#             dataframe=test_df,
#             x_col='Image',
#             y_col='Labels',
#             target_size=self.image_size,
#             batch_size=batch_size,
#             class_mode='input'
#         )
#
#         return train_generator, test_generator
#
#     def display_random_images(self, dir_path, n_rows, n_cols):
#         images = []
#         fig = plt.figure(figsize=(10, 10))
#
#         for class_name in os.listdir(dir_path):
#             class_path = os.path.join(dir_path, class_name)
#             img_name = np.random.choice(os.listdir(class_path))
#             img_path = os.path.join(class_path, img_name)
#             images.append(img_path)
#
#         for i in range(1, n_rows * n_cols + 1):
#             img = plt.imread(images[i - 1])
#             plt.subplot(n_rows, n_cols, i)
#             plt.imshow(img)
#             plt.axis('off')
#
#         plt.show()
