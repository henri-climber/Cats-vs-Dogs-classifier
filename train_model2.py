import math
import random

import numpy as np
import cv2
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_WIDTH, IMG_HEIGHT = 64, 64
TEST_SIZE = 0.3
EPOCHS = 10


def load_cats_dogs(dataset, path, count=None):
    """
    :param dataset: dataset (directory name) to load data from
    :param path: path to the directory where dataset lays in
    :param count: number of images to load from dataset. loads all images by default
    :return: an np.array() with all the images and an np.array() with all the corresponding labels to the images
    """
    if count is None:
        count = math.inf

    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []
    for label in labels:
        n = 0
        for file in os.listdir(os.path.join(path, dataset, label))[1:]:
            n += 1
            try:
                img = cv2.imread(os.path.join(path, dataset, label, file))
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                # normalize colour values
                X.append(img / 127.5)
                if label == "Cat":
                    y.append(0)
                else:
                    y.append(1)
            except Exception as e:
                print(f"skipped image {file} because of {e}")
            if n == count:
                break

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')


X, y = load_cats_dogs("PetImages", "kagglecatsanddogs_3367a", 100)


def shuffle(l1, l2):
    """
    :param l1: array
    :param l2: array (has to be the same length as l1
    :return: None
    l1 and l2 will be shuffled; l1[i] has the partner value still at l2[i]
    """
    for i in range(1000):
        p1 = random.randint(0, len(l1) - 1)
        p2 = random.randint(0, len(l1) - 1)
        l1[p1], l1[p2] = l1[p2], l1[p1]
        l2[p1], l2[p2] = l2[p2], l2[p1]


shuffle(X, y)

x_train, x_val, y_train, y_val = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.78, random_state=42)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
y_val = tf.keras.utils.to_categorical(y_val)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(2, activation='softmax'),

    ]
)

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=EPOCHS, validation_steps=20, validation_data=(x_val, y_val), )

model.evaluate(x_test, y_test, verbose=2)

model.save("model2.h5")
