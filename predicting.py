import math

import numpy as np
import cv2
import os
import tensorflow as tf

import matplotlib.pyplot as plt

IMG_WIDTH, IMG_HEIGHT = 64, 64


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


model1 = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=0.01)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(2, activation="softmax")
    ]
)

model1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model1.load_weights("model1.h5")

model2 = tf.keras.models.Sequential(
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

model2.compile(loss='categorical_crossentropy',
               optimizer='adam', metrics=['accuracy'])

model2.load_weights("model2.h5")

CLASSES = ["CAT", "DOG"]


def predict_multiple_images(count, model=2, show_images=True):
    """
    :param count: number of images you want to load and predict
    :param model: input 1 if you want to predict with model1.h5, uses model2.h5 by default
    :param show_images:
    :return:
    """
    X, y = load_cats_dogs("PetImages", "kagglecatsanddogs_3367a", count)

    c = 0

    if model == "1":
        prediction = model1.predict(X)
    else:
        prediction = model2.predict(X)

    for i in range(len(prediction)):
        actual_label = y[i]
        prediction_class = np.argmax(prediction[i])
        if actual_label == prediction_class:
            c += 1

        if show_images:
            plt.imshow(X[i])
            plt.xlabel("Actual: " + str(actual_label) + "  " + CLASSES[int(actual_label)])
            plt.title("Prediction: " + str(prediction_class) + "  " + CLASSES[int(prediction_class)])
            plt.show()

    print(f"Accuracy: {c / len(X)}")


def predict_file_img(file_path, label, model=2):
    """
    :param file_path: path to img you want to classify
    :param label: label of the img you want to classify
    :param model: input int:1 if you want to predict with model1.h5, uses model2.h5 by default
    :return: None
    """

    img_data, raw_image = load_img(file_path)

    if model == 1:
        prediction = model1.predict(img_data)
    else:
        prediction = model2.predict(img_data)

    prediction_class = np.argmax(prediction[0])
    plt.imshow(raw_image)
    plt.xlabel("Actual: " + str(label))
    plt.title("Prediction: " + str(prediction_class) + "  " + CLASSES[int(prediction_class)])
    plt.show()


def load_img(file_path):
    """
    :param file_path: path to file you want to load
    :return: first value is normalized img to feed into model, second value is the normal loaded img to visualize
    """
    img = cv2.imread(file_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    # normalize colour values
    img_normalized = img / 127.5

    b = np.array([img_normalized])
    return b, img


predict_multiple_images(50, show_images=False)
predict_file_img("images_to_test/cat1.jpg", "CAT")
