import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing import image


EPOCHS = 10
IMG_WIDTH = 32
IMG_HEIGHT = 32
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Add swish to keras layers
    tf.keras.utils.get_custom_objects().update({'swish': swish})

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    # model.fit(x_train, y_train, epochs=EPOCHS)
    # model.fit(x_train, y_train, validation_split=0.2, epochs=EPOCHS)

    datagen = image.ImageDataGenerator(
        brightness_range=[0.2,1.5],
        zoom_range=0.1,
        shear_range=0.05,
        rotation_range=10)

    datagen.fit(x_train)
    model.fit(datagen.flow(x_train, y_train, batch_size=16),
            steps_per_epoch=len(x_train) / 16, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")
    
    # Visualization
    # from keract import get_activations, display_activations
    # keract_inputs = x_train[:1]
    # # keract_targets = target_test[:1]
    # activations = get_activations(model, keract_inputs)
    # display_activations(activations, cmap="gray", save=False)

def swish(x, beta=2):
    """
    swish is a new activation function from Google.
    Thanks to @barbara bai for the heads up on this!
    """
    # Define swish activation function
    return (x * tf.keras.backend.sigmoid(beta * x))


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    # https://stackoverflow.com/questions/12023958/what-does-cvnormalize-src-dst-0-255-norm-minmax-cv-8uc1
    images, labels = [], []
    for i in range(NUM_CATEGORIES):
        for filename in os.listdir(os.path.join(data_dir, str(i))):
            # if filename.endswith(".ppm"):
            img = cv2.imread(os.path.join(data_dir, str(i), filename))
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            images.append(img)
            labels.append(i)
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # Create a convolutional neural network
    model = tf.keras.models.Sequential([

        preprocessing.Rescaling(1. / 255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        Conv2D(32, 5, use_bias=True, padding='same'),
        BatchNormalization(),
        Activation("swish"),

        Conv2D(32, 3, use_bias=True, padding='same'),
        BatchNormalization(),
        Activation("swish"),

        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, 3, use_bias=True, padding='same'),
        BatchNormalization(),
        Activation("swish"),

        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, 3, use_bias=True, padding='same'),
        BatchNormalization(),
        Activation("swish"),

        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        
        Dense(NUM_CATEGORIES * 2, use_bias=True),
        BatchNormalization(),
        Activation("swish"),

        Dropout(0.4),

        Dense(NUM_CATEGORIES, use_bias=True),
        BatchNormalization(),
        Activation("swish"),

        Dropout(0.25),

        Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(optimizer="nadam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


if __name__ == "__main__":
    main()