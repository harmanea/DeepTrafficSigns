#!/usr/bin/env python3
import os
from sys import argv, stderr
from PIL import Image
import numpy as np
from tensorflow import keras

_NUMBER_OF_CLASSES = 93
_INPUT_SHAPE = (32, 32, 3)


def read_dataset(path):
    images = []
    labels = []

    print('Loading data set')
    for c in range(_NUMBER_OF_CLASSES):
        print(f'{c}/{_NUMBER_OF_CLASSES - 1}')
        dir_name = path + '/' + format(c, '05d')
        for f in os.listdir(dir_name):
            img = Image.open(dir_name + '/' + f)
            images.append(np.array(img))
            labels.append(c)

    print('Data set loaded')
    return images, labels


def resize(image, size, method=Image.ANTIALIAS):
    img = Image.fromarray(image)
    return np.array(img.resize(size, method))


def resize_all(images, size):
    print('Resizing images')
    return [resize(image, size) for image in images]


def build_baseline_model():
    model = keras.models.Sequential()

    model.add(
        keras.layers.Conv2D(32, (3, 3), activation=keras.activations.relu, input_shape=_INPUT_SHAPE, name='conv_1.1'))
    model.add(keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu, name='conv_1.2'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name='pooling_1'))
    model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu, name='conv_2.1'))
    model.add(keras.layers.Conv2D(128, (3, 3), activation=keras.activations.relu, name='conv_2.2'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name='pooling_2'))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1600, activation=keras.activations.relu, name='dense_1'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(800, activation=keras.activations.relu, name='dense_2'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(_NUMBER_OF_CLASSES, activation=keras.activations.softmax, name='dense_softmax'))

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy, metrics=[keras.metrics.Accuracy()])

    return model


if __name__ == '__main__':
    if len(argv) != 2:
        print(f'usage: {argv[0]} path/to/data', file=stderr)
        exit(1)

    images, labels = read_dataset(argv[1])
    images = resize_all(images, _INPUT_SHAPE[:2])
    images = np.stack(images, axis=0)
    model = build_baseline_model()

    history = model.fit(images, labels, batch_size=64, epochs=10)
    print('\nhistory:', history.history)
