#!/usr/bin/env python3
import os
from sys import argv, stderr
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random as rnd
from datetime import datetime

_NUMBER_OF_CLASSES = 93

_flip_horizontally = [(0, 0), (1, 1), (3, 3), (7, 7), (8, 8), (34, 35), (39, 39), (41, 42), (43, 44), (45, 45),
                      (47, 47), (48, 49), (50, 50), (51, 51), (54, 54), (62, 62), (63, 63), (65, 65), (67, 67),
                      (68, 69), (70, 71), (72, 72), (77, 77), (80, 80), (82, 82), (83, 84), (86, 86), (88, 88),
                      (92, 92)]

_flip_vertically = [(1, 1), (7, 7), (8, 8), (24, 24), (39, 39), (74, 74), (75, 75), (83, 83), (84, 84)]

_rotate_180 = [(2, 2), (38, 38)]

_rotate_arrows = [(67, 0), (73, 90), (74, 270), (75, 135), (76, 225)]

rnd.seed(123)
tf.set_random_seed(123)
np.random.seed(123)


def read_dataset(path: str) -> list:
    images = [[] for _ in range(_NUMBER_OF_CLASSES)]

    print('Loading data set')
    for c in range(_NUMBER_OF_CLASSES):
        print(c, '/', str(_NUMBER_OF_CLASSES - 1))
        dir_name = path + '/' + format(c, '05d')

        for f in os.listdir(dir_name):
            img = Image.open(dir_name + '/' + f)
            images[c].append(np.array(img))

    print('Data set loaded')
    return images


def build_model(input_shape: tuple):
    model = keras.models.Sequential()

    model.add(
        keras.layers.Conv2D(32, (3, 3), activation=keras.activations.relu, input_shape=input_shape, name='conv_1.1'))
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

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['sparse_categorical_accuracy'])

    return model


def augment_images(images):
    images = augment(images, _flip_horizontally, flip_horizontally)
    images = augment(images, _flip_vertically, flip_vertically)
    images = augment(images, _rotate_180, rotate180)

    return images


def augment(images, list_of_pairs, method):
    new_images = [[] for _ in range(_NUMBER_OF_CLASSES)]

    for first, second in list_of_pairs:
        for image in images[first]:
            new_images[second].append(method(image))

        if first != second:
            for image in images[second]:
                new_images[first].append(method(image))

    return [images[c] + new_images[c] for c in range(_NUMBER_OF_CLASSES)]


def flip_horizontally(image):
    return np.fliplr(image)


def flip_vertically(image):
    return np.flipud(image)


def resize(image, size, method=Image.ANTIALIAS):
    img = Image.fromarray(image)
    return np.array(img.resize(size, method))


def rotate180(image):
    return np.rot90(image, 2)


def to_grayscale(image):
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


def normalize(image):
    return image / 255.0


def image_histogram_equalization(image, number_bins=256):
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()
    cdf = 255 * cdf / cdf[-1]

    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def list_of_lists_to_numpy_array_images_and_labels(lst):
    images = []
    labels = []

    for c in range(len(lst)):
        for image in lst[c]:
            images.append(image)
            labels.append(c)

    return np.stack(images, axis=0), labels


def split_images(images, ratio):
    a = [[] for _ in range(_NUMBER_OF_CLASSES)]
    b = [[] for _ in range(_NUMBER_OF_CLASSES)]

    for c in range(_NUMBER_OF_CLASSES):
        rnd.shuffle(images[c])
        n = len(images[c])
        t = int(n * ratio)

        a[c] += images[c][:t]
        b[c] += images[c][t:]

    return a, b


if __name__ == '__main__':
    if len(argv) != 5:
        print('usage: ', argv[0], ' path/to/data grayscale hist_equalization augment', file=stderr)
        exit(1)

    path = argv[1]

    testing_ratio = 0.1
    validation_ratio = 0.1

    batch_size = 64
    epochs = 10
    input_shape = (32, 32)

    do_grayscale = argv[2] == 'True'
    do_equalization = argv[3] == 'True'
    do_augment = argv[4] == 'True'

    print('Running with configuration:')
    print('Testing ratio:', testing_ratio)
    print('Validation ratio:', validation_ratio)
    print('Batch size:', batch_size)
    print('Epochs:', epochs)
    print('Input size:', input_shape)
    print('Grayscale:', do_grayscale)
    print('Histogram equalization:', do_equalization)
    print('Augment:', do_augment)
    print()

    images = read_dataset(path)

    print('Resizing images')
    images = [[resize(image, input_shape) for image in c] for c in images]

    if do_grayscale:
        print('Converting images to grayscale')
        images = [[to_grayscale(image) for image in c] for c in images]

    if do_equalization:
        print('Applying histogram equalization')
        images = [[image_histogram_equalization(image) for image in c] for c in images]

    print('Normalizing images')
    images = [[normalize(image) for image in c] for c in images]

    test_images, train_images = split_images(images, testing_ratio)
    del images

    if do_augment:
        print('Augmenting images')
        train_images = augment_images(train_images)

    validation_images, train_images = split_images(train_images, validation_ratio)

    test_images, test_labels = list_of_lists_to_numpy_array_images_and_labels(test_images)
    validation_images, validation_labels = list_of_lists_to_numpy_array_images_and_labels(validation_images)
    train_images, train_labels = list_of_lists_to_numpy_array_images_and_labels(train_images)

    if do_grayscale:
        test_images = np.expand_dims(test_images, 3)
        validation_images = np.expand_dims(validation_images, 3)
        train_images = np.expand_dims(train_images, 3)

    model = build_model(input_shape + (1 if do_grayscale else 3,))

    history = model.fit(train_images, train_labels,
                        validation_data=(validation_images, validation_labels),
                        batch_size=batch_size,
                        epochs=epochs)
    print('\nhistory:', history.history)

    eval = model.evaluate(test_images, test_labels)
    print('\neval:', eval)

    name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.h5'

    print('Saving model', name)
    model.save(os.path.join(path, name))
