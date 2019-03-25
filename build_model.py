from tensorflow import keras
import numpy as np

metrics = [keras.metrics.Accuracy()]
number_of_classes = 93
batch_size = 16

keras.backend.set_image_data_format('channels_last')


def build_model(input_shape, conv_depth, dense_depth, optimizer, loss):
    if conv_depth == 0 and dense_depth == 0:
        raise AttributeError

    shape = input_shape
    model = keras.Sequential()
    for i in range(conv_depth):
        model.add(keras.layers.Conv2D(32 * (i+1), (3, 3), input_shape=shape, name=f'conv_{i+1}'))
        shape = (shape[0] - 2, shape[1] - 2, 32 * (2**i))

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), name='pooling'))
    shape = (shape[0] / 2, shape[1] / 2, shape[2])

    model.add(keras.layers.Flatten(input_shape=shape, name='flatten'))

    for i in range(dense_depth):
        model.add(keras.layers.Dense(200 * (dense_depth - i), activation=keras.activations.relu, name=f'dense_{i+1}'))

    model.add(keras.layers.Dense(93, activation=keras.activations.softmax, name='dense_softmax'))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def build_models():
    models = []

    for conv_depth in range(1, 5):
        for dense_depth in range(1, 5):
            model = build_model((32, 32, 3), conv_depth, dense_depth, keras.optimizers.Adam(), keras.losses.categorical_crossentropy)
            models.append(model)

    return models


def build_baseline_model():
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(32, (3, 3), activation=keras.activations.relu, input_shape=(32, 32, 3), name='conv_1.1'))
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
    model.add(keras.layers.Dense(number_of_classes, activation=keras.activations.softmax, name='dense_softmax'))

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=metrics)

    return model
