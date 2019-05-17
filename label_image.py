import numpy as np
import tensorflow as tf
from PIL import Image
import argparse


def resize(image, size, method=Image.ANTIALIAS):
    img = Image.fromarray(image)
    return np.array(img.resize(size, method))


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


def read_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


if __name__ == '__main__':
    image_file = 'example_sign.jpg'
    model_file = 'models/model.h5'
    label_file = 'labels.txt'

    do_grayscale = False
    do_equalize = False

    input_shape = (32, 32)

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='image to be classified')
    parser.add_argument('--model', help='model to be executed')
    parser.add_argument('--labels', help='name of file containing labels')
    parser.add_argument('--grayscale', help='convert to grayscale')
    parser.add_argument('--equalize', help='apply histogram equalization')
    args = parser.parse_args()

    if args.image:
        image_file = args.image
    if args.model:
        model_file = args.model
    if args.labels:
        label_file = args.labels
    if args.grayscale:
        do_grayscale = args.grayscale
    if args.equalize:
        do_equalize = args.equalize

    image = Image.open(image_file).resize(input_shape, Image.LANCZOS)
    image = np.array(image)

    if do_grayscale:
        image = to_grayscale(image)

    if do_equalize:
        image = image_histogram_equalization(image)

    images = np.stack([normalize(image),], axis=0)
    if do_grayscale:
        images = np.expand_dims(images, 3)

    model = tf.keras.models.load_model(model_file)

    results = np.squeeze(model.predict(images))

    top_k = results.argsort()[-5:][::-1]
    labels = read_labels(label_file)

    for i in top_k:
        print(labels[i], '[' + str(results[i]) + ']')
