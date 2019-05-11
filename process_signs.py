from PIL import Image
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa


number_of_classes = 93

# 9 should be but since I'm using a different one, it isn't
# 55, 56, 78, 81 maybe?
_flip_horizontally = [(0, 0), (1, 1), (3, 3), (7, 7), (8, 8), (34, 35), (39, 39), (41, 42), (43, 44), (45, 45),
                      (47, 47), (48, 49), (50, 50), (51, 51), (54, 54), (62, 62), (63, 63), (65, 65), (67, 67),
                      (68, 69), (70, 71), (72, 72), (77, 77), (80, 80), (82, 82), (83, 84), (86, 86), (88, 88),
                      (92, 92)]

_flip_vertically = [(1, 1), (7, 7), (8, 8), (24, 24), (39, 39), (74, 74), (75, 75), (83, 83), (84, 84)]

# 66 maybe?
# 1, 7, 8, 39, 73 and 74, 83 and 84 are already in both flip horizontally and flip vertically
_rotate_180 = [(2, 2), (38, 38)]

# rotate arrows - (class_number, rotation) - where arrow pointing up is 0, counting clockwise
_rotate_arrows = [(67, 0), (73, 90), (74, 270), (75, 135), (76, 225)]


def augment_images(images, labels):
    augment(images, labels, _flip_horizontally, flip_horizontally)
    augment(images, labels, _flip_vertically, flip_vertically)
    augment(images, labels, _rotate_180, rotate180)
    augment_arrows(images, labels, _rotate_arrows)

    return images, labels


def augment(images, labels, list_of_pairs, method):
    new_images = []
    new_labels = []

    for image, label in zip(images, labels):
        for first, second in list_of_pairs:
            if label == str(first):
                new_images.append(method(image))
                new_labels.append(str(second))
            elif first != second and label == str(second):
                new_images.append(method(image))
                new_labels.append(str(first))

    images.extend(new_images)
    labels.extend(new_labels)


def augment_arrows(images, labels, list_of_arrows):
    new_images = []
    new_labels = []

    for image, label in zip(images, labels):
        for first_cat, first_angle in list_of_arrows:
            if str(first_cat) == label:
                for second_cat, second_angle in list_of_arrows:
                    if first_cat == second_cat:
                        continue

                    new_images.append(rotate(image, first_angle - second_angle))
                    new_labels.append(str(second_cat))

    images.extend(new_images)
    labels.extend(new_labels)


# All methods in this module should take and return image(s) as numpy array


def flip_horizontally(image):
    img = Image.fromarray(image)
    return np.array(img.transpose(Image.FLIP_LEFT_RIGHT))


def flip_vertically(image):
    img = Image.fromarray(image)
    return np.array(img.transpose(Image.FLIP_TOP_BOTTOM))


def resize(image, size, method=Image.ANTIALIAS):
    img = Image.fromarray(image)
    return np.array(img.resize(size, method))


def rotate(image, angle):
    img = Image.fromarray(image)
    return np.array(img.rotate(angle))


def rotate180(image):
    return rotate(image, 180)


def to_grayscale(image):
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


def normalize(image):
    return image / 255.0


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    # taken from https://stackoverflow.com/a/3823822

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


# an example augmentation sequence using imgaug library
# more on imgaug.readthedocs.io
ia.seed(42)
aug_seq = iaa.Sequential([
    # change shape
    iaa.OneOf([
        iaa.Crop(percent=(0, 0.1)),
        iaa.Affine(
            rotate=(-10, 10),
            shear=(-5, 5),
            mode=ia.ALL
        )
    ]),

    # change color
    iaa.OneOf([
        iaa.Multiply((0.9, 1.1), per_channel=0.5),
        iaa.Add((-40, 40), per_channel=0.5),
        iaa.Grayscale((0, 0.4)),
        iaa.ContrastNormalization((0.75, 1.25))
    ]),

    # add blur
    iaa.OneOf([
        iaa.GaussianBlur(sigma=(0, 0.2)),
        iaa.AverageBlur(k=(2, 5)),
        iaa.MedianBlur(k=(3, 5)),
        iaa.Superpixels(p_replace=0.2, n_segments=(16, 128)),
        iaa.Sharpen(alpha=(0, 0.25), lightness=(0.75, 1.25)),
        iaa.Emboss(alpha=(0, 0.25), strength=(0.5, 1.5))
    ]),

    # add noise
    iaa.OneOf([
        iaa.AdditiveGaussianNoise(scale=(0.0, 0.025 * 255), per_channel=0.5),
        iaa.MultiplyElementwise((0.75, 1.25)),
        iaa.AddElementwise((-40, 40), per_channel=0.5),
        iaa.Dropout((0, 0.2), per_channel=0.5),
        iaa.CoarseDropout((0.0, 0.2), size_percent=(0.4, 0.75), per_channel=0.5)
    ])
], random_order=True)
# image = Image.open('example_sign.jpg')
# signs = [np.array(image)]*32
# signs_aug = aug_seq.augment_images(signs)

# tf.image.per_image_standardization
