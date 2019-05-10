import matplotlib.pyplot as plt


def show_images(images: list, axes: bool = True) -> None:
    n = len(images)
    cols = int(n ** 0.5)
    rows = n // cols + (1 if n % cols > 0 else 0)

    fig = plt.figure(figsize=(8, 8))
    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1)
        if not axes:
            ax.axis('off')
        plt.imshow(images[i])

    plt.show()


def dataset_stats(images: list, labels: list) -> None:
    min_size = 100000
    max_size = 0

    x, y = 0, 0

    for image in images:
        size = image.shape[0] * image.shape[1]
        if size < min_size:
            min_size = size
            min_height = image.shape[0]
            min_width = image.shape[1]
        elif size > max_size:
            max_size = size
            max_height = image.shape[0]
            max_width = image.shape[1]

        x += image.shape[0]
        y += image.shape[1]

    x /= len(images)
    y /= len(images)

    print(f'Number of images: {len(images)}')
    print(f'Min size: {min_height}x{min_width}px')
    print(f'Max size: {max_height}x{max_width}px')
    print(f'Average size: {x}x{y}px')


def hist_values(labels: list, classes: int, print_coordinates=False) -> list:
    counts = [0 for _ in range(classes)]

    for label in labels:
        counts[int(label)] += 1

    if print_coordinates:
        print('coordinates{')
        for i in range(classes):
            print(f'({i}, {counts[i]})')

        print('};')

    return counts
