import numpy as np
import glob, os
from matplotlib import image as im

from dexpression_pytorch import settings


def file_reader(image_file, label_file):
    image = im.imread(image_file)

    with open(label_file, "r") as file:
        label = float(file.read())

    return image, label


def load_from_array():
    x = np.load(settings.data("x.npy"))
    y = np.load(settings.data("y.npy"))

    return x, y


def save_to_array(x, y):
    with open(settings.data("x.npy"), "wb") as file:
        np.save(file, x)

    with open(settings.data("y.npy"), "wb") as file:
        np.save(file, y)


def get_dataset(use_existing=True):
    if use_existing:
        x, y = load_from_array()
    else:
        data_dir = settings.DATA_FOLDER
        images = []
        labels = []

        for image_file in sorted(glob.glob(f"{data_dir}/images/**/**/*.png")):
            image_path = os.path.dirname(image_file)
            label_path = image_path.replace("images", "labels")

            if os.path.exists(label_path):
                if not len(os.listdir(label_path)) == 0:
                    label_file = os.path.join(label_path, os.listdir(label_path)[0])

                    image, label = file_reader(image_file, label_file)
                    images.append(image)
                    labels.append(label)

        x = np.stack(images, axis=0)
        y = np.stack(labels, axis=0)

        save_to_array(x, y)

    return x, y
