import numpy as np
import glob, os
from matplotlib import image as im

from dexpression_pytorch import settings


def read_files(image_file, label_file):
    image = im.imread(image_file)

    with open(label_file, "r") as f:
        label = float(f.read())

    return image, label


def get_arrays(load_from_file=False):
    if load_from_file:
        x = np.load(settings.data("x.npy"))
        y = np.load(settings.data("y.npy"))
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

                    image, label = read_files(image_file, label_file)
                    images.append(image)
                    labels.append(label)

        x, y = "", ""

    return x, y
