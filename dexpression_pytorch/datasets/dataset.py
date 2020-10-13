import numpy as np
import glob, os
from matplotlib import image as im

from dexpression_pytorch import settings


def file_reader(image_file, label_file):
    """
    Reads and converts the variables to numpy arrays.

    Parameters
    ----------
    image_file : str
        The path to the image file.
    label_file : str
        The path to the label file.

    Returns
    -------
    image : object
        The input variables as a numpy array.
    label : object
        The output variables as a numpy array.
    """

    image = im.imread(image_file)

    with open(label_file, "r") as file:
        label = float(file.read())

    return image, label


def load_from_array():
    """
    Loads existing dataset from a local folder.

    Returns
    -------
    x : object
        The input variables from the dataset.
    y : object
        The output variables from the dataset.
    """

    x = np.load(settings.data("x.npy")).reshape(-1, 1, 224, 224)
    y = np.load(settings.data("y.npy"))

    return x, y


def save_to_array(x, y):
    """
    Saves the dataset to a local folder.

    Parameters
    ----------
    x : object
        The input variables from the dataset.
    y : object
        The output variables from the dataset.
    """

    with open(settings.data("x.npy"), "wb") as file:
        np.save(file, x)

    with open(settings.data("y.npy"), "wb") as file:
        np.save(file, y)


def load_dataset(use_existing=True):
    """
    Returns the input and output variables from the dataset.

    Parameters
    ----------
    use_existing : bool, optional
        Loads existing dataset from a local folder (default is True).

    Returns
    -------
    x : object
        The input variables from the dataset.
    y : object
        The output variables from the dataset.
    """

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

        x = np.stack(images, axis=0).reshape(-1, 1, 224, 224)
        y = np.stack(labels, axis=0)

        save_to_array(x, y)

    print("Loaded datasets {} and {}...".format(x.shape, y.shape))
    print("")

    return x, y
