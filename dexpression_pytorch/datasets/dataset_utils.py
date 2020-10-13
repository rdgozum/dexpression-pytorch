import torch
from sklearn.model_selection import KFold
from sklearn.utils import shuffle as s


def kfold(x, y, splits=5, shuffle=True):
    """
    Performs kfold split on the dataset.

    Parameters
    ----------
    x : object
        The input variables from the dataset.
    y : object
        The output variables from the dataset.
    splits : int, optional
        Number of folds (default is 5).
    shuffle : bool, optional
        Whether to shuffle the data before splitting into batches (default is True).

    Returns
    -------
    x_train : object
        The input variables to be used during training.
    y_train : object
        The output variables to be used during training.
    x_test : object
        The input variables to be used during testing.
    y_test : object
        The output variables to be used during testing.
    """

    x, y = s(x, y)
    kfold = KFold(n_splits=splits, shuffle=shuffle)

    for train, test in kfold.split(x, y):
        x_train, y_train = x[train], y[train]
        x_test, y_test = x[test], y[test]

        yield x_train, y_train, x_test, y_test


def convert_to_torch(x_train, y_train, x_test, y_test):
    """Converts train and test data into torch tensors."""

    # converting training images into torch tensor
    x_train = torch.from_numpy(x_train)
    x_train = x_train.type(torch.FloatTensor)

    # converting the label into torch tensor
    y_train = y_train.astype(int)
    y_train = torch.from_numpy(y_train)

    # converting test images into torch tensor
    x_test = torch.from_numpy(x_test)
    x_test = x_test.type(torch.FloatTensor)

    # converting the label into torch tensor
    y_test = y_test.astype(int)
    y_test = torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test
