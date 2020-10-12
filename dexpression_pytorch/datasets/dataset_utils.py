import torch
from sklearn.model_selection import KFold
from sklearn.utils import shuffle as s


def kfold(x, y, splits=5, shuffle=True):
    x, y = s(x, y)
    kfold = KFold(n_splits=splits, shuffle=shuffle)

    for train, test in kfold.split(x, y):
        x_train, y_train = x[train], y[train]
        x_test, y_test = x[test], y[test]

        yield x_train, y_train, x_test, y_test


def convert_to_torch(x_train, y_train, x_test, y_test):
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
