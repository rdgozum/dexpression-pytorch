import numpy as np
import torch
from sklearn.model_selection import KFold


def kfold(x, y, splits=5, shuffle=True):
    kfold = KFold(n_splits=splits, shuffle=shuffle)
    for train, test in kfold.split(x, y):
        yield train, test


def convert_to_torch(x_train, y_train, x_test, y_test):
    # converting training images into torch format
    x_train = torch.from_numpy(x_train)
    x_train = x_train.type(torch.FloatTensor)

    # converting the label into torch format
    y_train = y_train.astype(int)
    y_train = torch.from_numpy(y_train)

    # converting test images into torch format
    x_test = torch.from_numpy(x_test)
    x_test = x_test.type(torch.FloatTensor)

    # converting the label into torch format
    y_test = y_test.astype(int)
    y_test = torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test
