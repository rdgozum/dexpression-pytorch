import numpy as np
from sklearn.model_selection import KFold


def kfold(x, y, splits=5, shuffle=True):
    kfold = KFold(n_splits=splits, shuffle=shuffle)
    for train, test in kfold.split(x, y):
        yield train, test
