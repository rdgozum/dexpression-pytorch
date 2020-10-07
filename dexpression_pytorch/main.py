"""Main module."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from dexpression_pytorch.datasets import dataset
from dexpression_pytorch.datasets import utils


def run():
    x, y = dataset.get_dataset()
    folds = utils.kfold(x, y)

    for train, test in folds:
        print(x[train].shape)
        print(y[train].shape)
        print(x[test].shape)
        print(y[test].shape)


if __name__ == "__main__":
    run()
