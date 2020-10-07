"""Main module."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from dexpression_pytorch.datasets import dataset, utils


def run():
    x, y = dataset.get_dataset()
    folds = utils.kfold(x, y)

    for train, test in folds:
        x_train, y_train, x_test, y_test = utils.convert_to_torch(
            x[train], y[train], x[test], y[test]
        )

        # training.run(x_train, y_train)
        # testing.run(x_test, y_test)


if __name__ == "__main__":
    run()
