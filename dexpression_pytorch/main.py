"""Main module."""
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from dexpression_pytorch.datasets import dataset, utils
from dexpression_pytorch.pipelines import network, training, testing


def run():
    x, y = dataset.get_dataset()
    print("Loading datasets {} and {}".format(x.shape, y.shape))
    folds = utils.kfold(x, y)

    for x_train, y_train, x_test, y_test in folds:
        x_train, y_train, x_test, y_test = utils.convert_to_torch(
            x_train, y_train, x_test, y_test
        )

        # model = network.initialize()
        # training.run(model, x_train, y_train)
        # testing.run(model, x_test, y_test)


if __name__ == "__main__":
    run()
