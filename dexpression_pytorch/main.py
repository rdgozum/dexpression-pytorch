"""Main module."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from dexpression_pytorch.datasets import dataset, utils
from dexpression_pytorch.pipelines import network, training, testing
from dexpression_pytorch.utilities import output_writer
from dexpression_pytorch.plots import plot


def run():
    x, y = dataset.get_dataset()
    folds = utils.kfold(x, y)

    for fold, (x_train, y_train, x_test, y_test) in enumerate(folds):
        x_train, y_train, x_test, y_test = utils.convert_to_torch(
            x_train, y_train, x_test, y_test
        )

        model = network.initialize()
        training.run(fold, model, x_train, y_train, x_test, y_test)
        # testing.run(model, x_test, y_test)

    # Save history
    output_writer.dump_dict_list(training.history)

    # Plot history
    plot.plot_confusion_matrix()
    plot.plot_accuracy()
    plot.plot_loss()


if __name__ == "__main__":
    run()
