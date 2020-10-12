"""Main module."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from dexpression_pytorch.datasets import dataset, dataset_utils
from dexpression_pytorch.pipelines import network, train_test, pipeline_utils
from dexpression_pytorch.plots import plot


def run():
    x, y = dataset.load_dataset()
    folds = dataset_utils.kfold(x, y)

    for fold, (x_train, y_train, x_test, y_test) in enumerate(folds):
        x_train, y_train, x_test, y_test = dataset_utils.convert_to_torch(
            x_train, y_train, x_test, y_test
        )

        model = network.initialize()
        train_test.run(fold, model, x_train, y_train, x_test, y_test)

    # Save history
    pipeline_utils.dump_dict_list(train_test.history)

    # Plot history
    print("Start plotting...")
    print("")
    for fold in range(5):
        plot.plot_confusion_matrix(fold + 1)
    plot.plot_metrics()


if __name__ == "__main__":
    run()
