"""Main module."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from dexpression_pytorch.datasets import transform_dataset


def run():
    x, y = transform_dataset.get_dataset()
    print(x)
    print(y)


if __name__ == "__main__":
    run()
