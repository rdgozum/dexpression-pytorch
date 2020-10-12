import numpy as np
import pandas as pd
import ast

from sklearn.metrics import confusion_matrix as cm
from dexpression_pytorch import settings

history = settings.results("history.csv")

labels = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Sadness",
    "Surprise",
]


def confusion_matrix(fold=1, epoch=25):
    test_pred = get_test_pred(fold=fold, epoch=epoch)
    test_truth = get_test_truth(fold=fold, epoch=epoch)
    matrix = cm(test_truth, test_pred)
    matrix = (matrix.T / matrix.astype(np.float).sum(axis=1)).T
    matrix_df = pd.DataFrame(matrix, labels, labels)

    filename = "confusion_matrix-fold{:d}".format(fold)
    output_filename = settings.results(filename)

    return matrix_df, output_filename


def get_test_pred(fold, epoch):
    df = pd.read_csv(history)
    filter = (df["fold"] == fold) & (df["epoch"] == epoch)
    test_pred = df.loc[filter, "test_pred"].item()
    test_pred = ast.literal_eval(test_pred)

    return test_pred


def get_test_truth(fold, epoch):
    df = pd.read_csv(history)
    filter = (df["fold"] == fold) & (df["epoch"] == epoch)
    test_truth = df.loc[filter, "test_truth"].item()
    test_truth = ast.literal_eval(test_truth)

    return test_truth
