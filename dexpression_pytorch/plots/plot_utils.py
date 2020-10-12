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


def confusion_matrix(fold, epoch=25):
    test_pred = get_test_pred(fold=fold, epoch=epoch)
    test_truth = get_test_truth(fold=fold, epoch=epoch)
    matrix = cm(test_truth, test_pred)
    matrix = (matrix.T / matrix.astype(np.float).sum(axis=1)).T
    matrix_df = pd.DataFrame(matrix, labels, labels)

    filename = settings.results("confusion_matrix-fold{:d}".format(fold))

    return matrix_df, filename


def get_metric(metric):
    if metric == "accuracy":
        train_output = get_data("avg_train_accuracy")
        test_output = get_data("avg_test_accuracy")
    else:
        train_output = get_data("avg_train_loss")
        test_output = get_data("avg_test_loss")

    filename = settings.results("train_test_{:s}".format(metric))

    return train_output, test_output, filename


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


def get_data(metric):
    list = []

    df = pd.read_csv(history)
    for fold in range(5):
        filter = df["fold"] == fold + 1
        train_accuracy = df.loc[filter, metric].tolist()

        list.append(train_accuracy)

    return list
