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

    filename = settings.results("confusion_matrix-fold{:d}".format(fold))

    return matrix_df, filename


def get_accuracy():
    train_accuracy = get_train_accuracy()
    test_accuracy = get_test_accuracy()

    filename = settings.results("train_test_accuracy")

    return train_accuracy, test_accuracy, filename


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


def get_train_accuracy():
    list = []

    df = pd.read_csv(history)
    for fold in range(5):
        filter = df["fold"] == fold + 1
        train_accuracy = df.loc[filter, "avg_train_accuracy"].tolist()

        list.append(train_accuracy)

    return list


def get_test_accuracy():
    list = []

    df = pd.read_csv(history)
    for fold in range(5):
        filter = df["fold"] == fold + 1
        test_accuracy = df.loc[filter, "avg_test_accuracy"].tolist()

        list.append(test_accuracy)

    return list
