"""Dump Dict List to File"""
import torch
import pandas as pd

from datetime import datetime

from dexpression_pytorch import settings


def dump_dict_list(history):
    filename = settings.results("history.csv")

    print("Saving history {}...".format(filename))
    print("")

    df = pd.DataFrame(history)
    df.to_csv(filename)


def print_progress(
    fold,
    epoch,
    n_epochs,
    avg_train_accuracy,
    avg_train_loss,
    avg_test_accuracy,
    avg_test_loss,
):
    print("Fold: %d, Epoch: %d/%d" % (fold + 1, epoch + 1, n_epochs))
    print("Train Accuracy: %.2f%%" % (avg_train_accuracy * 100))
    print("Train Loss: %.3f" % (avg_train_loss))
    print("Test Accuracy: %.2f%%" % (avg_test_accuracy * 100))
    print("Test Loss: %.3f" % (avg_test_loss))
    print("")


def save_progress(fold, epoch, avg_test_accuracy, model, model_optimizer):
    model_name = "cnn-fold{:d}-{:d}".format(fold + 1, int(datetime.now().timestamp()))
    checkpoint = "{:s}_{:d}-{:.2f}.tar".format(
        model_name, epoch + 1, avg_test_accuracy,
    )

    print("Saving checkpoint {}...".format(settings.results(checkpoint)))
    print("")

    torch.save(
        {
            "fold": fold + 1,
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "model_opt": model_optimizer.state_dict(),
        },
        settings.results(checkpoint),
    )
