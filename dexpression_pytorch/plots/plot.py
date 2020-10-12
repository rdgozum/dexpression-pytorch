import seaborn as sb
import matplotlib.pyplot as plt

from dexpression_pytorch.plots import plot_utils


def plot_confusion_matrix():
    matrix_df, output_filename = plot_utils.confusion_matrix()

    ax = sb.heatmap(matrix_df, annot=True, cmap="YlGnBu")
    ax.set(ylabel="True", xlabel="Predicted")
    plt.title("Confusion Matrix")
    plt.savefig(output_filename, bbox_inches="tight", dpi=200)
    plt.close()


def plot_train_accuracy():
    pass


def plot_test_accuracy():
    pass


def plot_train_loss():
    pass


def plot_test_loss():
    pass
