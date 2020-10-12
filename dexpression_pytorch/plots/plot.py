import seaborn as sb
import matplotlib.pyplot as plt

from dexpression_pytorch.plots import plot_utils


def plot_confusion_matrix():
    matrix_df, filename = plot_utils.confusion_matrix()

    ax = sb.heatmap(matrix_df, annot=True, cmap="YlGnBu")
    ax.set(ylabel="True", xlabel="Predicted")
    plt.title("Confusion Matrix")
    plt.savefig(filename, bbox_inches="tight", dpi=200)
    plt.close()


def plot_accuracy():
    x_axis = [epoch for epoch in range(25)]
    train_accuracy, test_accuracy, filename = plot_utils.get_accuracy()

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    for fold in range(5):
        ax1.plot(x_axis, train_accuracy[fold], label="fold {:d}".format(fold + 1))
        ax2.plot(x_axis, test_accuracy[fold], label="fold {:d}".format(fold + 1))

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Train Accuracy")
    ax1.legend()

    ax2.set_xlabel("Epochs")
    ax2.set_title("Test Accuracy")
    ax2.legend()

    ratio = 0.8
    for ax in [ax1, ax2]:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_aspect(abs((xmax - xmin) / (ymax - ymin)) * ratio, adjustable="box")

    plt.savefig(filename, bbox_inches="tight", dpi=200)
    plt.show()


def plot_loss():
    pass
