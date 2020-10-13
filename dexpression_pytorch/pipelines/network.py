import torch
from dexpression_pytorch.cnn_model.dexpression import Dexpression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize():
    """
    Loads model parameters into cuda.

    Returns
    -------
    model : object
        The convolutional neural network to be trained.
    """

    model = Dexpression()
    model = model.to(device)

    return model
