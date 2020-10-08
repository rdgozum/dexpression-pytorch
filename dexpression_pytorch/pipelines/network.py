import torch
from dexpression_pytorch.cnn_model.dexpression import Dexpression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize():
    model = Dexpression()
    model = model.to(device)

    return model
