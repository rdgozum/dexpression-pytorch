import torch
from torch.nn import Module, Conv2d, MaxPool2d, Linear, ReLU
from torch.nn import LayerNorm, BatchNorm1d, Dropout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dexpression(Module):
    def __init__(self):
        super(Dexpression, self).__init__()

        # First Block
        self.conv1 = Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
        )  #
        self.relu1 = ReLU(inplace=True)
        self.pool1 = MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.lrn1 = LayerNorm(64)

        # Second Block
        self.conv2a = Conv2d(
            in_channels=64, out_channels=96, kernel_size=1, stride=1, padding=0
        )
        self.relu2a = ReLU(inplace=True)
        self.conv2b = Conv2d(
            in_channels=96, out_channels=208, kernel_size=3, stride=1, padding=1
        )
        self.relu2b = ReLU(inplace=True)
        self.pool2a = MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2c = Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0
        )
        self.relu2c = ReLU(inplace=True)
        self.pool2b = MaxPool2d(kernel_size=3, stride=2, padding=0)

        # Third Block
        self.conv3a = Conv2d(
            in_channels=272, out_channels=96, kernel_size=1, stride=1, padding=0
        )
        self.relu3a = ReLU(inplace=True)
        self.conv3b = Conv2d(
            in_channels=96, out_channels=208, kernel_size=3, stride=1, padding=1
        )
        self.relu3b = ReLU(inplace=True)
        self.pool3a = MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv3c = Conv2d(
            in_channels=272, out_channels=64, kernel_size=1, stride=1, padding=0
        )
        self.relu3c = ReLU(inplace=True)
        self.pool3b = MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.fc = Linear(in_features=282 * 14 * 14, out_features=11)

        self.batch_normalization = BatchNorm1d(282)
        self.dropout = Dropout(p=0.2)

    def forward(self):
        pass
