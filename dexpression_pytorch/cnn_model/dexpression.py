import torch
import torch.nn.functional as F
from torch.nn import Module, Conv2d, MaxPool2d, Linear, ReLU, LogSoftmax
from torch.nn import LayerNorm, BatchNorm2d, Dropout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dexpression(Module):
    """Convolutional Neural Network for facial expression recognition."""

    def __init__(self):
        """The constructor for Dexpression class."""

        super(Dexpression, self).__init__()

        # First Block
        self.conv1 = Conv2d(
            in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.pool1 = MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.lrn1 = LayerNorm([64, 55, 55])

        # Second Block
        self.conv2a = Conv2d(
            in_channels=64, out_channels=96, kernel_size=1, stride=1, padding=0
        )
        self.conv2b = Conv2d(
            in_channels=96, out_channels=208, kernel_size=3, stride=1, padding=1
        )
        self.pool2a = MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2c = Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0
        )
        self.pool2b = MaxPool2d(kernel_size=3, stride=2, padding=0)

        # Third Block
        self.conv3a = Conv2d(
            in_channels=272, out_channels=96, kernel_size=1, stride=1, padding=0
        )
        self.conv3b = Conv2d(
            in_channels=96, out_channels=208, kernel_size=3, stride=1, padding=1
        )
        self.pool3a = MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv3c = Conv2d(
            in_channels=272, out_channels=64, kernel_size=1, stride=1, padding=0
        )
        self.pool3b = MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.fc = Linear(in_features=272 * 13 * 13, out_features=7)
        self.softmax = LogSoftmax(dim=1)

        self.batch_normalization = BatchNorm2d(272)
        self.dropout = Dropout(p=0.2)

    def forward(self, x, dropout=True, batch_normalization=True):
        """
        Performs a single forward pass through the network.

        Parameters
        ----------
        x : object
            The training batch containing input and output variables.
        dropout : bool, optional
            Enables dropout regularization method (default is True).
        LRN : bool, optional
            Enables Local Response Normalization (default is True).

        Returns
        -------
        logits : object
            The probability distribution over classes.
        """

        # First Block
        conv1_out = F.relu(self.conv1(x))
        pool1_out = self.pool1(conv1_out)
        lrn1_out = self.lrn1(pool1_out)

        # Second Block
        conv2a_out = F.relu(self.conv2a(lrn1_out))
        conv2b_out = F.relu(self.conv2b(conv2a_out))

        pool2a_out = self.pool2a(lrn1_out)
        conv2c_out = F.relu(self.conv2c(pool2a_out))

        concat2_out = torch.cat((conv2b_out, conv2c_out), 1)
        pool2b_out = self.pool2b(concat2_out)

        # Third Block
        conv3a_out = F.relu(self.conv3a(pool2b_out))
        conv3b_out = F.relu(self.conv3b(conv3a_out))

        pool3a_out = self.pool3a(pool2b_out)
        conv3c_out = F.relu(self.conv3c(pool3a_out))

        concat3_out = torch.cat((conv3b_out, conv3c_out), 1)
        pool3b_out = self.pool3b(concat3_out)

        if dropout:
            pool3b_out = self.dropout(pool3b_out)
        if batch_normalization:
            pool3b_out = self.batch_normalization(pool3b_out)

        pool3b_shape = pool3b_out.shape
        pool3b_flat = pool3b_out.reshape(
            [-1, pool3b_shape[1] * pool3b_shape[2] * pool3b_shape[3]]
        )

        output = self.fc(pool3b_flat)
        logits = self.softmax(output)

        return logits
