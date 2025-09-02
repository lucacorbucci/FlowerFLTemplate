import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn


def get_model(
    dataset: str,
) -> torch.nn.Module:
    """This function returns the model to train.

    Args:
        dataset (str): the name of the dataset
        device (torch.device): the device where the model will be trained

    Raises:
        ValueError: if the dataset is not supported

    Returns:
        torch.nn.Module: the model to train
    """
    if dataset == "celeba":
        return CelebaNet()
    elif dataset == "dutch":
        return LinearClassificationNet(input_size=12, output_size=2)
    elif dataset == "income":
        return LinearClassificationNet(input_size=54, output_size=2)
    elif dataset == "adult":
        return LinearClassificationNet(input_size=103, output_size=2)
    elif dataset == "mnist":
        return SimpleMNISTModel()
    else:
        raise ValueError(f"Dataset {dataset} not supported")

class Net(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LinearClassificationNet(nn.Module):
    """
    A fully-connected single-layer linear NN for classification.
    """

    def __init__(self, input_size, output_size):
        super(LinearClassificationNet, self).__init__()
        self.layer1 = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        x = self.layer1(x.float())
        return x



class SimpleMNISTModel(nn.Module):
    """
    Simple fully connected model for MNIST digit classification.
    """
    def __init__(self, num_classes=10):
        super(SimpleMNISTModel, self).__init__()
        
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 28 * 28)
        
        # Pass through layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class CelebaNet(nn.Module):
    """This class defines the CelebaNet."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        dropout_rate: float = 0,
    ) -> None:
        """Initializes the CelebaNet network.

        Args:
        ----
            in_channels (int, optional): Number of input channels . Defaults to 3.
            num_classes (int, optional): Number of classes . Defaults to 2.
            dropout_rate (float, optional): _description_. Defaults to 0.2.
        """
        super().__init__()
        self.cnn1 = nn.Conv2d(
            in_channels,
            8,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(1, 1),
        )
        self.cnn2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.cnn3 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.fc1 = nn.Linear(2048, 2)
        self.gn_relu = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_data: Tensor) -> Tensor:
        """Defines the forward pass of the network.

        Args:
            input_data (Tensor): Input data

        Returns
        -------
            Tensor: Output data
        """
        out = self.gn_relu(self.cnn1(input_data))
        out = self.gn_relu(self.cnn2(out))
        out = self.gn_relu(self.cnn3(out))
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out
