import torch.nn.functional as functional
from torch import Tensor, nn


class LinearClassificationNet(nn.Module):
    """
    A fully-connected single-layer linear NN for classification.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x.float())
        return x


class AbaloneNet(nn.Module):
    """Neural Network for Abalone age prediction"""

    def __init__(self, input_size: int, hidden_sizes: list[int] | None = None, dropout_rate: float = 0.2) -> None:
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.BatchNorm1d(hidden_size), nn.Dropout(dropout_rate)]
            )
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class SimpleMNISTModel(nn.Module):
    """
    A simpler fully connected model for MNIST.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # Flatten the input
        x = x.view(-1, 28 * 28)

        # Pass through layers
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class CelebaNet(nn.Module):
    """This class defines the CelebaNet."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        dropout_rate: float = 0,
    ) -> None:
        """
        Initializes the CelebaNet network.

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
        """
        Defines the forward pass of the network.

        Args:
            input_data (Tensor): Input data

        Returns:
        -------
            Tensor: Output data

        """
        out = self.gn_relu(self.cnn1(input_data))
        out = self.gn_relu(self.cnn2(out))
        out = self.gn_relu(self.cnn3(out))
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out
