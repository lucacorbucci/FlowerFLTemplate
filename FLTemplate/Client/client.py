from typing import Any

import torch
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from Models.regression_model import RegressionModel
from Models.simple_model import SimpleModel
from Models.utils import get_model
from torch import nn
from torch.utils.data import DataLoader

# from Training.training import test, train
from Utils.preferences import Preferences
from Utils.utils import get_optimizer, get_params, set_params


class FlowerClient(NumPyClient):
    def __init__(
        self, trainloader: DataLoader, valloader: DataLoader, preferences: Preferences, partition_id: int
    ) -> None:
        """
        Initializes a Flower client instance for federated learning.

        Sets up data loaders, device, preferences, and model (SimpleModel for classification or RegressionModel for regression).

        Args:
            trainloader (DataLoader): DataLoader for training data.
            valloader (DataLoader): DataLoader for validation data.
            preferences (Preferences): Configuration preferences for the FL setup.
            partition_id (int): Unique identifier for this client's data partition.

        Returns:
            None
        """
        super().__init__()

        self.partition_id = partition_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.preferences = preferences
        trained_model = get_model(dataset=preferences.dataset_name)
        if self.preferences.task == "classification":
            self.model = SimpleModel(
                model=trained_model,
                optimizer=get_optimizer(trained_model, self.preferences),
                criterion=nn.CrossEntropyLoss(),
                device=self.device,
            )
        elif self.preferences.task == "regression":
            self.model = RegressionModel(
                model=trained_model,
                optimizer=get_optimizer(trained_model, self.preferences),
                criterion=nn.MSELoss(),
                device=self.device,
            )
        else:
            error = f"Unknown task type: {self.preferences.task}"
            raise ValueError(error)

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Any]]:
        """
        Performs local training on the client's data using parameters received from the server.

        Updates the local model parameters over the specified number of epochs and returns updated parameters along with training metrics.

        Args:
            parameters (NDArrays): Model parameters from the server.
            config (dict[str, Scalar]): Configuration dictionary from the server.

        Returns:
            tuple[NDArrays, int, dict[str, Any]]: Updated model parameters, number of training examples, and training result dictionary (e.g., containing loss and accuracy).

        Raises:
            RuntimeError: If training fails due to device or model issues.
        """
        # copy parameters sent by the server into client's local model
        set_params(self.model.model, parameters)

        # do local training (call same function as centralised setting)
        for _ in range(self.preferences.num_epochs):
            result_dict = self.model.train(trainloader=self.trainloader, epochs=self.preferences.num_epochs)

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return get_params(self.model.model), len(self.trainloader), result_dict

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, int, dict[str, Any]]:
        """
        Evaluates the model using parameters received from the server on the client's validation set.

        Computes loss and other metrics (e.g., accuracy for classification, rmse/mae for regression).

        Args:
            parameters (NDArrays): Model parameters from the server.
            config (dict[str, Scalar]): Configuration dictionary from the server.

        Returns:
            tuple[float, int, dict[str, Any]]: Evaluation loss, number of validation examples, and evaluation result dictionary with metrics.

        Raises:
            RuntimeError: If evaluation fails due to device or model issues.
        """
        set_params(self.model.model, parameters)
        result_dict = self.model.evaluate(testloader=self.valloader)
        return float(result_dict["loss"]), len(self.valloader), result_dict
