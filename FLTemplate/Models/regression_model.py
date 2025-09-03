import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader

from Models.model import Model


class RegressionModel(Model):

    """
    A wrapper for PyTorch models that adds fairness-aware training and evaluation.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> None:
        """
        Initialize the RegressionModel wrapper.

        Args:
            model (nn.Module): The PyTorch model to wrap
            optimizer (torch.optim.Optimizer, optional): Optimizer for training. If None,
                                                         Adam will be used with default params.
            criterion (nn.Module): Loss function for training, default is cross entropy
            device (torch.device): Device to use for computation (CPU/GPU)

        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)

    def train(
        self,
        trainloader: DataLoader,
        epochs: int,
    ) -> (float, float):
        """
        Train the model with optional fairness regularization.

        Args:
            train_loader (DataLoader): DataLoader for the training data
            epochs (int): Number of epochs to train for
            val_loader (DataLoader, optional): DataLoader for validation data
            verbose (bool): Whether to print progress during training
            average_probabilities (Dict, optional): Dictionary of probabilities for FL if not all sensitive attributes present

        Returns:
            Dict[str, List[float]]: Dictionary of metrics tracked during training

        """
        # Initialize tracking metrics

        self.model.to(self.device)
        self.model.train()
        losses = 0.0
        for sample, _, label in trainloader:
            sample, labels = sample.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(sample)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            losses += loss.item()

        loss = torch.tensor(losses / len(trainloader), device=self.device)
        return {"loss": loss.item()}

    def evaluate(self, testloader: DataLoader) -> (float, float):
        """
        Evaluate the model on a dataset.

        Args:
            data_loader (DataLoader): DataLoader for evaluation
            is_validation (bool): Whether this is a validation set

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics

        """
        self.model.to(self.device)
        self.model.eval()
        losses = 0.0

        predictions = []
        actuals = []

        with torch.no_grad():
            for sample, _, label in testloader:
                images, labels = sample.to(self.device), label.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                losses += loss.item()
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(label.numpy())

        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals).flatten()

        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        rmse = np.sqrt(mse)

        loss = torch.tensor(losses / len(testloader), device=self.device)

        return {"loss": loss.item(), "rmse": rmse, "mae": mae, "r2": r2, "mse": mse}
