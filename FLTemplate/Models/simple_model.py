import torch
from torch import nn
from torch.utils.data import DataLoader

from Models.model import Model


class SimpleModel(Model):

    """
    A wrapper for PyTorch models that adds fairness-aware training and evaluation.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> None:
        """
        Initialize the SimpleModel wrapper.

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
    ) -> dict[str, float]:
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
        criterion = torch.nn.CrossEntropyLoss()
        self.model.to(self.device)
        self.model.train()
        losses = 0.0
        correct = 0
        for sample, _, label in trainloader:
            sample, labels = sample.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(sample)
            loss = criterion(output, labels.long())
            loss.backward()
            self.optimizer.step()
            losses += loss.item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).sum().item()

        loss = torch.tensor(losses / len(trainloader), device=self.device)
        accuracy = correct / len(trainloader.dataset)  # type: ignore
        return {"loss": loss.item(), "accuracy": accuracy}

    def evaluate(self, testloader: DataLoader) -> dict[str, float]:
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
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        losses = 0.0

        with torch.no_grad():
            for sample, _, label in testloader:
                images, labels = sample.to(self.device), label.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels.long())
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                losses += loss.item()

        loss = torch.tensor(losses / len(testloader), device=self.device)
        accuracy = correct / len(testloader.dataset)  # type: ignore
        return {"loss": loss.item(), "accuracy": accuracy}
