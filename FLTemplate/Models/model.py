from abc import ABC, abstractmethod

import torch
from torch import nn


class Model(ABC):

    """
    A wrapper for PyTorch models that adds fairness-aware training and evaluation.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
