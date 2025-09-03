import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn

from Models.architectures import AbaloneNet, CelebaNet, LinearClassificationNet, SimpleMNISTModel


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
    elif dataset == "abalone":
        return AbaloneNet(input_size=8, hidden_sizes=[128, 64, 32], dropout_rate=0.2)

    else:
        raise ValueError(f"Dataset {dataset} not supported")
