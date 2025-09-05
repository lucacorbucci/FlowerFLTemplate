import torch

from Models.architectures import AbaloneNet, CelebaNet, LinearClassificationNet, SimpleMNISTModel


def get_model(
    dataset: str,
) -> torch.nn.Module:
    """
    This function returns the model to train.

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
    if dataset == "dutch":
        return LinearClassificationNet(input_size=12, output_size=2)
    if dataset == "income":
        return LinearClassificationNet(input_size=10, output_size=2)
    if dataset == "adult":
        return LinearClassificationNet(input_size=103, output_size=2)
    if dataset == "mnist":
        return SimpleMNISTModel()
    if dataset == "abalone":
        return AbaloneNet(input_size=8, hidden_sizes=[128, 64, 32], dropout_rate=0.2)

    raise ValueError(f"Dataset {dataset} not supported")
