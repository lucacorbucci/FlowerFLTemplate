from collections import OrderedDict

import torch
from flwr.common import NDArrays


# Two auxhiliary functions to set and extract parameters of a model
def set_params(model: torch.nn.Module, parameters: NDArrays) -> None:
    """Replace model parameters with those passed as `parameters`."""
    params_dict = zip(model.state_dict().keys(), parameters, strict=False)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    # now replace the parameters
    model.load_state_dict(state_dict, strict=True)


def get_params(model: torch.nn.Module) -> NDArrays:
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def get_optimizer(model: torch.nn.Module, preferences) -> torch.optim.Optimizer:
    match preferences.optimizer.lower():
        case "sgd":
            return torch.optim.SGD(model.parameters(), lr=preferences.lr, momentum=preferences.momentum)
        case "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=preferences.lr,
                weight_decay=preferences.weight_decay if hasattr(preferences, "weight_decay") else 0,
            )
        case _:
            raise ValueError(f"Unsupported optimizer: {preferences.optimizer}")
