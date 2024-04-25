from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Preferences:
    dataset: str
    dataset_path: str
    # The number of epochs that we want to use during the training
    epochs: int
    batch_size: int
    # seed to use during the training
    seed: int
    optimizer: str
    learning_rate: float
    # regularization is True if we want to use the regularization to
    # reduce the unfairness of the model during the training
    epsilon: float = None
    sweep: bool = False
    fl_rounds: int = None
    noise_multiplier: float = None
    cross_device: bool = False
    tabular: bool = False
    num_nodes: int = None
    split_approach: str = None
    alpha_dirichlet: float = None
    ratio_unfair_nodes: float = None
    group_to_reduce: tuple = None
    group_to_increment: tuple = None
    ratio_unfairness: tuple = None
    validation: bool = False
