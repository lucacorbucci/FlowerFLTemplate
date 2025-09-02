from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Preferences:
    num_clients: int = None
    num_rounds: int = None
    cross_device: bool = False
    num_test_nodes: int = None
    num_validation_nodes: int = None
    num_train_nodes: int = None
    num_epochs: int = 1
    sampled_validation_nodes_per_round: int = None
    sampled_training_nodes_per_round: int = None
    sampled_test_nodes_per_round: int = None
    seed: int = 42
    node_shuffle_seed: int = 11
    fed_dir: str = None
    fl_setting: str = None
    dataset_path: str = None
    sweep: bool = False
    dataset_name: str = None
    scaler: Any = None
    partitioner_type: str = None
    partitioner_alpha: float = None
    partitioner_by: str = None
