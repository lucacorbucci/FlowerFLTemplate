"""
Dataclass holding all configuration parameters for federated learning setup.

Includes client numbers, rounds, device/silo settings, sampling fractions, seeds, dataset info, partitioning, training hyperparameters, and preprocessors (scaler/encoder).
"""
from dataclasses import dataclass
from typing import Any

from sklearn.preprocessing import TargetEncoder


@dataclass
class Preferences:
    num_clients: int | None = None
    num_rounds: int | None = None
    cross_device: bool = False
    num_test_nodes: int | None = None
    num_validation_nodes: int | None = None
    num_train_nodes: int | None = None
    num_epochs: int = 1
    sampled_validation_nodes_per_round: int | None = None
    sampled_training_nodes_per_round: int | None = None
    sampled_test_nodes_per_round: int | None = None
    seed: int = 42
    node_shuffle_seed: int | None = None
    fed_dir: str | None = None
    fl_setting: str | None = None
    dataset_path: str | None = None
    sweep: bool = False
    dataset_name: str | None = None
    scaler: Any = None
    partitioner_type: str | None = None
    partitioner_alpha: float | None = None
    partitioner_by: str | None = None
    encoder: TargetEncoder | None = None

    task: str = "classification"

    batch_size: int = 32
    lr: float = 0.01
    optimizer: str = "adam"
    momentum: float = 0.9

    image_path : str | None = None
