from typing import Any

import numpy as np
import pandas as pd
import torch
from Client.client import FlowerClient
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from Utils.preferences import Preferences


class AbaloneDataset(Dataset):
    """Custom Dataset for Abalone data"""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y).view(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Any, Any, Any]:
        return self.X[idx], -1, self.y[idx]


def get_abalone_scaler(
    sweep: bool,
    seed: int,
    abalone_df: pd.DataFrame | None = None,
    validation_seed: int | None = None,
) -> StandardScaler:
    if abalone_df is None:
        error = "abalone_df cannot be None"
        raise ValueError(error)
    if type(abalone_df) is not pd.DataFrame:
        error = "abalone_df must be a pandas DataFrame"
        raise ValueError(error)

    _, _, scaler = prepare_abalone(
        abalone_df=abalone_df,
    )
    return scaler


def prepare_abalone(
    abalone_df: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    # Separate features and target
    x = abalone_df.drop("Rings", axis=1)
    y_train = abalone_df["Rings"].values  # Age = Rings + 1.5, but we'll predict rings directly

    # Encode categorical variable (Sex)
    le = LabelEncoder()
    x["Sex"] = le.fit_transform(x["Sex"])

    # Convert to numpy array
    x = x.values

    # Scale the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x)

    print(f"\nTraining set size: {x_train.shape[0]}")
    print(f"Number of features: {x_train.shape[1]}")

    return x_train, np.array(y_train), scaler


def prepare_abalone_for_cross_silo(preferences: Preferences, partition: Any) -> Any:
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    if preferences.sweep:
        print("[Preparing data for cross-silo for sweep...]")

        partition_loader_train_val = partition_train_test["train"].train_test_split(
            test_size=0.2, seed=preferences.node_shuffle_seed
        )
        train = partition_loader_train_val["train"].to_pandas()
        val = partition_loader_train_val["test"].to_pandas()

        x_train, y_train, _ = prepare_abalone(
            abalone_df=train,
            scaler=preferences.scaler,
        )

        x_val, y_val, _ = prepare_abalone(
            abalone_df=val,
            scaler=preferences.scaler,
        )

        train_dataset = AbaloneDataset(
            x=x_train,
            y=y_train,
        )
        val_dataset = AbaloneDataset(
            x=x_val,
            y=y_val,
        )

        trainloader = DataLoader(train_dataset, batch_size=preferences.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=preferences.batch_size, shuffle=False)

        return FlowerClient(trainloader=trainloader, valloader=val_loader, preferences=preferences).to_client()
    print("[Preparing data for cross-silo...]")
    train = partition_train_test["train"].to_pandas()
    test = partition_train_test["test"].to_pandas()

    x_train, y_train, _ = prepare_abalone(
        abalone_df=train,
        scaler=preferences.scaler,
    )

    x_test, y_test, _ = prepare_abalone(
        abalone_df=test,
        scaler=preferences.scaler,
    )

    train_dataset = AbaloneDataset(
        x=x_train,
        y=y_train,
    )
    test_dataset = AbaloneDataset(
        x=x_test,
        y=y_test,
    )

    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    trainloader = DataLoader(train_dataset, batch_size=preferences.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=preferences.batch_size, shuffle=False)

    return FlowerClient(trainloader=trainloader, valloader=test_loader, preferences=preferences).to_client()
