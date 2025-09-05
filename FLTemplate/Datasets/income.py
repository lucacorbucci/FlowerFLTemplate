import os
from typing import Any

import numpy as np
import pandas as pd
from Client.client import FlowerClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, TargetEncoder
from torch.utils.data import DataLoader, Dataset
from Utils.preferences import Preferences


class IncomeDataset(Dataset):
    def __init__(self, x: np.ndarray, z: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the custom dataset with x (features), z (sensitive values), and y (targets).

        Args:
        x (list of tensors): List of input feature tensors.
        z (list): List of sensitive values.
        y (list): List of target values.

        """
        self.samples = x
        self.sensitive_features = z
        self.targets = y
        self.indexes = range(len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, Any, Any]:
        """
        Get a single data point from the dataset.

        Args:
        idx (int): Index to retrieve the data point.

        Returns:
        sample (dict): A dictionary containing 'x', 'z', and 'y'.

        """
        x_sample = self.samples[idx]
        z_sample = self.sensitive_features[idx]
        y_sample = self.targets[idx]

        return x_sample, z_sample, y_sample


def get_income_scaler(
    sweep: bool,
    seed: int,
    df: pd.DataFrame | None = None,
    validation_seed: int | None = None,
) -> tuple[StandardScaler, TargetEncoder]:
    if df is None:
        error = "df cannot be None"
        raise ValueError(error)
    if type(df) is not pd.DataFrame:
        error = "df must be a pandas DataFrame"
        raise ValueError(error)

    _, _, _, scaler, encoder = prepare_income(
        df=df,
    )
    return scaler, encoder


def prepare_income(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    encoder: TargetEncoder | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, TargetEncoder]:
    # Separate features and target

    categorical_columns = ["COW", "SCHL", "RAC1P"]
    continuous_columns = ["AGEP", "WKHP", "OCCP", "POBP", "RELP"]

    # get the target and sensitive attributes
    target_attributes = df[">50K"].values
    sensitive_attributes = df["SEX"].values

    df = df.drop(
        columns=[
            ">50K",
            "SEX",
        ]
    )

    if encoder is None:
        encoder = TargetEncoder(smooth="auto").fit(df[categorical_columns], target_attributes)
    df[categorical_columns] = encoder.transform(df[categorical_columns])

    # normalize the continuous using standard scaler
    if scaler is None:
        scaler = StandardScaler().fit(df[continuous_columns])
    df[continuous_columns] = scaler.transform(df[continuous_columns])

    # convert to numpy arrays
    x_train = df.to_numpy(dtype=np.float32)

    return x_train, np.array(sensitive_attributes), np.array(target_attributes), scaler, encoder


def prepare_income_for_cross_silo(preferences: Preferences, partition_id: int) -> Any:
    path = f"{preferences.dataset_path}/{partition_id}/"
    for file in os.listdir(path):
        if file.endswith(".csv"):
            if "train" in file:
                train = pd.read_csv(f"{preferences.dataset_path}/{partition_id}/{file}")
            elif "test" in file:
                test = pd.read_csv(f"{preferences.dataset_path}/{partition_id}/{file}")

    if preferences.sweep:
        print("[Preparing data for cross-silo for sweep...]")

        train, val = train_test_split(train, test_size=0.2, random_state=preferences.node_shuffle_seed)

        x_train, z_train, y_train, _, _ = prepare_income(
            df=train,
            scaler=preferences.scaler,
            encoder=preferences.encoder,
        )

        x_val, z_val, y_val, _, _ = prepare_income(
            df=val,
            scaler=preferences.scaler,
            encoder=preferences.encoder,
        )

        train_dataset = IncomeDataset(
            x=np.hstack((x_train, np.ones((x_train.shape[0], 1)))).astype(np.float32),
            z=z_train.astype(np.float32),
            y=y_train.astype(np.float32),
        )
        val_dataset = IncomeDataset(
            x=np.hstack((x_val, np.ones((x_val.shape[0], 1)))).astype(np.float32),
            z=z_val.astype(np.float32),
            y=y_val.astype(np.float32),
        )

        trainloader = DataLoader(train_dataset, batch_size=preferences.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=preferences.batch_size, shuffle=False)

        return FlowerClient(
            trainloader=trainloader, valloader=val_loader, preferences=preferences, partition_id=partition_id
        ).to_client()
    print("[Preparing data for cross-silo...]")

    x_train, z_train, y_train, _, _ = prepare_income(
        df=train,
        scaler=preferences.scaler,
        encoder=preferences.encoder,
    )

    x_test, z_test, y_test, _, _ = prepare_income(
        df=test,
        scaler=preferences.scaler,
        encoder=preferences.encoder,
    )

    train_dataset = IncomeDataset(
        x=np.hstack((x_train, np.ones((x_train.shape[0], 1)))).astype(np.float32),
        z=z_train.astype(np.float32),
        y=y_train.astype(np.float32),
    )
    test_dataset = IncomeDataset(
        x=np.hstack((x_test, np.ones((x_test.shape[0], 1)))).astype(np.float32),
        z=z_test.astype(np.float32),
        y=y_test.astype(np.float32),
    )

    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    trainloader = DataLoader(train_dataset, batch_size=preferences.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=preferences.batch_size, shuffle=False)

    return FlowerClient(trainloader=trainloader, valloader=test_loader, preferences=preferences).to_client()
