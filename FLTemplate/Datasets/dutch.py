import numpy as np
import pandas as pd
from Client.client import FlowerClient
from Datasets.tabular_datasets import TabularDataset
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from Utils.preferences import Preferences


def get_dutch_scaler(
    sweep: bool,
    seed: int,
    dutch_df: pd.DataFrame | None = None,
    validation_seed: int | None = None,
) -> MinMaxScaler:
    if dutch_df is None:
        raise ValueError("dutch_df cannot be None")

    _, _, _, scaler = prepare_dutch(
        dutch_df=dutch_df,
    )
    return scaler


def prepare_dutch(
    dutch_df: pd.DataFrame,
    scaler: MinMaxScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    # check the columns with missign values:
    missing_values_columns = dutch_df.columns[dutch_df.isna().any()].tolist()
    for column in missing_values_columns:
        dutch_df[column] = dutch_df[column].fillna(dutch_df[column].mode()[0])

    if len(dutch_df.columns[dutch_df.isna().any()].tolist()) != 0:
        error_message = "There are still missing values in the dataset"
        raise ValueError(error_message)

    dutch_df["sex_binary"] = np.where(dutch_df["sex"] == 1, 1, 0)
    dutch_df["occupation_binary"] = np.where(dutch_df["occupation"] >= 300, 1, 0)

    del dutch_df["sex"]
    del dutch_df["occupation"]

    y_train = dutch_df["occupation_binary"].astype(int).values
    z_train = dutch_df["sex_binary"].astype(int).values
    del dutch_df["occupation_binary"]
    dutch_df = pd.get_dummies(dutch_df, columns=None, drop_first=False)

    if scaler is None:
        scaler = MinMaxScaler()
    x_train = scaler.fit_transform(dutch_df)

    return x_train, np.array(z_train), np.array(y_train), scaler


def prepare_dutch_for_cross_silo(preferences: Preferences, partition):
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    if preferences.sweep:
        print("[Preparing data for cross-silo for sweep...]")


        partition_loader_train_val = partition_train_test["train"].train_test_split(
            test_size=0.2, seed=preferences.node_shuffle_seed
        )
        train = partition_loader_train_val["train"]
        val = partition_loader_train_val["test"]

        x_train, z_train, y_train, _ = prepare_dutch(
            dutch_df=train,
            scaler=preferences.scaler,
        )

        x_val, z_val, y_val, _ = prepare_dutch(
            dutch_df=val,
            scaler=preferences.scaler,
        )

        train_dataset = TabularDataset(
            x=np.hstack((x_train, np.ones((x_train.shape[0], 1)))).astype(np.float32),
            z=z_train.astype(np.float32),
            y=y_train.astype(np.float32),
        )
        val_dataset = TabularDataset(
            x=np.hstack((x_val, np.ones((x_val.shape[0], 1)))).astype(np.float32),
            z=z_val.astype(np.float32),
            y=y_val.astype(np.float32),
        )

        trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        return FlowerClient(trainloader=trainloader, valloader=val_loader, preferences=preferences).to_client()
    else:
        print("[Preparing data for cross-silo...]")
        train = partition_train_test["train"].to_pandas()
        test = partition_train_test["test"].to_pandas()

        x_train, z_train, y_train, _ = prepare_dutch(
            dutch_df=train,
            scaler=preferences.scaler,
        )

        x_test, z_test, y_test, _ = prepare_dutch(
            dutch_df=test,
            scaler=preferences.scaler,
        )

        train_dataset = TabularDataset(
            x=np.hstack((x_train, np.ones((x_train.shape[0], 1)))).astype(np.float32),
            z=z_train.astype(np.float32),
            y=y_train.astype(np.float32),
        )
        test_dataset = TabularDataset(
            x=np.hstack((x_test, np.ones((x_test.shape[0], 1)))).astype(np.float32),
            z=z_test.astype(np.float32),
            y=y_test.astype(np.float32),
        )

        print("Train dataset size:", len(train_dataset))
        print("Test dataset size:", len(test_dataset))

        trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return FlowerClient(trainloader=trainloader, valloader=test_loader, preferences=preferences).to_client()