import os

import numpy as np
import pandas as pd
from Client.client import FlowerClient
from Datasets.abalone import AbaloneDataset, get_abalone_scaler, prepare_abalone, prepare_abalone_for_cross_silo
from Datasets.dutch import get_dutch_scaler, prepare_dutch, prepare_dutch_for_cross_silo
from Datasets.mnist import download_mnist, prepare_mnist, prepare_mnist_for_cross_silo
from Datasets.tabular_datasets import TabularDataset
from flwr.common import Context
from torch.utils.data import DataLoader
from Utils.preferences import Preferences


def get_data_info(preferences: Preferences):
    match preferences.dataset_name:
        case "dutch":
            df = pd.read_csv(preferences.dataset_path)
            scaler = get_dutch_scaler(
                sweep=preferences.sweep,
                seed=preferences.seed,
                dutch_df=df,
                validation_seed=preferences.node_shuffle_seed,
            )

            return {"data_type": "csv", "target": "occupation", "sensitive_attribute": "sex", "scaler": scaler}

        case "mnist":
            if not os.path.exists(os.path.join(preferences.dataset_path)):
                download_mnist()
            return {"data_type": "imagefolder"}
        case "abalone":
            df = pd.read_csv(preferences.dataset_path)
            scaler = get_abalone_scaler(
                sweep=preferences.sweep,
                seed=preferences.seed,
                abalone_df=df,
                validation_seed=preferences.node_shuffle_seed,
            )

            return {"data_type": "csv", "target": "Rings", "scaler": scaler}

        case _:
            raise ValueError(f"Unsupported dataset: {preferences.dataset_name}")


def prepare_data_for_cross_device(context: Context, partition, preferences: Preferences):
    if preferences.dataset_name == "dutch":
        train = partition.to_pandas()
        x_train, z_train, y_train, _ = prepare_dutch(
            dutch_df=train,
            scaler=preferences.scaler,
        )
        train_dataset = TabularDataset(
            x=np.hstack((x_train, np.ones((x_train.shape[0], 1)))).astype(np.float32),
            z=z_train.astype(np.float32),
            y=y_train.astype(np.float32),
        )
        trainloader = DataLoader(train_dataset, batch_size=preferences.batch_size, shuffle=True)
    elif preferences.dataset_name == "mnist":
        trainloader = prepare_mnist(partition, preferences)
    elif preferences.dataset_name == "abalone":
        train = partition.to_pandas()
        x_train, y_train, _ = prepare_abalone(
            abalone_df=train,
            scaler=preferences.scaler,
        )
        train_dataset = AbaloneDataset(
            X=x_train,
            y=y_train,
        )
        trainloader = DataLoader(train_dataset, batch_size=preferences.batch_size, shuffle=True)
    else:
        raise ValueError(f"Unsupported dataset: {preferences.dataset_name}")

    return FlowerClient(trainloader=trainloader, valloader=trainloader, preferences=preferences).to_client()


def prepare_data_for_cross_silo(context: Context, partition, preferences: Preferences):
    if preferences.dataset_name == "dutch":
        return prepare_dutch_for_cross_silo(preferences, partition)
    elif preferences.dataset_name == "mnist":
        return prepare_mnist_for_cross_silo(preferences, partition)
    elif preferences.dataset_name == "abalone":
        return prepare_abalone_for_cross_silo(preferences, partition)
    else:
        raise ValueError(f"Unsupported dataset: {preferences.dataset_name}")
