
from logging import INFO, WARNING

import numpy as np
import torch
import torchvision.transforms as transforms
from Client.client import FlowerClient
from flwr.common import Context
from torch.utils.data import DataLoader
from torchvision import transforms
from Utils.preferences import Preferences

from Datasets.dutch import prepare_dutch
from Datasets.tabular_datasets import TabularDataset


def collate_fn(examples):
    """
    Collate function to prepare batches for the DataLoader.
    It applies the transforms and stacks the tensors.
    """
    images = []
    labels = []
    for example in examples:
        # Get the PIL image and label from the example
        image = example['image']
        label = example['label']

        # Define transformations
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        
        # Apply transformations to the image
        if transform:
            image = transform(image)
        
        images.append(image)
        labels.append(label)
    
    # Stack the images and labels into tensors
    return {
        'images': torch.stack(images),
        'labels': torch.tensor(labels)
    }


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
        trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    elif preferences.dataset_name == "mnist":
        train = partition
        print(partition)
        print(type(partition))
        
        
        
        trainloader =  DataLoader(
            train,
            batch_size=32,
            shuffle=True,  # Important for training
            collate_fn=collate_fn
        )
    else:
        raise ValueError(f"Unsupported dataset: {preferences.dataset_name}")

    return FlowerClient(trainloader=trainloader, valloader=trainloader).to_client()

def prepare_data_for_cross_silo(context: Context, partition, preferences: Preferences):
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    if preferences.sweep:
        log(INFO, "[Preparing data for cross-silo for sweep...]")
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

        return FlowerClient(trainloader=trainloader, valloader=val_loader).to_client()
    else:
        log(INFO, "[Preparing data for cross-silo...]")
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

        return FlowerClient(trainloader=trainloader, valloader=test_loader).to_client()