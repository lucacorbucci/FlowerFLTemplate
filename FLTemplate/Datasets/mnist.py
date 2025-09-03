
import os

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from Client.client import FlowerClient
from Datasets.tabular_datasets import TabularDataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from Utils.preferences import Preferences


class ImageDataset(Dataset):
    """
    Custom Dataset class that handles images, labels, and sensitive attributes."""
    def __init__(self, data, transform):
        """
        Initialize the dataset.

        Args:
            data: List of dictionaries or dataset containing 'image', 'label', and 'sensitive_attribute'
            transform: Optional transform to be applied to images
        """
        self.data = data
        self.transform = transform


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Tuple containing (image, label, sensitive_attribute)
        """
        # Get the example at the given index
        example = self.data[idx]

        # Extract image, label, and sensitive attribute
        image = example['image']
        label = example['label']
        sensitive_attribute = example.get('sensitive_attribute', -1)

        # Apply transforms to the image
        if self.transform:
            image = self.transform(image)
            

        return image, sensitive_attribute, label

def download_mnist(data_root='../data/'):
    """
    Downloads the MNIST dataset and saves the images as PNG files
    into separate directories for each class (0-9).

    Args:
        data_root (str): The directory to store the dataset.
    """
    print("Starting download of MNIST dataset...")
    # Download the training and testing datasets
    transformer = transforms.ToTensor()
    mnist_train = torchvision.datasets.MNIST("../data", train=True, download=True, transform=transformer)
    mnist_test = torchvision.datasets.MNIST("../data", train=False, download=True, transform=transformer)

    # Combine training and testing data for a complete dataset
    full_dataset = torch.utils.data.ConcatDataset([mnist_train, mnist_test])

    # Create root directory for saving images
    save_root = os.path.join(data_root, 'MNIST/train/')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # Create a subfolder for each digit class
    for i in range(10):
        class_dir = os.path.join(save_root, str(i))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    # Iterate through the full dataset and save images
    for i, (image_tensor, label) in enumerate(full_dataset):
        # Convert the PyTorch tensor to a PIL Image
        # The image tensor is 1x28x28, so we need to squeeze it to 28x28
        image = Image.fromarray((image_tensor.squeeze() * 255).numpy().astype('uint8'))
        
        # Define the save path
        save_path = os.path.join(save_root, str(label), f'{label}_{i}.png')
        
        # Save the image
        image.save(save_path)
        
        # Print progress every 1000 images
        if (i + 1) % 10000 == 0:
            print(f"Saved {i + 1} images...")

    return full_dataset


def prepare_mnist(partition, preferences):
    train = partition

    train_dataset = ImageDataset(train, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    trainloader = DataLoader(
            train_dataset,
            batch_size=preferences.batch_size,
            shuffle=True
        )
    
    return trainloader

def prepare_mnist_for_cross_silo(preferences: Preferences, partition):
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    if preferences.sweep:
        print("[Preparing data for cross-silo for sweep...]")

        partition_loader_train_val = partition_train_test["train"].train_test_split(
            test_size=0.2, seed=preferences.node_shuffle_seed
        )
        train = partition_loader_train_val["train"]
        val = partition_loader_train_val["test"]

        trainloader = prepare_mnist(train, preferences)
        valloader = prepare_mnist(val, preferences)

        return FlowerClient(trainloader=trainloader, valloader=valloader, preferences=preferences).to_client()
    else:
        print("[Preparing data for cross-silo...]")

        train = partition_train_test["train"]
        test = partition_train_test["test"]


        trainloader = prepare_mnist(train, preferences)
        testloader = prepare_mnist(test, preferences)

        return FlowerClient(trainloader=trainloader, valloader=testloader, preferences=preferences).to_client()