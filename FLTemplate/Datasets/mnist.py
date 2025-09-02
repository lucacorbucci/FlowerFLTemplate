
import os

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def download_mnist(data_root='../data/'):
    """
    Downloads the MNIST dataset and saves the images as PNG files
    into separate directories for each class (0-9).

    Args:
        data_root (str): The directory to store the dataset.
    """
    print("Starting download of MNIST dataset...")
    # Download the training and testing datasets
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
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

