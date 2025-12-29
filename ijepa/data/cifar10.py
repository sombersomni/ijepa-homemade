"""
CIFAR-10 Data Loading Utilities

Provides train and test dataloaders for CIFAR-10 dataset.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# CIFAR-10 normalization statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(
    batch_size: int = 256,
    data_dir: str = './data',
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 train and test data loaders.

    For I-JEPA pretraining, we use simple augmentations:
    - Random horizontal flip
    - Normalize using CIFAR-10 statistics

    Args:
        batch_size: Batch size for both loaders
        data_dir: Directory to download/load data
        num_workers: Number of data loading workers

    Returns:
        train_loader: Training data loader
        test_loader: Test data loader
    """
    # Training transforms: minimal augmentation for self-supervised learning
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Test transforms: just normalize
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Consistent batch sizes for training
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader
