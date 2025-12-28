"""
CIFAR-10 Data Loading Utilities

Provides train and test dataloaders for CIFAR-10 dataset.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def get_cifar10_loaders(
    batch_size: int = 256,
    data_dir: str = './data',
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 train and test data loaders.

    For I-JEPA pretraining, we use simple augmentations:
    - Random horizontal flip
    - Normalize to [-1, 1] or use ImageNet stats

    Args:
        batch_size: Batch size for both loaders
        data_dir: Directory to download/load data
        num_workers: Number of data loading workers

    Returns:
        train_loader: Training data loader
        test_loader: Test data loader
    """
    raise NotImplementedError("get_cifar10_loaders not yet implemented")
