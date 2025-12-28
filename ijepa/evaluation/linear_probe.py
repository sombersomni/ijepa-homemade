"""
Linear Probe Evaluation for I-JEPA

Evaluate learned representations by training a linear classifier
on frozen features from the target encoder.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ijepa.models import ViTEncoder


@torch.no_grad()
def extract_features(
    encoder: ViTEncoder,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features from encoder for all samples in dataloader.

    Features are computed by average pooling all patch representations.

    Args:
        encoder: Frozen target encoder
        dataloader: Data loader with images and labels
        device: Device to run on

    Returns:
        features: (N, embed_dim) feature vectors
        labels: (N,) class labels
    """
    raise NotImplementedError("extract_features not yet implemented")


def train_linear_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    embed_dim: int,
    num_classes: int,
    epochs: int = 100,
    lr: float = 0.1,
    batch_size: int = 256,
    device: torch.device = torch.device('cpu'),
) -> nn.Linear:
    """
    Train a linear classifier on extracted features.

    Args:
        features: (N, embed_dim) training features
        labels: (N,) training labels
        embed_dim: Feature dimension
        num_classes: Number of output classes
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        device: Device to train on

    Returns:
        Trained linear classifier
    """
    raise NotImplementedError("train_linear_classifier not yet implemented")


def evaluate_linear_probe(
    config: dict,
    checkpoint_path: str,
) -> float:
    """
    Full linear probe evaluation pipeline.

    1. Load target encoder from checkpoint
    2. Extract features from train and test sets
    3. Train linear classifier on train features
    4. Evaluate on test features

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to I-JEPA checkpoint

    Returns:
        Test accuracy (0-100%)
    """
    raise NotImplementedError("evaluate_linear_probe not yet implemented")
