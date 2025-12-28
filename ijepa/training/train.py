"""
I-JEPA Training Loop

Main training logic for I-JEPA pretraining on CIFAR-10.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ijepa.models import IJEPA
from ijepa.data import MultiBlockMaskGenerator, get_cifar10_loaders
from ijepa.training.scheduler import cosine_scheduler, get_momentum_schedule
from ijepa.training.ema import update_ema


def ijepa_loss(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> torch.Tensor:
    """
    Compute I-JEPA loss: L2 distance between predictions and targets.

    Args:
        predictions: List of M predicted representations (B, num_patches, embed_dim)
        targets: List of M target representations (B, num_patches, embed_dim)

    Returns:
        Scalar loss value (mean L2 loss per patch)
    """
    raise NotImplementedError("ijepa_loss not yet implemented")


def train_one_epoch(
    model: IJEPA,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mask_generator: MultiBlockMaskGenerator,
    epoch: int,
    total_epochs: int,
    device: torch.device,
    ema_momentum_schedule: list[float],
    global_step: int,
) -> tuple[float, int]:
    """
    Train for one epoch.

    Args:
        model: IJEPA model
        dataloader: Training data loader
        optimizer: Optimizer
        mask_generator: Mask generator for context/target blocks
        epoch: Current epoch number
        total_epochs: Total number of epochs
        device: Device to train on
        ema_momentum_schedule: Pre-computed EMA momentum values per step
        global_step: Current global training step

    Returns:
        avg_loss: Average loss for the epoch
        global_step: Updated global step counter
    """
    raise NotImplementedError("train_one_epoch not yet implemented")


def train_ijepa(
    config: dict,
    checkpoint_path: str | None = None,
) -> IJEPA:
    """
    Full I-JEPA training loop.

    Args:
        config: Training configuration dictionary
        checkpoint_path: Optional path to resume from checkpoint

    Returns:
        Trained IJEPA model
    """
    raise NotImplementedError("train_ijepa not yet implemented")
