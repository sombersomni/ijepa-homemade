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
    total_loss = 0.0
    total_patches = 0

    for pred, tgt in zip(predictions, targets):
        # L2 loss per patch: sum over embed_dim, then average
        patch_loss = (pred - tgt).pow(2).sum(dim=-1)  # (B, num_patches)
        total_loss = total_loss + patch_loss.sum()
        total_patches += patch_loss.numel()

    return total_loss / total_patches


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
    lr_schedule: list[float] | None = None,
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
        lr_schedule: Optional pre-computed learning rate schedule

    Returns:
        avg_loss: Average loss for the epoch
        global_step: Updated global step counter
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")

    for images, _ in pbar:
        images = images.to(device)

        # Update learning rate if schedule provided
        if lr_schedule is not None and global_step < len(lr_schedule):
            lr = lr_schedule[global_step]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Generate masks for this batch
        context_indices, target_indices_list = mask_generator()

        # Forward pass
        predictions, targets = model(images, context_indices, target_indices_list)

        # Compute loss
        loss = ijepa_loss(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA update
        if global_step < len(ema_momentum_schedule):
            momentum = ema_momentum_schedule[global_step]
            model.update_target_encoder(momentum)

        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/num_batches:.4f}',
        })

    avg_loss = total_loss / num_batches
    return avg_loss, global_step


def train_ijepa(
    config: dict,
    checkpoint_path: str | None = None,
) -> IJEPA:
    """
    Full I-JEPA training loop.

    Args:
        config: Training configuration dictionary containing:
            - img_size: Image size (default 32)
            - patch_size: Patch size (default 4)
            - encoder_embed_dim: Encoder embedding dim (default 192)
            - encoder_depth: Encoder depth (default 6)
            - encoder_num_heads: Encoder heads (default 6)
            - predictor_embed_dim: Predictor embed dim (default 96)
            - predictor_depth: Predictor depth (default 4)
            - batch_size: Batch size (default 256)
            - epochs: Number of epochs (default 100)
            - lr: Base learning rate (default 1e-3)
            - min_lr: Minimum learning rate (default 1e-5)
            - weight_decay: Weight decay (default 0.05)
            - warmup_epochs: Warmup epochs (default 10)
            - ema_momentum_start: Starting EMA momentum (default 0.996)
            - ema_momentum_end: Final EMA momentum (default 1.0)
            - num_targets: Number of target blocks (default 4)
            - target_scale: Target block scale range (default (0.15, 0.25))
            - context_scale: Context block scale range (default (0.75, 0.95))
            - data_dir: Data directory (default './data')
            - save_dir: Checkpoint save directory (default './checkpoints')
        checkpoint_path: Optional path to resume from checkpoint

    Returns:
        Trained IJEPA model
    """
    import os

    # Extract config with defaults
    img_size = config.get('img_size', 32)
    patch_size = config.get('patch_size', 4)
    encoder_embed_dim = config.get('encoder_embed_dim', 192)
    encoder_depth = config.get('encoder_depth', 6)
    encoder_num_heads = config.get('encoder_num_heads', 6)
    predictor_embed_dim = config.get('predictor_embed_dim', 96)
    predictor_depth = config.get('predictor_depth', 4)
    batch_size = config.get('batch_size', 256)
    epochs = config.get('epochs', 100)
    lr = config.get('lr', 1e-3)
    min_lr = config.get('min_lr', 1e-5)
    weight_decay = config.get('weight_decay', 0.05)
    warmup_epochs = config.get('warmup_epochs', 10)
    ema_momentum_start = config.get('ema_momentum_start', 0.996)
    ema_momentum_end = config.get('ema_momentum_end', 1.0)
    num_targets = config.get('num_targets', 4)
    target_scale = config.get('target_scale', (0.15, 0.25))
    context_scale = config.get('context_scale', (0.75, 0.95))
    data_dir = config.get('data_dir', './data')
    save_dir = config.get('save_dir', './checkpoints')

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = IJEPA(
        img_size=img_size,
        patch_size=patch_size,
        encoder_embed_dim=encoder_embed_dim,
        encoder_depth=encoder_depth,
        encoder_num_heads=encoder_num_heads,
        predictor_embed_dim=predictor_embed_dim,
        predictor_depth=predictor_depth,
    ).to(device)

    # Create optimizer (only context encoder and predictor)
    optimizer = AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Create data loader
    train_loader, _ = get_cifar10_loaders(
        batch_size=batch_size,
        data_dir=data_dir,
    )
    steps_per_epoch = len(train_loader)

    # Create schedules
    lr_schedule = cosine_scheduler(
        base_value=lr,
        final_value=min_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=warmup_epochs,
        warmup_start_value=0.0,
    )

    ema_schedule = get_momentum_schedule(
        base_momentum=ema_momentum_start,
        final_momentum=ema_momentum_end,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )

    # Create mask generator
    grid_size = img_size // patch_size
    mask_generator = MultiBlockMaskGenerator(
        input_size=(grid_size, grid_size),
        num_targets=num_targets,
        target_scale=target_scale,
        context_scale=context_scale,
    )

    # Resume from checkpoint if provided
    start_epoch = 0
    global_step = 0
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        print(f"Resumed from epoch {start_epoch}")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(start_epoch, epochs):
        avg_loss, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            mask_generator=mask_generator,
            epoch=epoch,
            total_epochs=epochs,
            device=device,
            ema_momentum_schedule=ema_schedule,
            global_step=global_step,
            lr_schedule=lr_schedule,
        )

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': config,
        }
        torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_latest.pt'))

    print("Training complete!")
    return model
