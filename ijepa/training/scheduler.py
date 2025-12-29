"""
Learning Rate and Momentum Schedulers for I-JEPA

Provides:
- Cosine learning rate schedule with warmup
- Linear EMA momentum schedule (0.996 -> 1.0)
"""

import numpy as np


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 0,
    warmup_start_value: float = 0.0,
) -> np.ndarray:
    """
    Create a cosine annealing schedule with optional warmup.

    Args:
        base_value: Peak value (after warmup)
        final_value: Final value at end of training
        epochs: Total number of epochs
        steps_per_epoch: Number of steps per epoch
        warmup_epochs: Number of warmup epochs
        warmup_start_value: Starting value during warmup

    Returns:
        Array of scheduled values for each step
    """
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    schedule = np.zeros(total_steps)

    # Warmup phase: linear increase
    if warmup_steps > 0:
        schedule[:warmup_steps] = np.linspace(
            warmup_start_value, base_value, warmup_steps
        )

    # Cosine annealing phase
    cosine_steps = total_steps - warmup_steps
    if cosine_steps > 0:
        cosine_schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * np.arange(cosine_steps) / cosine_steps)
        )
        schedule[warmup_steps:] = cosine_schedule

    return schedule


def get_momentum_schedule(
    base_momentum: float,
    final_momentum: float,
    epochs: int,
    steps_per_epoch: int,
) -> np.ndarray:
    """
    Create a linear momentum schedule for EMA updates.

    Linearly increases from base_momentum to final_momentum.

    Args:
        base_momentum: Starting momentum (e.g., 0.996)
        final_momentum: Final momentum (e.g., 1.0)
        epochs: Total number of epochs
        steps_per_epoch: Number of steps per epoch

    Returns:
        Array of momentum values for each step
    """
    total_steps = epochs * steps_per_epoch
    return np.linspace(base_momentum, final_momentum, total_steps)
