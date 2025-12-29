"""
Exponential Moving Average (EMA) Utilities

Used to update the target encoder from the context encoder.
"""

import torch
import torch.nn as nn


@torch.no_grad()
def update_ema(
    target_model: nn.Module,
    source_model: nn.Module,
    momentum: float,
) -> None:
    """
    Update target model parameters with EMA of source model.

    Formula: target = momentum * target + (1 - momentum) * source

    Args:
        target_model: Model to update (target encoder)
        source_model: Model to copy from (context encoder)
        momentum: EMA momentum (0.996 -> 1.0 during training)
    """
    for param_t, param_s in zip(
        target_model.parameters(), source_model.parameters()
    ):
        param_t.data.mul_(momentum).add_(param_s.data, alpha=1 - momentum)
