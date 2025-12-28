"""
I-JEPA: Image-based Joint-Embedding Predictive Architecture

Combines context encoder, target encoder (EMA), and predictor into a single model.
"""

import copy

import torch
import torch.nn as nn

from ijepa.models.vit import ViTEncoder
from ijepa.models.predictor import Predictor


class IJEPA(nn.Module):
    """
    Full I-JEPA model.

    Components:
        - context_encoder: Trainable ViT encoder, processes only visible context patches
        - target_encoder: EMA copy of context encoder, processes full image
        - predictor: Predicts target representations from context

    Args:
        img_size: Input image size
        patch_size: Patch size
        in_channels: Number of input channels
        encoder_embed_dim: Encoder embedding dimension
        encoder_depth: Number of encoder transformer blocks
        encoder_num_heads: Number of encoder attention heads
        predictor_embed_dim: Predictor embedding dimension (bottleneck)
        predictor_depth: Number of predictor transformer blocks
        predictor_num_heads: Number of predictor attention heads
        mlp_ratio: MLP hidden dim ratio
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        encoder_embed_dim: int = 192,
        encoder_depth: int = 6,
        encoder_num_heads: int = 6,
        predictor_embed_dim: int = 96,
        predictor_depth: int = 4,
        predictor_num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        # TODO: Initialize components
        # self.context_encoder = ViTEncoder(...)
        # self.target_encoder = copy.deepcopy(self.context_encoder)
        # self.target_encoder.requires_grad_(False)  # No gradients for EMA
        # self.predictor = Predictor(...)
        raise NotImplementedError("IJEPA not yet implemented")

    def forward(
        self,
        images: torch.Tensor,
        context_indices: list[list[int]],
        target_indices_list: list[list[list[int]]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Forward pass for I-JEPA training.

        Args:
            images: Input images (B, C, H, W)
            context_indices: List of context patch indices for each sample
            target_indices_list: List of M target block indices for each sample
                                Shape: [batch_size][num_target_blocks][patches_per_block]

        Returns:
            predictions: List of M predicted representations, each (B, num_patches, embed_dim)
            targets: List of M target representations (detached), each (B, num_patches, embed_dim)
        """
        raise NotImplementedError("IJEPA.forward not yet implemented")

    def get_target_encoder(self) -> ViTEncoder:
        """
        Get the target encoder for downstream evaluation.

        Returns:
            The EMA-updated target encoder
        """
        raise NotImplementedError("IJEPA.get_target_encoder not yet implemented")

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """
        Update target encoder with EMA of context encoder.

        Args:
            momentum: EMA momentum (0.996 -> 1.0 during training)
        """
        raise NotImplementedError("IJEPA.update_target_encoder not yet implemented")
