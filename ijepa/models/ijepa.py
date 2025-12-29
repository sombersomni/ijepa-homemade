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
        self.encoder_embed_dim = encoder_embed_dim

        # Context encoder: trainable, processes only visible context patches
        self.context_encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
        )

        # Target encoder: EMA copy, processes full image, no gradients
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)

        # Predictor: predicts target representations from context
        self.predictor = Predictor(
            num_patches=self.num_patches,
            context_dim=encoder_embed_dim,
            predictor_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            mlp_ratio=mlp_ratio,
        )

    def forward(
        self,
        images: torch.Tensor,
        context_indices: list[int] | list[list[int]],
        target_indices_list: list[list[int]] | list[list[list[int]]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Forward pass for I-JEPA training.

        Args:
            images: Input images (B, C, H, W)
            context_indices: Context patch indices (flat list or per-sample list)
            target_indices_list: List of M target blocks, each a list of patch indices

        Returns:
            predictions: List of M predicted representations, each (B, num_patches, embed_dim)
            targets: List of M target representations (detached), each (B, num_patches, embed_dim)
        """
        device = images.device

        # Normalize indices format
        if isinstance(context_indices[0], list):
            ctx_indices = context_indices[0]
        else:
            ctx_indices = context_indices

        # Handle target indices - can be [M][patches] or [B][M][patches]
        if isinstance(target_indices_list[0][0], list):
            # Format: [B][M][patches] - take first sample's targets
            tgt_blocks = target_indices_list[0]
        else:
            # Format: [M][patches]
            tgt_blocks = target_indices_list

        # ============ Target Branch (no gradients) ============
        with torch.no_grad():
            # Target encoder processes full image
            target_output = self.target_encoder(images)  # (B, num_patches, embed_dim)

            # Extract target block representations
            targets = []
            for tgt_indices in tgt_blocks:
                tgt_tensor = torch.tensor(tgt_indices, device=device)
                target_patches = target_output[:, tgt_tensor, :]  # (B, T, embed_dim)
                targets.append(target_patches)

        # ============ Context Branch ============
        # Get patch embeddings from context encoder's patch_embed
        patch_embeddings = self.context_encoder.patch_embed(images)  # (B, num_patches, embed_dim)

        # Context encoder processes only context patches
        context_output = self.context_encoder(
            patch_embeddings, patch_indices=[ctx_indices]
        )  # (B, num_context, embed_dim)

        # ============ Predictor ============
        predictions = []
        for tgt_indices in tgt_blocks:
            pred = self.predictor(context_output, ctx_indices, tgt_indices)
            predictions.append(pred)

        return predictions, targets

    def get_target_encoder(self) -> ViTEncoder:
        """
        Get the target encoder for downstream evaluation.

        Returns:
            The EMA-updated target encoder
        """
        return self.target_encoder

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """
        Update target encoder with EMA of context encoder.

        Formula: θ̄ = momentum * θ̄ + (1 - momentum) * θ

        Args:
            momentum: EMA momentum (typically 0.996 -> 1.0 during training)
        """
        for param_t, param_s in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            param_t.data.mul_(momentum).add_(param_s.data, alpha=1 - momentum)
