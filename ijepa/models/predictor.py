"""
I-JEPA Predictor

A narrow/lightweight ViT that predicts target representations from context.
Uses mask tokens with positional embeddings to indicate prediction locations.
"""

import torch
import torch.nn as nn


class Predictor(nn.Module):
    """
    Predictor network for I-JEPA.

    Takes context encoder output and predicts target patch representations.
    Uses a bottleneck design (narrower than encoder) and mask tokens
    with positional embeddings.

    Architecture:
        1. Project context from encoder dim to predictor dim
        2. Add positional embeddings to context tokens
        3. Create mask tokens: shared_mask_embedding + pos_embed[target_positions]
        4. Concatenate context and mask tokens
        5. Process through transformer blocks
        6. Extract predictions at mask token positions
        7. Project back to encoder dimension

    Args:
        num_patches: Total number of patches in the image
        context_dim: Dimension of context encoder output
        predictor_dim: Internal predictor dimension (narrower bottleneck)
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
    """

    def __init__(
        self,
        num_patches: int = 64,
        context_dim: int = 192,
        predictor_dim: int = 96,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.context_dim = context_dim
        self.predictor_dim = predictor_dim

        # TODO: Implement predictor components
        # self.input_proj = nn.Linear(context_dim, predictor_dim)
        #
        # # Shared mask token embedding (learned)
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        # nn.init.normal_(self.mask_token, std=0.02)
        #
        # # Predictor's own positional embeddings
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_dim))
        # nn.init.normal_(self.pos_embed, std=0.02)
        #
        # # Transformer blocks
        # self.blocks = nn.ModuleList([...])
        # self.norm = nn.LayerNorm(predictor_dim)
        #
        # # Project back to context encoder dimension
        # self.output_proj = nn.Linear(predictor_dim, context_dim)
        raise NotImplementedError("Predictor not yet implemented")

    def forward(
        self,
        context_output: torch.Tensor,
        context_positions: list[list[int]],
        target_positions: list[list[int]],
    ) -> torch.Tensor:
        """
        Predict target representations from context.

        Args:
            context_output: Context encoder output (B, num_context_patches, context_dim)
            context_positions: List of patch indices in context for each sample
            target_positions: List of patch indices to predict for each sample

        Returns:
            predictions: Predicted target representations (B, num_target_patches, context_dim)
        """
        raise NotImplementedError("Predictor.forward not yet implemented")
