"""
I-JEPA Predictor

A narrow/lightweight ViT that predicts target representations from context.
Uses mask tokens with positional embeddings to indicate prediction locations.
"""

import torch
import torch.nn as nn

from ijepa.models.vit import TransformerBlock


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

        # Project from context encoder dimension to narrow predictor dimension
        self.input_proj = nn.Linear(context_dim, predictor_dim)

        # Shared mask token embedding (learned)
        # This single vector is combined with positional embeddings for each target
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Predictor's own positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Transformer blocks (narrower than encoder)
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, num_heads, mlp_ratio, dropout=0.0)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)

        # Project back to context encoder dimension for loss computation
        self.output_proj = nn.Linear(predictor_dim, context_dim)

    def forward(
        self,
        context_output: torch.Tensor,
        context_positions: list[int] | list[list[int]],
        target_positions: list[int] | list[list[int]],
    ) -> torch.Tensor:
        """
        Predict target representations from context.

        Args:
            context_output: Context encoder output (B, num_context_patches, context_dim)
            context_positions: Patch indices in context (list of ints, or list of lists for per-sample)
            target_positions: Patch indices to predict (list of ints, or list of lists for per-sample)

        Returns:
            predictions: Predicted target representations (B, num_target_patches, context_dim)
        """
        B = context_output.shape[0]
        device = context_output.device

        # Handle both flat list and nested list formats
        # For simplicity, assume same mask for all samples in batch (common in I-JEPA)
        if isinstance(context_positions[0], list):
            ctx_indices = context_positions[0]
        else:
            ctx_indices = context_positions

        if isinstance(target_positions[0], list):
            tgt_indices = target_positions[0]
        else:
            tgt_indices = target_positions

        # Convert to tensors
        ctx_indices_tensor = torch.tensor(ctx_indices, device=device)
        tgt_indices_tensor = torch.tensor(tgt_indices, device=device)

        # 1. Project context to predictor dimension
        context_tokens = self.input_proj(context_output)  # (B, C, predictor_dim)

        # 2. Add positional embeddings to context tokens
        context_pos = self.pos_embed[:, ctx_indices_tensor, :]  # (1, C, predictor_dim)
        context_tokens = context_tokens + context_pos

        # 3. Create mask tokens for target positions
        # mask_token: (1, 1, predictor_dim) + pos_embed subset: (1, T, predictor_dim)
        target_pos = self.pos_embed[:, tgt_indices_tensor, :]  # (1, T, predictor_dim)
        mask_tokens = self.mask_token + target_pos  # (1, T, predictor_dim)
        mask_tokens = mask_tokens.expand(B, -1, -1)  # (B, T, predictor_dim)

        # 4. Concatenate: [context_tokens, mask_tokens]
        x = torch.cat([context_tokens, mask_tokens], dim=1)  # (B, C+T, predictor_dim)

        # 5. Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # 6. Extract only the predictions (the mask token positions)
        num_context = len(ctx_indices)
        predictions = x[:, num_context:, :]  # (B, T, predictor_dim)

        # 7. Project back to context encoder dimension
        predictions = self.output_proj(predictions)  # (B, T, context_dim)

        return predictions
