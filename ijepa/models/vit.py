"""
Vision Transformer (ViT) Components for I-JEPA

Contains:
- PatchEmbed: Convert image to patch embeddings
- TransformerBlock: Standard transformer block (LN -> Attention -> LN -> MLP)
- ViTEncoder: Full encoder used for context and target encoders
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple MLP with GELU activation.

    Args:
        in_features: Input dimension
        hidden_features: Hidden dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PatchEmbed(nn.Module):
    """
    Convert image to patch embeddings using a convolutional layer.

    Args:
        img_size: Input image size (assumes square)
        patch_size: Patch size (assumes square)
        in_channels: Number of input channels
        embed_dim: Embedding dimension

    Input: (B, C, H, W)
    Output: (B, num_patches, embed_dim)
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 192,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        # Conv2d projection: each patch becomes an embedding
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, C, H, W)

        Returns:
            Patch embeddings (B, num_patches, embed_dim)
        """
        # x: (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        x = self.proj(x)
        # Flatten spatial dimensions and transpose
        # (B, embed_dim, grid_size, grid_size) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        # (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """
    Standard Transformer block.

    Architecture:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        dropout: Dropout rate
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Pre-norm architecture
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, embed_dim)

        Returns:
            Output tensor (B, N, embed_dim)
        """
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder for I-JEPA.

    Used for both context encoder (trainable) and target encoder (EMA).
    NOTE: I-JEPA does NOT use a [CLS] token. All representations are patch-level.

    Args:
        img_size: Input image size
        patch_size: Patch size
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        dropout: Dropout rate
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding layer
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        patch_indices: list[list[int]] | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x: Either full images (B, C, H, W) or pre-extracted patch embeddings (B, N, embed_dim)
            patch_indices: Optional list of patch indices per sample for masked encoding.
                          If provided, only these patches are processed (for context encoder efficiency).

        Returns:
            Encoded patch representations (B, N, embed_dim)
        """
        # Check if input is images or already patch embeddings
        if x.dim() == 4:
            # Input is images (B, C, H, W) -> convert to patch embeddings
            x = self.patch_embed(x)
        # else: x is already (B, N, embed_dim) patch embeddings

        if patch_indices is None:
            # Process all patches (e.g., for target encoder)
            x = x + self.pos_embed
        else:
            # Process only specified patches (for context encoder efficiency)
            # patch_indices is a list of lists, one per sample in batch
            B = x.shape[0]
            device = x.device

            # For simplicity, assume all samples have the same indices
            # (This is common in I-JEPA where the same mask is used for the batch)
            if isinstance(patch_indices[0], list):
                indices = patch_indices[0]
            else:
                indices = patch_indices

            # Select only the specified patches
            indices_tensor = torch.tensor(indices, device=device)
            x = x[:, indices_tensor, :]  # (B, num_selected, embed_dim)

            # Add corresponding positional embeddings
            pos_embed_selected = self.pos_embed[:, indices_tensor, :]
            x = x + pos_embed_selected

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        return x
