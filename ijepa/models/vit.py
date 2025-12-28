"""
Vision Transformer (ViT) Components for I-JEPA

Contains:
- PatchEmbed: Convert image to patch embeddings
- TransformerBlock: Standard transformer block (LN -> Attention -> LN -> MLP)
- ViTEncoder: Full encoder used for context and target encoders
"""

import torch
import torch.nn as nn


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

        # TODO: Implement Conv2d projection
        # self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        raise NotImplementedError("PatchEmbed not yet implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, C, H, W)

        Returns:
            Patch embeddings (B, num_patches, embed_dim)
        """
        raise NotImplementedError("PatchEmbed.forward not yet implemented")


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

        # TODO: Implement transformer block components
        # self.norm1 = nn.LayerNorm(embed_dim)
        # self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # self.norm2 = nn.LayerNorm(embed_dim)
        # self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
        raise NotImplementedError("TransformerBlock not yet implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, embed_dim)

        Returns:
            Output tensor (B, N, embed_dim)
        """
        raise NotImplementedError("TransformerBlock.forward not yet implemented")


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

        # TODO: Implement encoder components
        # self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        # self.blocks = nn.ModuleList([TransformerBlock(...) for _ in range(depth)])
        # self.norm = nn.LayerNorm(embed_dim)
        raise NotImplementedError("ViTEncoder not yet implemented")

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
        raise NotImplementedError("ViTEncoder.forward not yet implemented")
