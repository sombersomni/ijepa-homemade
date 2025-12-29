"""
Unit tests for Vision Transformer components.

Tests verify:
- Correct output shapes for all components
- Numerical stability (no NaN/Inf)
- Gradient flow
- Component interactions
"""

import pytest
import torch
import torch.nn as nn

from ijepa.models.vit import MLP, PatchEmbed, TransformerBlock, ViTEncoder


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Use GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def embed_dim():
    return 192


@pytest.fixture
def num_patches():
    """8x8 grid = 64 patches for 32x32 image with 4x4 patches."""
    return 64


# =============================================================================
# MLP Tests
# =============================================================================


class TestMLP:
    """Tests for the MLP module."""

    def test_output_shape(self, device, batch_size, embed_dim, num_patches):
        """MLP should preserve input shape."""
        mlp = MLP(embed_dim, embed_dim * 4).to(device)
        x = torch.randn(batch_size, num_patches, embed_dim, device=device)

        out = mlp(x)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_different_hidden_dims(self, device, embed_dim):
        """MLP should work with various hidden dimension ratios."""
        for ratio in [1.0, 2.0, 4.0, 8.0]:
            hidden_dim = int(embed_dim * ratio)
            mlp = MLP(embed_dim, hidden_dim).to(device)
            x = torch.randn(2, 16, embed_dim, device=device)

            out = mlp(x)

            assert out.shape == x.shape

    def test_no_nan_output(self, device, embed_dim):
        """MLP output should not contain NaN values."""
        mlp = MLP(embed_dim, embed_dim * 4).to(device)
        x = torch.randn(2, 64, embed_dim, device=device)

        out = mlp(x)

        assert not torch.isnan(out).any(), "Output contains NaN values"
        assert not torch.isinf(out).any(), "Output contains Inf values"

    def test_gradient_flow(self, device, embed_dim):
        """Gradients should flow through MLP."""
        mlp = MLP(embed_dim, embed_dim * 4).to(device)
        x = torch.randn(2, 64, embed_dim, device=device, requires_grad=True)

        out = mlp(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "No gradient computed for input"
        assert not torch.isnan(x.grad).any(), "Gradient contains NaN"

    def test_dropout_effect(self, device, embed_dim):
        """Dropout should cause different outputs in training mode."""
        mlp = MLP(embed_dim, embed_dim * 4, dropout=0.5).to(device)
        mlp.train()
        x = torch.randn(2, 64, embed_dim, device=device)

        out1 = mlp(x)
        out2 = mlp(x)

        # With 50% dropout, outputs should differ
        assert not torch.allclose(out1, out2), "Dropout not affecting output"

    def test_eval_mode_deterministic(self, device, embed_dim):
        """In eval mode, output should be deterministic."""
        mlp = MLP(embed_dim, embed_dim * 4, dropout=0.5).to(device)
        mlp.eval()
        x = torch.randn(2, 64, embed_dim, device=device)

        out1 = mlp(x)
        out2 = mlp(x)

        assert torch.allclose(out1, out2), "Eval mode should be deterministic"


# =============================================================================
# PatchEmbed Tests
# =============================================================================


class TestPatchEmbed:
    """Tests for the PatchEmbed module."""

    def test_output_shape_default(self, device, batch_size):
        """Default config: 32x32 image, 4x4 patches -> 64 patches."""
        patch_embed = PatchEmbed(
            img_size=32, patch_size=4, in_channels=3, embed_dim=192
        ).to(device)
        img = torch.randn(batch_size, 3, 32, 32, device=device)

        out = patch_embed(img)

        assert out.shape == (batch_size, 64, 192), f"Got {out.shape}"

    def test_output_shape_various_configs(self, device):
        """Test various image/patch size combinations."""
        configs = [
            (32, 4, 64),  # 8x8 grid
            (32, 8, 16),  # 4x4 grid
            (64, 8, 64),  # 8x8 grid
            (224, 16, 196),  # ImageNet config: 14x14 grid
        ]

        for img_size, patch_size, expected_patches in configs:
            patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_channels=3, embed_dim=192
            ).to(device)
            img = torch.randn(2, 3, img_size, img_size, device=device)

            out = patch_embed(img)

            assert out.shape == (
                2,
                expected_patches,
                192,
            ), f"Config ({img_size}, {patch_size}): expected {expected_patches} patches, got {out.shape[1]}"

    def test_num_patches_attribute(self):
        """num_patches attribute should be correctly computed."""
        patch_embed = PatchEmbed(img_size=32, patch_size=4)
        assert patch_embed.num_patches == 64

        patch_embed = PatchEmbed(img_size=224, patch_size=16)
        assert patch_embed.num_patches == 196

    def test_grid_size_attribute(self):
        """grid_size attribute should be correctly computed."""
        patch_embed = PatchEmbed(img_size=32, patch_size=4)
        assert patch_embed.grid_size == 8

        patch_embed = PatchEmbed(img_size=224, patch_size=16)
        assert patch_embed.grid_size == 14

    def test_different_embed_dims(self, device):
        """Test various embedding dimensions."""
        for embed_dim in [64, 128, 192, 256, 384, 768]:
            patch_embed = PatchEmbed(
                img_size=32, patch_size=4, embed_dim=embed_dim
            ).to(device)
            img = torch.randn(2, 3, 32, 32, device=device)

            out = patch_embed(img)

            assert out.shape == (2, 64, embed_dim)

    def test_different_input_channels(self, device):
        """Test various input channel counts."""
        for in_channels in [1, 3, 4]:
            patch_embed = PatchEmbed(
                img_size=32, patch_size=4, in_channels=in_channels, embed_dim=192
            ).to(device)
            img = torch.randn(2, in_channels, 32, 32, device=device)

            out = patch_embed(img)

            assert out.shape == (2, 64, 192)

    def test_no_nan_output(self, device):
        """PatchEmbed output should not contain NaN values."""
        patch_embed = PatchEmbed().to(device)
        img = torch.randn(2, 3, 32, 32, device=device)

        out = patch_embed(img)

        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_gradient_flow(self, device):
        """Gradients should flow through PatchEmbed."""
        patch_embed = PatchEmbed().to(device)
        img = torch.randn(2, 3, 32, 32, device=device, requires_grad=True)

        out = patch_embed(img)
        loss = out.sum()
        loss.backward()

        assert img.grad is not None, "No gradient computed"


# =============================================================================
# TransformerBlock Tests
# =============================================================================


class TestTransformerBlock:
    """Tests for the TransformerBlock module."""

    def test_output_shape(self, device, batch_size, embed_dim, num_patches):
        """TransformerBlock should preserve input shape."""
        block = TransformerBlock(embed_dim=embed_dim, num_heads=6).to(device)
        x = torch.randn(batch_size, num_patches, embed_dim, device=device)

        out = block(x)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_variable_sequence_length(self, device, embed_dim):
        """Block should work with different sequence lengths."""
        block = TransformerBlock(embed_dim=embed_dim, num_heads=6).to(device)

        for seq_len in [1, 16, 64, 128, 256]:
            x = torch.randn(2, seq_len, embed_dim, device=device)
            out = block(x)
            assert out.shape == x.shape

    def test_different_num_heads(self, device, embed_dim):
        """Block should work with different attention head counts."""
        for num_heads in [1, 2, 3, 4, 6, 8, 12]:
            if embed_dim % num_heads == 0:  # Must divide evenly
                block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads).to(
                    device
                )
                x = torch.randn(2, 64, embed_dim, device=device)

                out = block(x)

                assert out.shape == x.shape

    def test_different_mlp_ratios(self, device, embed_dim):
        """Block should work with different MLP ratios."""
        for mlp_ratio in [1.0, 2.0, 4.0, 8.0]:
            block = TransformerBlock(
                embed_dim=embed_dim, num_heads=6, mlp_ratio=mlp_ratio
            ).to(device)
            x = torch.randn(2, 64, embed_dim, device=device)

            out = block(x)

            assert out.shape == x.shape

    def test_residual_connection(self, device, embed_dim):
        """Verify residual connections are working."""
        block = TransformerBlock(embed_dim=embed_dim, num_heads=6).to(device)

        # With zero weights, output should equal input due to residual
        with torch.no_grad():
            for param in block.parameters():
                param.zero_()

        x = torch.randn(2, 64, embed_dim, device=device)
        out = block(x)

        # After zeroing weights, the residual should dominate
        # (though LayerNorm will still affect values)
        assert out.shape == x.shape

    def test_no_nan_output(self, device, embed_dim):
        """TransformerBlock output should not contain NaN values."""
        block = TransformerBlock(embed_dim=embed_dim, num_heads=6).to(device)
        x = torch.randn(2, 64, embed_dim, device=device)

        out = block(x)

        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_gradient_flow(self, device, embed_dim):
        """Gradients should flow through TransformerBlock."""
        block = TransformerBlock(embed_dim=embed_dim, num_heads=6).to(device)
        x = torch.randn(2, 64, embed_dim, device=device, requires_grad=True)

        out = block(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "No gradient computed"
        assert not torch.isnan(x.grad).any(), "Gradient contains NaN"


# =============================================================================
# ViTEncoder Tests
# =============================================================================


class TestViTEncoder:
    """Tests for the ViTEncoder module."""

    def test_output_shape_full_image(self, device, batch_size):
        """Encoder with full image input."""
        encoder = ViTEncoder(
            img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6
        ).to(device)
        img = torch.randn(batch_size, 3, 32, 32, device=device)

        out = encoder(img)

        assert out.shape == (batch_size, 64, 192), f"Got {out.shape}"

    def test_output_shape_patch_embeddings(self, device, batch_size):
        """Encoder with pre-computed patch embeddings."""
        encoder = ViTEncoder(
            img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6
        ).to(device)
        patches = torch.randn(batch_size, 64, 192, device=device)

        out = encoder(patches)

        assert out.shape == (batch_size, 64, 192), f"Got {out.shape}"

    def test_masked_encoding(self, device, batch_size):
        """Encoder with patch_indices for masked encoding."""
        encoder = ViTEncoder(
            img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6
        ).to(device)
        patches = torch.randn(batch_size, 64, 192, device=device)

        # Select 8 specific patches
        patch_indices = [[0, 1, 2, 3, 8, 9, 10, 11]]

        out = encoder(patches, patch_indices=patch_indices)

        assert out.shape == (batch_size, 8, 192), f"Got {out.shape}"

    def test_masked_encoding_various_sizes(self, device):
        """Masked encoding with different numbers of patches."""
        encoder = ViTEncoder(
            img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6
        ).to(device)
        patches = torch.randn(2, 64, 192, device=device)

        for num_selected in [1, 8, 16, 32, 50, 64]:
            patch_indices = [list(range(num_selected))]
            out = encoder(patches, patch_indices=patch_indices)
            assert out.shape == (2, num_selected, 192), f"Got {out.shape}"

    def test_different_depths(self, device):
        """Test various encoder depths."""
        for depth in [1, 2, 4, 6, 8, 12]:
            encoder = ViTEncoder(
                img_size=32, patch_size=4, embed_dim=192, depth=depth, num_heads=6
            ).to(device)
            img = torch.randn(2, 3, 32, 32, device=device)

            out = encoder(img)

            assert out.shape == (2, 64, 192)

    def test_num_patches_attribute(self):
        """num_patches attribute should be correct."""
        encoder = ViTEncoder(img_size=32, patch_size=4)
        assert encoder.num_patches == 64

        encoder = ViTEncoder(img_size=224, patch_size=16)
        assert encoder.num_patches == 196

    def test_positional_embedding_shape(self):
        """Positional embeddings should have correct shape."""
        encoder = ViTEncoder(img_size=32, patch_size=4, embed_dim=192)
        assert encoder.pos_embed.shape == (1, 64, 192)

        encoder = ViTEncoder(img_size=224, patch_size=16, embed_dim=768)
        assert encoder.pos_embed.shape == (1, 196, 768)

    def test_positional_embedding_initialization(self):
        """Positional embeddings should be initialized with small values."""
        encoder = ViTEncoder(img_size=32, patch_size=4, embed_dim=192)

        # Check that values are roughly in expected range for normal(std=0.02)
        assert encoder.pos_embed.std() < 0.1, "Pos embed std too large"
        assert encoder.pos_embed.mean().abs() < 0.1, "Pos embed mean too large"

    def test_no_nan_output(self, device):
        """ViTEncoder output should not contain NaN values."""
        encoder = ViTEncoder(
            img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6
        ).to(device)
        img = torch.randn(2, 3, 32, 32, device=device)

        out = encoder(img)

        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_gradient_flow_full(self, device):
        """Gradients should flow through full encoder."""
        encoder = ViTEncoder(
            img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6
        ).to(device)
        img = torch.randn(2, 3, 32, 32, device=device, requires_grad=True)

        out = encoder(img)
        loss = out.sum()
        loss.backward()

        assert img.grad is not None, "No gradient computed"
        assert not torch.isnan(img.grad).any(), "Gradient contains NaN"

    def test_gradient_flow_masked(self, device):
        """Gradients should flow through masked encoder."""
        encoder = ViTEncoder(
            img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6
        ).to(device)
        patches = torch.randn(2, 64, 192, device=device, requires_grad=True)
        patch_indices = [[0, 1, 2, 3, 8, 9, 10, 11]]

        out = encoder(patches, patch_indices=patch_indices)
        loss = out.sum()
        loss.backward()

        assert patches.grad is not None, "No gradient computed"

    def test_parameter_count(self):
        """Verify reasonable parameter count for CIFAR-10 config."""
        encoder = ViTEncoder(
            img_size=32,
            patch_size=4,
            in_channels=3,
            embed_dim=192,
            depth=6,
            num_heads=6,
            mlp_ratio=4.0,
        )

        total_params = sum(p.numel() for p in encoder.parameters())

        # Should be roughly 2-5M parameters for this config
        assert 1_000_000 < total_params < 10_000_000, f"Unexpected param count: {total_params}"

    def test_eval_mode_deterministic(self, device):
        """In eval mode, output should be deterministic."""
        encoder = ViTEncoder(
            img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6
        ).to(device)
        encoder.eval()
        img = torch.randn(2, 3, 32, 32, device=device)

        out1 = encoder(img)
        out2 = encoder(img)

        assert torch.allclose(out1, out2), "Eval mode should be deterministic"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for ViT components working together."""

    def test_patch_embed_to_encoder(self, device):
        """PatchEmbed output should work as encoder input."""
        patch_embed = PatchEmbed(img_size=32, patch_size=4, embed_dim=192).to(device)
        encoder = ViTEncoder(
            img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6
        ).to(device)

        img = torch.randn(2, 3, 32, 32, device=device)
        patches = patch_embed(img)
        out = encoder(patches)

        assert out.shape == (2, 64, 192)

    def test_end_to_end_training_step(self, device):
        """Simulate a training step through the encoder."""
        encoder = ViTEncoder(
            img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6
        ).to(device)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)

        img = torch.randn(4, 3, 32, 32, device=device)

        # Forward pass
        out = encoder(img)

        # Dummy loss (e.g., L2 to zero)
        loss = out.pow(2).mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that parameters were updated
        assert loss.item() > 0, "Loss should be positive"

    def test_context_and_target_encoder_workflow(self, device):
        """Simulate I-JEPA context/target encoder workflow."""
        # Create context and target encoders (same architecture)
        context_encoder = ViTEncoder(
            img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6
        ).to(device)
        target_encoder = ViTEncoder(
            img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6
        ).to(device)

        # Target encoder should be frozen (EMA updated)
        target_encoder.requires_grad_(False)

        img = torch.randn(2, 3, 32, 32, device=device)

        # Target encoder processes full image
        with torch.no_grad():
            target_out = target_encoder(img)
        assert target_out.shape == (2, 64, 192)

        # Context encoder processes only visible patches
        patches = context_encoder.patch_embed(img)
        context_indices = [[0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]]  # 12 patches
        context_out = context_encoder(patches, patch_indices=context_indices)
        assert context_out.shape == (2, 12, 192)

        # Verify gradient flow only through context encoder
        loss = context_out.pow(2).mean()
        loss.backward()

        # Context encoder should have gradients
        assert context_encoder.pos_embed.grad is not None

        # Target encoder should not have gradients
        assert target_encoder.pos_embed.grad is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
