"""
Unit tests for I-JEPA Predictor.

Tests verify:
- Correct output shapes for various input configurations
- Mask token mechanism works correctly
- Bottleneck architecture (dimension reduction)
- Gradient flow through the predictor
- Integration with encoder outputs
"""

import pytest
import torch
import torch.nn as nn

from ijepa.models.predictor import Predictor
from ijepa.models.vit import ViTEncoder


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
def context_dim():
    return 192


@pytest.fixture
def predictor_dim():
    return 96


@pytest.fixture
def num_patches():
    """8x8 grid = 64 patches for 32x32 image with 4x4 patches."""
    return 64


@pytest.fixture
def context_positions():
    """Sample context patch indices (50 out of 64 patches)."""
    return list(range(50))


@pytest.fixture
def target_positions():
    """Sample target patch indices (8 patches to predict)."""
    return [50, 51, 52, 53, 58, 59, 60, 61]


# =============================================================================
# Basic Shape Tests
# =============================================================================


class TestPredictorShapes:
    """Tests for Predictor output shapes."""

    def test_basic_output_shape(
        self, device, batch_size, context_dim, context_positions, target_positions
    ):
        """Basic forward pass with expected output shape."""
        predictor = Predictor(
            num_patches=64, context_dim=context_dim, predictor_dim=96, depth=4
        ).to(device)

        # Simulate context encoder output
        context_output = torch.randn(
            batch_size, len(context_positions), context_dim, device=device
        )

        predictions = predictor(context_output, context_positions, target_positions)

        expected_shape = (batch_size, len(target_positions), context_dim)
        assert predictions.shape == expected_shape, f"Got {predictions.shape}"

    def test_output_shape_with_nested_lists(
        self, device, batch_size, context_dim, context_positions, target_positions
    ):
        """Forward pass with nested list format for positions."""
        predictor = Predictor(num_patches=64, context_dim=context_dim).to(device)

        context_output = torch.randn(
            batch_size, len(context_positions), context_dim, device=device
        )

        # Use nested list format
        predictions = predictor(
            context_output, [context_positions], [target_positions]
        )

        expected_shape = (batch_size, len(target_positions), context_dim)
        assert predictions.shape == expected_shape

    def test_various_context_sizes(self, device, context_dim):
        """Test with different numbers of context patches."""
        predictor = Predictor(num_patches=64, context_dim=context_dim).to(device)
        target_pos = [60, 61, 62, 63]

        for num_context in [8, 16, 32, 48, 56]:
            context_pos = list(range(num_context))
            context_output = torch.randn(2, num_context, context_dim, device=device)

            predictions = predictor(context_output, context_pos, target_pos)

            assert predictions.shape == (2, 4, context_dim)

    def test_various_target_sizes(self, device, context_dim):
        """Test with different numbers of target patches."""
        predictor = Predictor(num_patches=64, context_dim=context_dim).to(device)
        context_pos = list(range(50))
        context_output = torch.randn(2, 50, context_dim, device=device)

        for num_targets in [1, 4, 8, 12, 14]:
            target_pos = list(range(50, 50 + num_targets))

            predictions = predictor(context_output, context_pos, target_pos)

            assert predictions.shape == (2, num_targets, context_dim)

    def test_single_target_patch(self, device, context_dim):
        """Test predicting a single target patch."""
        predictor = Predictor(num_patches=64, context_dim=context_dim).to(device)
        context_pos = list(range(63))
        context_output = torch.randn(2, 63, context_dim, device=device)
        target_pos = [63]

        predictions = predictor(context_output, context_pos, target_pos)

        assert predictions.shape == (2, 1, context_dim)


# =============================================================================
# Architecture Tests
# =============================================================================


class TestPredictorArchitecture:
    """Tests for Predictor architecture properties."""

    def test_bottleneck_dimension(self):
        """Verify predictor uses narrower dimension than context."""
        predictor = Predictor(
            num_patches=64, context_dim=192, predictor_dim=96, depth=4
        )

        # Check projection dimensions
        assert predictor.input_proj.in_features == 192
        assert predictor.input_proj.out_features == 96
        assert predictor.output_proj.in_features == 96
        assert predictor.output_proj.out_features == 192

    def test_mask_token_shape(self):
        """Mask token should have correct shape."""
        predictor = Predictor(num_patches=64, context_dim=192, predictor_dim=96)

        assert predictor.mask_token.shape == (1, 1, 96)

    def test_positional_embedding_shape(self):
        """Positional embeddings should match num_patches."""
        predictor = Predictor(num_patches=64, context_dim=192, predictor_dim=96)

        assert predictor.pos_embed.shape == (1, 64, 96)

    def test_positional_embedding_initialization(self):
        """Positional embeddings should be initialized with small values."""
        predictor = Predictor(num_patches=64, context_dim=192, predictor_dim=96)

        assert predictor.pos_embed.std() < 0.1
        assert predictor.mask_token.std() < 0.1

    def test_number_of_blocks(self):
        """Verify correct number of transformer blocks."""
        for depth in [2, 4, 6, 8]:
            predictor = Predictor(num_patches=64, depth=depth)
            assert len(predictor.blocks) == depth

    def test_different_predictor_dims(self, device):
        """Test various predictor dimensions (must be divisible by num_heads=6)."""
        for pred_dim in [48, 96, 144, 192]:
            predictor = Predictor(
                num_patches=64, context_dim=192, predictor_dim=pred_dim
            ).to(device)

            context_output = torch.randn(2, 50, 192, device=device)
            predictions = predictor(
                context_output, list(range(50)), [50, 51, 52, 53]
            )

            # Output should always be context_dim, not predictor_dim
            assert predictions.shape == (2, 4, 192)

    def test_parameter_count(self):
        """Verify reasonable parameter count for predictor."""
        predictor = Predictor(
            num_patches=64, context_dim=192, predictor_dim=96, depth=4, num_heads=6
        )

        total_params = sum(p.numel() for p in predictor.parameters())

        # Predictor should be smaller than encoder due to bottleneck
        # Roughly 0.5-2M parameters
        assert 100_000 < total_params < 5_000_000, f"Params: {total_params}"


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestPredictorStability:
    """Tests for numerical stability."""

    def test_no_nan_output(self, device, context_dim, context_positions, target_positions):
        """Predictor output should not contain NaN values."""
        predictor = Predictor(num_patches=64, context_dim=context_dim).to(device)
        context_output = torch.randn(
            2, len(context_positions), context_dim, device=device
        )

        predictions = predictor(context_output, context_positions, target_positions)

        assert not torch.isnan(predictions).any(), "Output contains NaN"
        assert not torch.isinf(predictions).any(), "Output contains Inf"

    def test_gradient_flow(self, device, context_dim, context_positions, target_positions):
        """Gradients should flow through the predictor."""
        predictor = Predictor(num_patches=64, context_dim=context_dim).to(device)
        context_output = torch.randn(
            2, len(context_positions), context_dim, device=device, requires_grad=True
        )

        predictions = predictor(context_output, context_positions, target_positions)
        loss = predictions.pow(2).mean()
        loss.backward()

        assert context_output.grad is not None, "No gradient for input"
        assert not torch.isnan(context_output.grad).any(), "Gradient contains NaN"

    def test_all_parameters_have_gradients(
        self, device, context_dim, context_positions, target_positions
    ):
        """All predictor parameters should receive gradients."""
        predictor = Predictor(num_patches=64, context_dim=context_dim).to(device)
        context_output = torch.randn(
            2, len(context_positions), context_dim, device=device
        )

        predictions = predictor(context_output, context_positions, target_positions)
        loss = predictions.pow(2).mean()
        loss.backward()

        for name, param in predictor.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_eval_mode_deterministic(
        self, device, context_dim, context_positions, target_positions
    ):
        """In eval mode, output should be deterministic."""
        predictor = Predictor(num_patches=64, context_dim=context_dim).to(device)
        predictor.eval()

        context_output = torch.randn(
            2, len(context_positions), context_dim, device=device
        )

        out1 = predictor(context_output, context_positions, target_positions)
        out2 = predictor(context_output, context_positions, target_positions)

        assert torch.allclose(out1, out2), "Eval mode should be deterministic"


# =============================================================================
# Mask Token Mechanism Tests
# =============================================================================


class TestMaskTokenMechanism:
    """Tests for the mask token mechanism."""

    def test_different_target_positions_different_outputs(self, device, context_dim):
        """Different target positions should produce different predictions."""
        predictor = Predictor(num_patches=64, context_dim=context_dim).to(device)
        predictor.eval()

        context_output = torch.randn(2, 50, context_dim, device=device)
        context_pos = list(range(50))

        # Predict different target positions
        pred1 = predictor(context_output, context_pos, [50, 51, 52, 53])
        pred2 = predictor(context_output, context_pos, [54, 55, 56, 57])

        # Predictions should differ due to different positional embeddings
        assert not torch.allclose(pred1, pred2), "Different targets should differ"

    def test_same_positions_same_outputs(self, device, context_dim):
        """Same positions should produce identical predictions (in eval mode)."""
        predictor = Predictor(num_patches=64, context_dim=context_dim).to(device)
        predictor.eval()

        context_output = torch.randn(2, 50, context_dim, device=device)
        context_pos = list(range(50))
        target_pos = [50, 51, 52, 53]

        pred1 = predictor(context_output, context_pos, target_pos)
        pred2 = predictor(context_output, context_pos, target_pos)

        assert torch.allclose(pred1, pred2)

    def test_mask_token_contributes_to_output(self, device, context_dim):
        """Mask token should affect output (verify it's being used)."""
        predictor = Predictor(num_patches=64, context_dim=context_dim).to(device)
        predictor.eval()

        context_output = torch.randn(2, 50, context_dim, device=device)
        context_pos = list(range(50))
        target_pos = [50, 51, 52, 53]

        # Get baseline prediction
        pred1 = predictor(context_output, context_pos, target_pos)

        # Modify mask token and predict again
        with torch.no_grad():
            predictor.mask_token.add_(1.0)

        pred2 = predictor(context_output, context_pos, target_pos)

        # Predictions should differ since mask token changed
        assert not torch.allclose(pred1, pred2), "Mask token should affect output"


# =============================================================================
# Integration Tests
# =============================================================================


class TestPredictorIntegration:
    """Integration tests with encoder."""

    def test_with_encoder_output(self, device):
        """Test predictor with actual encoder output."""
        encoder = ViTEncoder(
            img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6
        ).to(device)
        predictor = Predictor(
            num_patches=64, context_dim=192, predictor_dim=96, depth=4
        ).to(device)

        # Simulate I-JEPA forward pass
        img = torch.randn(2, 3, 32, 32, device=device)

        # Context encoder processes only context patches
        context_pos = list(range(50))
        patches = encoder.patch_embed(img)
        context_output = encoder(patches, patch_indices=[context_pos])

        # Predictor predicts target patches
        target_pos = [50, 51, 52, 53, 58, 59, 60, 61]
        predictions = predictor(context_output, context_pos, target_pos)

        assert predictions.shape == (2, 8, 192)

    def test_end_to_end_training_step(self, device):
        """Simulate a full I-JEPA training step."""
        context_encoder = ViTEncoder(
            img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6
        ).to(device)
        target_encoder = ViTEncoder(
            img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=6
        ).to(device)
        target_encoder.requires_grad_(False)

        predictor = Predictor(
            num_patches=64, context_dim=192, predictor_dim=96, depth=4
        ).to(device)

        optimizer = torch.optim.Adam(
            list(context_encoder.parameters()) + list(predictor.parameters()),
            lr=1e-4,
        )

        img = torch.randn(4, 3, 32, 32, device=device)
        context_pos = list(range(50))
        target_pos = [50, 51, 52, 53, 58, 59, 60, 61]

        # Target encoder (no gradients)
        with torch.no_grad():
            target_output = target_encoder(img)
            # Extract target patch representations
            target_patches = target_output[:, target_pos, :]

        # Context encoder
        patches = context_encoder.patch_embed(img)
        context_output = context_encoder(patches, patch_indices=[context_pos])

        # Predictor
        predictions = predictor(context_output, context_pos, target_pos)

        # L2 loss
        loss = (predictions - target_patches).pow(2).mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert predictions.shape == target_patches.shape

    def test_multiple_target_blocks(self, device):
        """Test predicting multiple target blocks (as in I-JEPA)."""
        predictor = Predictor(num_patches=64, context_dim=192).to(device)

        context_output = torch.randn(2, 50, 192, device=device)
        context_pos = list(range(50))

        # Multiple target blocks
        target_blocks = [
            [50, 51, 52, 53],
            [54, 55, 56, 57],
            [58, 59, 60, 61],
            [62, 63],
        ]

        # Predict each target block
        predictions = []
        for target_pos in target_blocks:
            pred = predictor(context_output, context_pos, target_pos)
            predictions.append(pred)
            assert pred.shape == (2, len(target_pos), 192)

        # All predictions should be different
        assert not torch.allclose(predictions[0], predictions[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
