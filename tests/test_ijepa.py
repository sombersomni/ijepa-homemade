"""
Unit tests for I-JEPA full model.

Tests verify:
- Correct integration of context encoder, target encoder, and predictor
- EMA update mechanism
- Forward pass with various masking configurations
- Gradient flow (only through context encoder and predictor)
- Output shapes and types
"""

import pytest
import torch
import torch.nn as nn

from ijepa.models.ijepa import IJEPA
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
def ijepa_model(device):
    """Create a default IJEPA model."""
    model = IJEPA(
        img_size=32,
        patch_size=4,
        encoder_embed_dim=192,
        encoder_depth=6,
        encoder_num_heads=6,
        predictor_embed_dim=96,
        predictor_depth=4,
    ).to(device)
    return model


@pytest.fixture
def context_indices():
    """50 context patches (out of 64)."""
    return list(range(50))


@pytest.fixture
def target_blocks():
    """4 target blocks with different sizes."""
    return [
        [50, 51, 52, 53],  # 4 patches
        [54, 55, 56, 57],  # 4 patches
        [58, 59, 60, 61],  # 4 patches
        [62, 63],  # 2 patches
    ]


# =============================================================================
# Initialization Tests
# =============================================================================


class TestIJEPAInit:
    """Tests for IJEPA initialization."""

    def test_components_created(self, ijepa_model):
        """All three components should be created."""
        assert hasattr(ijepa_model, "context_encoder")
        assert hasattr(ijepa_model, "target_encoder")
        assert hasattr(ijepa_model, "predictor")

    def test_encoders_same_architecture(self, ijepa_model):
        """Context and target encoders should have same architecture."""
        ctx_params = sum(p.numel() for p in ijepa_model.context_encoder.parameters())
        tgt_params = sum(p.numel() for p in ijepa_model.target_encoder.parameters())
        assert ctx_params == tgt_params

    def test_target_encoder_no_gradients(self, ijepa_model):
        """Target encoder should not require gradients."""
        for param in ijepa_model.target_encoder.parameters():
            assert not param.requires_grad, "Target encoder should not require grad"

    def test_context_encoder_has_gradients(self, ijepa_model):
        """Context encoder should require gradients."""
        for param in ijepa_model.context_encoder.parameters():
            assert param.requires_grad, "Context encoder should require grad"

    def test_predictor_has_gradients(self, ijepa_model):
        """Predictor should require gradients."""
        for param in ijepa_model.predictor.parameters():
            assert param.requires_grad, "Predictor should require grad"

    def test_encoders_start_identical(self):
        """Context and target encoders should start with identical weights."""
        model = IJEPA()

        for (name_c, param_c), (name_t, param_t) in zip(
            model.context_encoder.named_parameters(),
            model.target_encoder.named_parameters(),
        ):
            assert name_c == name_t, "Parameter names should match"
            assert torch.allclose(param_c, param_t), f"Params {name_c} should be equal"

    def test_num_patches_attribute(self):
        """num_patches should be correctly computed."""
        model = IJEPA(img_size=32, patch_size=4)
        assert model.num_patches == 64

        model = IJEPA(img_size=64, patch_size=8)
        assert model.num_patches == 64


# =============================================================================
# Forward Pass Tests
# =============================================================================


class TestIJEPAForward:
    """Tests for IJEPA forward pass."""

    def test_basic_forward(
        self, device, ijepa_model, batch_size, context_indices, target_blocks
    ):
        """Basic forward pass should return predictions and targets."""
        images = torch.randn(batch_size, 3, 32, 32, device=device)

        predictions, targets = ijepa_model(images, context_indices, target_blocks)

        assert len(predictions) == len(target_blocks)
        assert len(targets) == len(target_blocks)

    def test_output_shapes(
        self, device, ijepa_model, batch_size, context_indices, target_blocks
    ):
        """Output shapes should match target block sizes."""
        images = torch.randn(batch_size, 3, 32, 32, device=device)

        predictions, targets = ijepa_model(images, context_indices, target_blocks)

        for i, (pred, tgt) in enumerate(zip(predictions, targets)):
            expected_patches = len(target_blocks[i])
            assert pred.shape == (batch_size, expected_patches, 192), f"Block {i} pred"
            assert tgt.shape == (batch_size, expected_patches, 192), f"Block {i} tgt"

    def test_predictions_targets_same_shape(
        self, device, ijepa_model, context_indices, target_blocks
    ):
        """Predictions and targets should have matching shapes."""
        images = torch.randn(2, 3, 32, 32, device=device)

        predictions, targets = ijepa_model(images, context_indices, target_blocks)

        for pred, tgt in zip(predictions, targets):
            assert pred.shape == tgt.shape

    def test_nested_indices_format(self, device, ijepa_model):
        """Forward should work with nested list format."""
        images = torch.randn(2, 3, 32, 32, device=device)
        context_indices = [list(range(50))]  # Nested format
        target_blocks = [[50, 51, 52, 53], [54, 55, 56, 57]]

        predictions, targets = ijepa_model(images, context_indices, target_blocks)

        assert len(predictions) == 2
        assert predictions[0].shape == (2, 4, 192)

    def test_single_target_block(self, device, ijepa_model, context_indices):
        """Forward should work with a single target block."""
        images = torch.randn(2, 3, 32, 32, device=device)
        target_blocks = [[60, 61, 62, 63]]

        predictions, targets = ijepa_model(images, context_indices, target_blocks)

        assert len(predictions) == 1
        assert predictions[0].shape == (2, 4, 192)

    def test_various_context_sizes(self, device, ijepa_model):
        """Forward should work with different context sizes."""
        images = torch.randn(2, 3, 32, 32, device=device)
        target_blocks = [[60, 61, 62, 63]]

        for ctx_size in [8, 16, 32, 48, 56]:
            context_indices = list(range(ctx_size))
            predictions, targets = ijepa_model(images, context_indices, target_blocks)
            assert predictions[0].shape == (2, 4, 192)


# =============================================================================
# Gradient Flow Tests
# =============================================================================


class TestIJEPAGradients:
    """Tests for gradient flow through IJEPA."""

    def test_gradients_flow_to_context_encoder(
        self, device, ijepa_model, context_indices, target_blocks
    ):
        """Gradients should flow to context encoder."""
        images = torch.randn(2, 3, 32, 32, device=device)

        predictions, targets = ijepa_model(images, context_indices, target_blocks)

        # Compute L2 loss
        loss = sum((p - t).pow(2).mean() for p, t in zip(predictions, targets))
        loss.backward()

        # Check context encoder has gradients
        for name, param in ijepa_model.context_encoder.named_parameters():
            assert param.grad is not None, f"No grad for context_encoder.{name}"

    def test_gradients_flow_to_predictor(
        self, device, ijepa_model, context_indices, target_blocks
    ):
        """Gradients should flow to predictor."""
        images = torch.randn(2, 3, 32, 32, device=device)

        predictions, targets = ijepa_model(images, context_indices, target_blocks)
        loss = sum((p - t).pow(2).mean() for p, t in zip(predictions, targets))
        loss.backward()

        for name, param in ijepa_model.predictor.named_parameters():
            assert param.grad is not None, f"No grad for predictor.{name}"

    def test_no_gradients_to_target_encoder(
        self, device, ijepa_model, context_indices, target_blocks
    ):
        """Target encoder should not receive gradients."""
        images = torch.randn(2, 3, 32, 32, device=device)

        predictions, targets = ijepa_model(images, context_indices, target_blocks)
        loss = sum((p - t).pow(2).mean() for p, t in zip(predictions, targets))
        loss.backward()

        for name, param in ijepa_model.target_encoder.named_parameters():
            assert param.grad is None, f"Target encoder.{name} should have no grad"

    def test_targets_are_detached(
        self, device, ijepa_model, context_indices, target_blocks
    ):
        """Target representations should be detached (no grad)."""
        images = torch.randn(2, 3, 32, 32, device=device)

        predictions, targets = ijepa_model(images, context_indices, target_blocks)

        for tgt in targets:
            assert not tgt.requires_grad, "Targets should be detached"


# =============================================================================
# EMA Update Tests
# =============================================================================


class TestEMAUpdate:
    """Tests for EMA update mechanism."""

    def test_ema_update_changes_target(self, device):
        """EMA update should modify target encoder parameters."""
        model = IJEPA().to(device)

        # Store original target params
        original_params = {
            name: param.clone()
            for name, param in model.target_encoder.named_parameters()
        }

        # Modify context encoder
        with torch.no_grad():
            for param in model.context_encoder.parameters():
                param.add_(1.0)

        # Apply EMA update
        model.update_target_encoder(momentum=0.9)

        # Check that target params changed
        for name, param in model.target_encoder.named_parameters():
            assert not torch.allclose(
                param, original_params[name]
            ), f"{name} should have changed"

    def test_ema_momentum_0_copies_context(self, device):
        """With momentum=0, target should become context."""
        model = IJEPA().to(device)

        # Modify context encoder
        with torch.no_grad():
            for param in model.context_encoder.parameters():
                param.fill_(42.0)

        # EMA with momentum=0 means full copy
        model.update_target_encoder(momentum=0.0)

        for param_t, param_c in zip(
            model.target_encoder.parameters(), model.context_encoder.parameters()
        ):
            assert torch.allclose(param_t, param_c), "Should be identical"

    def test_ema_momentum_1_keeps_target(self, device):
        """With momentum=1, target should stay unchanged."""
        model = IJEPA().to(device)

        # Store original target
        original_params = [p.clone() for p in model.target_encoder.parameters()]

        # Modify context encoder
        with torch.no_grad():
            for param in model.context_encoder.parameters():
                param.add_(100.0)

        # EMA with momentum=1 means no update
        model.update_target_encoder(momentum=1.0)

        for param_t, original in zip(model.target_encoder.parameters(), original_params):
            assert torch.allclose(param_t, original), "Should be unchanged"

    def test_ema_formula_correct(self, device):
        """EMA formula: θ̄ = m * θ̄ + (1-m) * θ should be applied correctly."""
        model = IJEPA().to(device)
        momentum = 0.9

        # Get specific parameter values
        ctx_param = list(model.context_encoder.parameters())[0]
        tgt_param = list(model.target_encoder.parameters())[0]

        original_tgt = tgt_param.clone()
        original_ctx = ctx_param.clone()

        # Compute expected value
        expected = momentum * original_tgt + (1 - momentum) * original_ctx

        # Apply EMA
        model.update_target_encoder(momentum)

        assert torch.allclose(
            tgt_param, expected, atol=1e-6
        ), "EMA formula not applied correctly"

    def test_ema_no_gradients_context(self, device):
        """EMA update should not affect context encoder gradients."""
        model = IJEPA().to(device)
        images = torch.randn(2, 3, 32, 32, device=device)
        context_indices = list(range(50))
        target_blocks = [[50, 51, 52, 53]]

        # Forward and backward
        predictions, targets = model(images, context_indices, target_blocks)
        loss = (predictions[0] - targets[0]).pow(2).mean()
        loss.backward()

        # Store gradients
        ctx_grads = [p.grad.clone() for p in model.context_encoder.parameters() if p.grad is not None]

        # EMA update
        model.update_target_encoder(momentum=0.9)

        # Gradients should be unchanged
        new_grads = [p.grad for p in model.context_encoder.parameters() if p.grad is not None]
        for old, new in zip(ctx_grads, new_grads):
            assert torch.allclose(old, new), "EMA should not affect gradients"


# =============================================================================
# Get Target Encoder Tests
# =============================================================================


class TestGetTargetEncoder:
    """Tests for get_target_encoder method."""

    def test_returns_vit_encoder(self, ijepa_model):
        """Should return a ViTEncoder."""
        target_enc = ijepa_model.get_target_encoder()
        assert isinstance(target_enc, ViTEncoder)

    def test_returns_same_instance(self, ijepa_model):
        """Should return the same target encoder instance."""
        enc1 = ijepa_model.get_target_encoder()
        enc2 = ijepa_model.get_target_encoder()
        assert enc1 is enc2

    def test_target_encoder_usable_for_inference(self, device, ijepa_model):
        """Target encoder should work for inference."""
        target_enc = ijepa_model.get_target_encoder()
        target_enc.eval()

        images = torch.randn(2, 3, 32, 32, device=device)
        with torch.no_grad():
            features = target_enc(images)

        assert features.shape == (2, 64, 192)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIJEPAIntegration:
    """Integration tests for full I-JEPA workflow."""

    def test_training_step(self, device, context_indices, target_blocks):
        """Simulate a complete training step."""
        model = IJEPA(
            img_size=32,
            patch_size=4,
            encoder_embed_dim=192,
            encoder_depth=2,  # Smaller for speed
            predictor_depth=2,
        ).to(device)

        optimizer = torch.optim.Adam(
            list(model.context_encoder.parameters())
            + list(model.predictor.parameters()),
            lr=1e-4,
        )

        images = torch.randn(4, 3, 32, 32, device=device)

        # Forward
        predictions, targets = model(images, context_indices, target_blocks)

        # Loss
        loss = sum((p - t).pow(2).mean() for p, t in zip(predictions, targets))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA update
        model.update_target_encoder(momentum=0.996)

        assert loss.item() > 0

    def test_multiple_training_steps(self, device, context_indices, target_blocks):
        """Multiple training steps should progressively update target encoder."""
        model = IJEPA(encoder_depth=2, predictor_depth=2).to(device)

        optimizer = torch.optim.Adam(
            list(model.context_encoder.parameters())
            + list(model.predictor.parameters()),
            lr=1e-3,
        )

        # Store initial target params
        initial_target = [p.clone() for p in model.target_encoder.parameters()]

        for step in range(5):
            images = torch.randn(2, 3, 32, 32, device=device)
            predictions, targets = model(images, context_indices, target_blocks)
            loss = sum((p - t).pow(2).mean() for p, t in zip(predictions, targets))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update with increasing momentum
            momentum = 0.9 + 0.02 * step
            model.update_target_encoder(momentum=momentum)

        # Target should have changed from initial
        for param, initial in zip(model.target_encoder.parameters(), initial_target):
            assert not torch.allclose(param, initial), "Target should have evolved"

    def test_no_nan_during_training(self, device, context_indices, target_blocks):
        """No NaN values should appear during training."""
        model = IJEPA(encoder_depth=2, predictor_depth=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for _ in range(10):
            images = torch.randn(2, 3, 32, 32, device=device)
            predictions, targets = model(images, context_indices, target_blocks)
            loss = sum((p - t).pow(2).mean() for p, t in zip(predictions, targets))

            assert not torch.isnan(loss), "Loss is NaN"

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target_encoder(momentum=0.99)

    def test_eval_mode_deterministic(self, device, context_indices, target_blocks):
        """Eval mode should produce deterministic outputs."""
        model = IJEPA(encoder_depth=2, predictor_depth=2).to(device)
        model.eval()

        images = torch.randn(2, 3, 32, 32, device=device)

        with torch.no_grad():
            pred1, tgt1 = model(images, context_indices, target_blocks)
            pred2, tgt2 = model(images, context_indices, target_blocks)

        for p1, p2 in zip(pred1, pred2):
            assert torch.allclose(p1, p2)

        for t1, t2 in zip(tgt1, tgt2):
            assert torch.allclose(t1, t2)


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestIJEPAStability:
    """Tests for numerical stability."""

    def test_no_nan_output(self, device, context_indices, target_blocks):
        """Outputs should not contain NaN."""
        model = IJEPA().to(device)
        images = torch.randn(2, 3, 32, 32, device=device)

        predictions, targets = model(images, context_indices, target_blocks)

        for pred in predictions:
            assert not torch.isnan(pred).any()
            assert not torch.isinf(pred).any()

        for tgt in targets:
            assert not torch.isnan(tgt).any()
            assert not torch.isinf(tgt).any()

    def test_no_nan_gradients(self, device, context_indices, target_blocks):
        """Gradients should not contain NaN."""
        model = IJEPA().to(device)
        images = torch.randn(2, 3, 32, 32, device=device)

        predictions, targets = model(images, context_indices, target_blocks)
        loss = sum((p - t).pow(2).mean() for p, t in zip(predictions, targets))
        loss.backward()

        for param in model.context_encoder.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()

        for param in model.predictor.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
