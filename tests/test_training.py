"""
Unit tests for I-JEPA training utilities.

Tests verify:
- Learning rate and momentum schedulers
- EMA update mechanism
- Loss function
- Training step integration
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from ijepa.training.scheduler import cosine_scheduler, get_momentum_schedule
from ijepa.training.ema import update_ema
from ijepa.training.train import ijepa_loss


# =============================================================================
# Scheduler Tests
# =============================================================================


class TestCosineScheduler:
    """Tests for cosine learning rate scheduler."""

    def test_output_shape(self):
        """Output should have correct length."""
        schedule = cosine_scheduler(
            base_value=1e-3,
            final_value=1e-5,
            epochs=10,
            steps_per_epoch=100,
        )
        assert len(schedule) == 10 * 100

    def test_starts_at_base_value_no_warmup(self):
        """Without warmup, should start at base value."""
        schedule = cosine_scheduler(
            base_value=1e-3,
            final_value=1e-5,
            epochs=10,
            steps_per_epoch=100,
            warmup_epochs=0,
        )
        assert abs(schedule[0] - 1e-3) < 1e-6

    def test_ends_at_final_value(self):
        """Should end at final value."""
        schedule = cosine_scheduler(
            base_value=1e-3,
            final_value=1e-5,
            epochs=10,
            steps_per_epoch=100,
        )
        assert abs(schedule[-1] - 1e-5) < 1e-6

    def test_warmup_starts_at_warmup_value(self):
        """With warmup, should start at warmup_start_value."""
        schedule = cosine_scheduler(
            base_value=1e-3,
            final_value=1e-5,
            epochs=10,
            steps_per_epoch=100,
            warmup_epochs=2,
            warmup_start_value=0.0,
        )
        assert schedule[0] == 0.0

    def test_warmup_reaches_base_value(self):
        """Warmup should reach base value at end of warmup."""
        schedule = cosine_scheduler(
            base_value=1e-3,
            final_value=1e-5,
            epochs=10,
            steps_per_epoch=100,
            warmup_epochs=2,
            warmup_start_value=0.0,
        )
        warmup_end_idx = 2 * 100 - 1
        assert abs(schedule[warmup_end_idx] - 1e-3) < 1e-5

    def test_monotonic_decrease_after_warmup(self):
        """After warmup, values should generally decrease."""
        schedule = cosine_scheduler(
            base_value=1e-3,
            final_value=1e-5,
            epochs=10,
            steps_per_epoch=100,
            warmup_epochs=1,
        )
        after_warmup = schedule[100:]
        # Check overall trend (first > last)
        assert after_warmup[0] > after_warmup[-1]

    def test_warmup_linear_increase(self):
        """Warmup should be approximately linear."""
        schedule = cosine_scheduler(
            base_value=1.0,
            final_value=0.1,
            epochs=10,
            steps_per_epoch=100,
            warmup_epochs=2,
            warmup_start_value=0.0,
        )
        warmup = schedule[:200]
        # Check it's increasing
        assert warmup[-1] > warmup[0]
        # Check approximately linear (midpoint should be ~0.5)
        assert 0.4 < warmup[100] < 0.6


class TestMomentumSchedule:
    """Tests for EMA momentum scheduler."""

    def test_output_shape(self):
        """Output should have correct length."""
        schedule = get_momentum_schedule(
            base_momentum=0.996,
            final_momentum=1.0,
            epochs=10,
            steps_per_epoch=100,
        )
        assert len(schedule) == 10 * 100

    def test_starts_at_base(self):
        """Should start at base momentum."""
        schedule = get_momentum_schedule(
            base_momentum=0.996,
            final_momentum=1.0,
            epochs=10,
            steps_per_epoch=100,
        )
        assert abs(schedule[0] - 0.996) < 1e-6

    def test_ends_at_final(self):
        """Should end at final momentum."""
        schedule = get_momentum_schedule(
            base_momentum=0.996,
            final_momentum=1.0,
            epochs=10,
            steps_per_epoch=100,
        )
        assert abs(schedule[-1] - 1.0) < 1e-6

    def test_monotonic_increase(self):
        """Should monotonically increase."""
        schedule = get_momentum_schedule(
            base_momentum=0.996,
            final_momentum=1.0,
            epochs=10,
            steps_per_epoch=100,
        )
        diffs = np.diff(schedule)
        assert np.all(diffs >= 0)

    def test_linear_increase(self):
        """Should be linear."""
        schedule = get_momentum_schedule(
            base_momentum=0.0,
            final_momentum=1.0,
            epochs=10,
            steps_per_epoch=100,
        )
        # Midpoint should be ~0.5
        mid_idx = len(schedule) // 2
        assert 0.49 < schedule[mid_idx] < 0.51


# =============================================================================
# EMA Update Tests
# =============================================================================


class TestEMAUpdate:
    """Tests for EMA update function."""

    def test_momentum_zero_copies(self):
        """Momentum=0 should copy source to target."""
        target = nn.Linear(10, 10)
        source = nn.Linear(10, 10)

        # Initialize differently
        with torch.no_grad():
            target.weight.fill_(1.0)
            source.weight.fill_(2.0)

        update_ema(target, source, momentum=0.0)

        assert torch.allclose(target.weight, source.weight)

    def test_momentum_one_keeps_target(self):
        """Momentum=1 should keep target unchanged."""
        target = nn.Linear(10, 10)
        source = nn.Linear(10, 10)

        with torch.no_grad():
            target.weight.fill_(1.0)
            source.weight.fill_(2.0)
            original = target.weight.clone()

        update_ema(target, source, momentum=1.0)

        assert torch.allclose(target.weight, original)

    def test_ema_formula(self):
        """Should apply correct EMA formula."""
        target = nn.Linear(10, 10)
        source = nn.Linear(10, 10)

        with torch.no_grad():
            target.weight.fill_(1.0)
            source.weight.fill_(2.0)

        momentum = 0.9
        expected = momentum * 1.0 + (1 - momentum) * 2.0  # = 1.1

        update_ema(target, source, momentum=momentum)

        assert torch.allclose(target.weight, torch.full_like(target.weight, expected))

    def test_no_gradients(self):
        """Should not create gradients."""
        target = nn.Linear(10, 10)
        source = nn.Linear(10, 10)

        update_ema(target, source, momentum=0.9)

        assert target.weight.grad is None
        assert source.weight.grad is None

    def test_works_with_complex_model(self):
        """Should work with multi-layer models."""
        target = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        source = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

        # Initialize differently
        with torch.no_grad():
            for p in target.parameters():
                p.fill_(1.0)
            for p in source.parameters():
                p.fill_(2.0)

        update_ema(target, source, momentum=0.5)

        # All params should be 1.5 = 0.5*1 + 0.5*2
        for p in target.parameters():
            assert torch.allclose(p, torch.full_like(p, 1.5))


# =============================================================================
# Loss Function Tests
# =============================================================================


class TestIJEPALoss:
    """Tests for I-JEPA loss function."""

    def test_zero_loss_for_identical(self):
        """Loss should be zero for identical predictions and targets."""
        pred = torch.randn(2, 4, 192)
        loss = ijepa_loss([pred], [pred.clone()])
        assert loss.item() < 1e-6

    def test_positive_loss_for_different(self):
        """Loss should be positive for different predictions and targets."""
        pred = torch.randn(2, 4, 192)
        target = torch.randn(2, 4, 192)
        loss = ijepa_loss([pred], [target])
        assert loss.item() > 0

    def test_multiple_target_blocks(self):
        """Should handle multiple target blocks."""
        predictions = [torch.randn(2, 4, 192) for _ in range(4)]
        targets = [torch.randn(2, 4, 192) for _ in range(4)]
        loss = ijepa_loss(predictions, targets)
        assert loss.item() > 0

    def test_gradient_flow(self):
        """Gradients should flow through loss."""
        pred = torch.randn(2, 4, 192, requires_grad=True)
        target = torch.randn(2, 4, 192)
        loss = ijepa_loss([pred], [target])
        loss.backward()
        assert pred.grad is not None

    def test_loss_scales_with_difference(self):
        """Larger differences should give larger loss."""
        pred = torch.zeros(2, 4, 192)
        target1 = torch.ones(2, 4, 192)
        target2 = torch.ones(2, 4, 192) * 2

        loss1 = ijepa_loss([pred], [target1])
        loss2 = ijepa_loss([pred], [target2])

        assert loss2.item() > loss1.item()

    def test_different_block_sizes(self):
        """Should handle blocks of different sizes."""
        predictions = [
            torch.randn(2, 4, 192),
            torch.randn(2, 6, 192),
            torch.randn(2, 2, 192),
        ]
        targets = [
            torch.randn(2, 4, 192),
            torch.randn(2, 6, 192),
            torch.randn(2, 2, 192),
        ]
        loss = ijepa_loss(predictions, targets)
        assert loss.item() > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestTrainingIntegration:
    """Integration tests for training components."""

    def test_schedules_same_length(self):
        """LR and momentum schedules should have same length."""
        epochs = 10
        steps_per_epoch = 100

        lr_schedule = cosine_scheduler(
            base_value=1e-3,
            final_value=1e-5,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=1,
        )

        ema_schedule = get_momentum_schedule(
            base_momentum=0.996,
            final_momentum=1.0,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )

        assert len(lr_schedule) == len(ema_schedule)

    def test_ema_with_ijepa_model(self):
        """EMA should work with IJEPA model components."""
        from ijepa.models.vit import ViTEncoder

        context = ViTEncoder(img_size=32, patch_size=4, depth=2)
        target = ViTEncoder(img_size=32, patch_size=4, depth=2)

        # Initialize differently
        with torch.no_grad():
            for p in context.parameters():
                p.fill_(2.0)
            for p in target.parameters():
                p.fill_(1.0)

        update_ema(target, context, momentum=0.9)

        # Check params changed
        for p in target.parameters():
            expected = 0.9 * 1.0 + 0.1 * 2.0  # = 1.1
            assert torch.allclose(p, torch.full_like(p, expected), atol=1e-5)

    def test_loss_with_model_output(self):
        """Loss should work with actual model outputs."""
        from ijepa.models.ijepa import IJEPA
        from ijepa.data.masking import MultiBlockMaskGenerator

        model = IJEPA(img_size=32, patch_size=4, encoder_depth=2, predictor_depth=2)
        mask_gen = MultiBlockMaskGenerator(input_size=(8, 8))

        images = torch.randn(2, 3, 32, 32)
        context_idx, target_idx_list = mask_gen()

        predictions, targets = model(images, context_idx, target_idx_list)
        loss = ijepa_loss(predictions, targets)

        assert loss.item() > 0
        loss.backward()

        # Check gradients exist for context encoder
        for p in model.context_encoder.parameters():
            assert p.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
