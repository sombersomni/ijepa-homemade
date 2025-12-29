"""
Unit tests for CIFAR-10 data loading utilities.

Tests verify:
- Data loaders return correct types
- Batch shapes are correct
- Normalization is applied
- Train/test splits are correct
"""

import pytest
import torch

from ijepa.data.cifar10 import get_cifar10_loaders, CIFAR10_MEAN, CIFAR10_STD


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def data_loaders():
    """Load CIFAR-10 data loaders (cached for module)."""
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=32,
        data_dir='./data',
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
    )
    return train_loader, test_loader


@pytest.fixture
def train_loader(data_loaders):
    return data_loaders[0]


@pytest.fixture
def test_loader(data_loaders):
    return data_loaders[1]


# =============================================================================
# Basic Tests
# =============================================================================


class TestDataLoaderBasics:
    """Basic tests for data loaders."""

    def test_returns_two_loaders(self):
        """Should return train and test loaders."""
        train_loader, test_loader = get_cifar10_loaders(
            batch_size=32, num_workers=0
        )
        assert train_loader is not None
        assert test_loader is not None

    def test_train_loader_is_dataloader(self, train_loader):
        """Train loader should be a DataLoader."""
        assert isinstance(train_loader, torch.utils.data.DataLoader)

    def test_test_loader_is_dataloader(self, test_loader):
        """Test loader should be a DataLoader."""
        assert isinstance(test_loader, torch.utils.data.DataLoader)

    def test_dataset_sizes(self, train_loader, test_loader):
        """Check dataset sizes are correct for CIFAR-10."""
        # CIFAR-10 has 50,000 train and 10,000 test images
        assert len(train_loader.dataset) == 50000
        assert len(test_loader.dataset) == 10000


# =============================================================================
# Batch Tests
# =============================================================================


class TestBatchProperties:
    """Tests for batch properties."""

    def test_train_batch_shape(self, train_loader):
        """Train batch should have correct shape."""
        images, labels = next(iter(train_loader))

        assert images.shape == (32, 3, 32, 32)
        assert labels.shape == (32,)

    def test_test_batch_shape(self, test_loader):
        """Test batch should have correct shape."""
        images, labels = next(iter(test_loader))

        assert images.shape == (32, 3, 32, 32)
        assert labels.shape == (32,)

    def test_images_are_float_tensors(self, train_loader):
        """Images should be float tensors."""
        images, _ = next(iter(train_loader))
        assert images.dtype == torch.float32

    def test_labels_are_long_tensors(self, train_loader):
        """Labels should be long tensors."""
        _, labels = next(iter(train_loader))
        assert labels.dtype == torch.int64

    def test_labels_in_valid_range(self, train_loader):
        """Labels should be in [0, 9] for CIFAR-10."""
        _, labels = next(iter(train_loader))
        assert labels.min() >= 0
        assert labels.max() <= 9

    def test_custom_batch_size(self):
        """Should respect custom batch size."""
        train_loader, test_loader = get_cifar10_loaders(
            batch_size=64, num_workers=0
        )

        images, _ = next(iter(train_loader))
        assert images.shape[0] == 64


# =============================================================================
# Normalization Tests
# =============================================================================


class TestNormalization:
    """Tests for data normalization."""

    def test_normalization_constants_exist(self):
        """Normalization constants should be defined."""
        assert CIFAR10_MEAN is not None
        assert CIFAR10_STD is not None

    def test_normalization_constants_correct(self):
        """Normalization constants should match expected values."""
        assert len(CIFAR10_MEAN) == 3
        assert len(CIFAR10_STD) == 3

        # Check approximate values
        assert abs(CIFAR10_MEAN[0] - 0.4914) < 0.01
        assert abs(CIFAR10_STD[0] - 0.247) < 0.01

    def test_images_are_normalized(self, train_loader):
        """Images should be normalized (not in [0, 1] range)."""
        images, _ = next(iter(train_loader))

        # After normalization with CIFAR-10 stats, values should
        # be roughly centered around 0
        # Min should be around -2 to -1, max around 2 to 3
        assert images.min() < 0, "Normalized images should have negative values"
        assert images.max() > 1, "Normalized images should exceed 1"


# =============================================================================
# DataLoader Settings Tests
# =============================================================================


class TestDataLoaderSettings:
    """Tests for DataLoader configuration."""

    def test_train_loader_shuffles(self):
        """Train loader should shuffle data."""
        train_loader, _ = get_cifar10_loaders(batch_size=32, num_workers=0)

        # Get first batch twice (with new loaders to reset)
        batch1_labels = next(iter(train_loader))[1].tolist()

        train_loader2, _ = get_cifar10_loaders(batch_size=32, num_workers=0)
        batch2_labels = next(iter(train_loader2))[1].tolist()

        # With shuffling, batches should differ (very high probability)
        # Note: There's a tiny chance they're the same by random chance
        # We'll just check the loaders are configured for shuffling
        assert train_loader.batch_sampler.sampler.__class__.__name__ == 'RandomSampler'

    def test_test_loader_no_shuffle(self, test_loader):
        """Test loader should not shuffle data."""
        assert test_loader.batch_sampler.sampler.__class__.__name__ == 'SequentialSampler'

    def test_train_loader_drops_last(self, train_loader):
        """Train loader should drop last incomplete batch."""
        assert train_loader.drop_last is True

    def test_test_loader_keeps_last(self, test_loader):
        """Test loader should keep last batch."""
        assert test_loader.drop_last is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with I-JEPA model."""

    def test_batch_compatible_with_model(self, train_loader):
        """Batch should be compatible with I-JEPA model input."""
        from ijepa.models.ijepa import IJEPA

        model = IJEPA(img_size=32, patch_size=4)
        images, _ = next(iter(train_loader))

        # Should be able to pass through patch embedding
        patches = model.context_encoder.patch_embed(images)
        assert patches.shape == (32, 64, 192)

    def test_iteration_works(self, train_loader):
        """Should be able to iterate over loader."""
        count = 0
        for images, labels in train_loader:
            count += 1
            if count >= 3:
                break

        assert count == 3

    def test_full_epoch_iteration(self, test_loader):
        """Should be able to iterate full epoch over test set."""
        total_samples = 0
        for images, labels in test_loader:
            total_samples += images.shape[0]

        # Test set has 10,000 images
        assert total_samples == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
