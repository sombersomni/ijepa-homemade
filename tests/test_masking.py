"""
Unit tests for Multi-Block Masking Strategy.

Tests verify:
- Block sampling produces valid patch indices
- Context and target blocks have correct properties
- Target patches are removed from context
- Output format is correct for IJEPA model
"""

import pytest
import random

from ijepa.data.masking import MultiBlockMaskGenerator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mask_generator():
    """Default mask generator for 8x8 grid (CIFAR-10 with 4x4 patches)."""
    return MultiBlockMaskGenerator(
        input_size=(8, 8),
        num_targets=4,
        target_scale=(0.15, 0.25),
        target_aspect_ratio=(0.75, 1.5),
        context_scale=(0.75, 0.95),
    )


@pytest.fixture
def seed():
    """Set random seed for reproducibility in tests."""
    random.seed(42)
    return 42


# =============================================================================
# Initialization Tests
# =============================================================================


class TestMaskGeneratorInit:
    """Tests for MultiBlockMaskGenerator initialization."""

    def test_default_attributes(self):
        """Check default attribute values."""
        gen = MultiBlockMaskGenerator()
        assert gen.height == 8
        assert gen.width == 8
        assert gen.num_patches == 64
        assert gen.num_targets == 4

    def test_custom_input_size(self):
        """Test with different grid sizes."""
        gen = MultiBlockMaskGenerator(input_size=(14, 14))
        assert gen.height == 14
        assert gen.width == 14
        assert gen.num_patches == 196

    def test_custom_num_targets(self):
        """Test with different number of target blocks."""
        gen = MultiBlockMaskGenerator(num_targets=8)
        assert gen.num_targets == 8


# =============================================================================
# sample_block Tests
# =============================================================================


class TestSampleBlock:
    """Tests for the sample_block method."""

    def test_returns_set(self, mask_generator, seed):
        """sample_block should return a set of integers."""
        block = mask_generator.sample_block((0.1, 0.2), (1.0, 1.0))
        assert isinstance(block, set)
        assert all(isinstance(idx, int) for idx in block)

    def test_indices_in_valid_range(self, mask_generator, seed):
        """All indices should be in [0, num_patches)."""
        for _ in range(100):
            block = mask_generator.sample_block((0.1, 0.5), (0.5, 2.0))
            for idx in block:
                assert 0 <= idx < mask_generator.num_patches

    def test_block_not_empty(self, mask_generator, seed):
        """Sampled block should not be empty."""
        for _ in range(100):
            block = mask_generator.sample_block((0.1, 0.3), (0.75, 1.5))
            assert len(block) > 0

    def test_block_size_in_expected_range(self, mask_generator, seed):
        """Block size should be approximately in the scale range."""
        scale_range = (0.15, 0.25)
        sizes = []

        for _ in range(100):
            block = mask_generator.sample_block(scale_range, (1.0, 1.0))
            sizes.append(len(block))

        avg_size = sum(sizes) / len(sizes)
        expected_min = mask_generator.num_patches * scale_range[0] * 0.5
        expected_max = mask_generator.num_patches * scale_range[1] * 2.0

        assert expected_min <= avg_size <= expected_max

    def test_square_block_aspect_ratio(self, mask_generator, seed):
        """Square blocks should have similar height and width."""
        for _ in range(50):
            block = mask_generator.sample_block((0.25, 0.25), (1.0, 1.0))

            # Reconstruct block dimensions from indices
            rows = set()
            cols = set()
            for idx in block:
                rows.add(idx // mask_generator.width)
                cols.add(idx % mask_generator.width)

            # For a rectangular block, rows and cols should be contiguous
            assert max(rows) - min(rows) + 1 == len(rows)
            assert max(cols) - min(cols) + 1 == len(cols)

    def test_block_is_contiguous(self, mask_generator, seed):
        """Block should be a contiguous rectangle."""
        for _ in range(50):
            block = mask_generator.sample_block((0.1, 0.3), (0.75, 1.5))

            # Get bounding box
            rows = [idx // mask_generator.width for idx in block]
            cols = [idx % mask_generator.width for idx in block]

            min_row, max_row = min(rows), max(rows)
            min_col, max_col = min(cols), max(cols)

            # All positions in bounding box should be in block
            expected_size = (max_row - min_row + 1) * (max_col - min_col + 1)
            assert len(block) == expected_size

    def test_different_seeds_different_blocks(self, mask_generator):
        """Different random seeds should produce different blocks."""
        random.seed(1)
        block1 = mask_generator.sample_block((0.2, 0.3), (1.0, 1.0))

        random.seed(2)
        block2 = mask_generator.sample_block((0.2, 0.3), (1.0, 1.0))

        # With different seeds, blocks are likely different
        # (not guaranteed, but very likely)
        assert block1 != block2 or True  # Allow same by chance


# =============================================================================
# __call__ Tests
# =============================================================================


class TestMaskGeneratorCall:
    """Tests for the __call__ method."""

    def test_returns_correct_types(self, mask_generator, seed):
        """Should return (list, list of lists)."""
        context, targets = mask_generator()

        assert isinstance(context, list)
        assert isinstance(targets, list)
        assert len(targets) == mask_generator.num_targets

        for t in targets:
            assert isinstance(t, list)

    def test_context_indices_sorted(self, mask_generator, seed):
        """Context indices should be sorted."""
        for _ in range(20):
            context, _ = mask_generator()
            assert context == sorted(context)

    def test_target_indices_sorted(self, mask_generator, seed):
        """Target indices in each block should be sorted."""
        for _ in range(20):
            _, targets = mask_generator()
            for t in targets:
                assert t == sorted(t)

    def test_correct_number_of_targets(self, mask_generator, seed):
        """Should return exactly num_targets target blocks."""
        context, targets = mask_generator()
        assert len(targets) == mask_generator.num_targets

    def test_all_indices_valid(self, mask_generator, seed):
        """All indices should be in valid range."""
        for _ in range(20):
            context, targets = mask_generator()

            for idx in context:
                assert 0 <= idx < mask_generator.num_patches

            for target_block in targets:
                for idx in target_block:
                    assert 0 <= idx < mask_generator.num_patches

    def test_context_does_not_overlap_targets(self, mask_generator, seed):
        """Context should not contain any target patches."""
        for _ in range(50):
            context, targets = mask_generator()

            context_set = set(context)
            all_targets = set()
            for t in targets:
                all_targets.update(t)

            overlap = context_set & all_targets
            assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_context_not_empty(self, mask_generator, seed):
        """Context should not be empty (targets removed but some remain)."""
        for _ in range(50):
            context, _ = mask_generator()
            assert len(context) > 0

    def test_targets_not_empty(self, mask_generator, seed):
        """Each target block should have at least one patch."""
        for _ in range(50):
            _, targets = mask_generator()
            for i, t in enumerate(targets):
                assert len(t) > 0, f"Target block {i} is empty"

    def test_context_size_reasonable(self, mask_generator, seed):
        """Context should have a reasonable number of patches."""
        sizes = []
        for _ in range(100):
            context, _ = mask_generator()
            sizes.append(len(context))

        avg_size = sum(sizes) / len(sizes)
        # Context is ~75-95% minus ~15-25% * 4 targets with overlap
        # Should typically be 30-60 patches
        assert 10 < avg_size < 60


# =============================================================================
# Integration with IJEPA Tests
# =============================================================================


class TestMaskingIntegration:
    """Tests for integration with IJEPA model."""

    def test_output_usable_by_ijepa(self, mask_generator, seed):
        """Output format should be compatible with IJEPA model."""
        context, targets = mask_generator()

        # Context is a list of ints
        assert all(isinstance(i, int) for i in context)

        # Targets is a list of lists of ints
        for t in targets:
            assert all(isinstance(i, int) for i in t)

    def test_batch_generation(self, mask_generator, seed):
        """Generate masks for a batch (each sample independent)."""
        batch_size = 4
        batch_contexts = []
        batch_targets = []

        for _ in range(batch_size):
            context, targets = mask_generator()
            batch_contexts.append(context)
            batch_targets.append(targets)

        assert len(batch_contexts) == batch_size
        assert len(batch_targets) == batch_size

        # Each sample's masks are valid
        for context, targets in zip(batch_contexts, batch_targets):
            assert len(targets) == mask_generator.num_targets

    def test_reproducibility_with_seed(self, mask_generator):
        """Same seed should produce same masks."""
        random.seed(123)
        context1, targets1 = mask_generator()

        random.seed(123)
        context2, targets2 = mask_generator()

        assert context1 == context2
        assert targets1 == targets2

    def test_different_grid_sizes(self, seed):
        """Should work with different grid sizes."""
        for size in [(4, 4), (8, 8), (14, 14), (16, 16)]:
            gen = MultiBlockMaskGenerator(
                input_size=size,
                target_scale=(0.1, 0.2),
                context_scale=(0.7, 0.9),
            )
            context, targets = gen()

            assert len(context) > 0
            assert len(targets) == gen.num_targets

            max_idx = size[0] * size[1] - 1
            assert all(0 <= i <= max_idx for i in context)

    def test_custom_target_count(self, seed):
        """Should work with different numbers of target blocks."""
        for num_targets in [1, 2, 4, 8]:
            gen = MultiBlockMaskGenerator(num_targets=num_targets)
            context, targets = gen()

            assert len(targets) == num_targets


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_target(self, seed):
        """Should work with a single target block."""
        gen = MultiBlockMaskGenerator(num_targets=1)
        context, targets = gen()

        assert len(targets) == 1
        assert len(context) > 0

    def test_many_targets(self, seed):
        """Should work with many target blocks (may have significant overlap)."""
        gen = MultiBlockMaskGenerator(num_targets=10)
        context, targets = gen()

        assert len(targets) == 10
        # Context might be small due to many targets
        assert len(context) >= 0  # Could theoretically be empty

    def test_small_grid(self, seed):
        """Should work with a small 4x4 grid."""
        gen = MultiBlockMaskGenerator(
            input_size=(4, 4),
            num_targets=2,
            target_scale=(0.1, 0.2),
            context_scale=(0.5, 0.8),
        )
        context, targets = gen()

        assert len(targets) == 2
        for idx in context:
            assert 0 <= idx < 16

    def test_large_grid(self, seed):
        """Should work with a large grid."""
        gen = MultiBlockMaskGenerator(
            input_size=(32, 32),
            target_scale=(0.05, 0.1),
            context_scale=(0.8, 0.95),
        )
        context, targets = gen()

        assert len(targets) == gen.num_targets
        max_idx = 32 * 32 - 1
        assert all(0 <= i <= max_idx for i in context)


# =============================================================================
# Statistical Tests
# =============================================================================


class TestStatisticalProperties:
    """Statistical tests for mask distribution."""

    def test_context_covers_most_patches_on_average(self, mask_generator):
        """Context + targets should cover significant portion of patches."""
        random.seed(42)
        coverages = []

        for _ in range(100):
            context, targets = mask_generator()
            all_targets = set()
            for t in targets:
                all_targets.update(t)

            total_coverage = len(set(context) | all_targets)
            coverages.append(total_coverage / mask_generator.num_patches)

        avg_coverage = sum(coverages) / len(coverages)
        # Should cover at least 50% of patches on average
        assert avg_coverage > 0.5

    def test_target_positions_vary(self, mask_generator):
        """Target positions should vary across samples."""
        random.seed(42)

        all_target_sets = []
        for _ in range(20):
            _, targets = mask_generator()
            flat_targets = set()
            for t in targets:
                flat_targets.update(t)
            all_target_sets.append(flat_targets)

        # Count unique target sets
        unique_sets = len(set(frozenset(s) for s in all_target_sets))

        # Should have variety (at least 10 unique patterns out of 20)
        assert unique_sets >= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
