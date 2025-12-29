"""
Multi-Block Masking Strategy for I-JEPA

Generates context and target masks following the I-JEPA paper:
- Multiple small target blocks (to predict)
- One large context block (visible)
- Target patches removed from context to ensure non-trivial prediction
"""

import random
import math


class MultiBlockMaskGenerator:
    """
    Generate context and target masks for I-JEPA training.

    Target blocks:
        - Sample M=4 (possibly overlapping) blocks
        - Scale range: (0.15, 0.25) of image area
        - Aspect ratio range: (0.75, 1.5)

    Context block:
        - Sample 1 large block
        - Scale range: (0.75, 0.95) of image area
        - Aspect ratio: 1.0 (square)
        - Remove overlap with target blocks

    Args:
        input_size: Tuple of (height, width) in patches (e.g., (8, 8) for CIFAR-10)
        num_targets: Number of target blocks to sample
        target_scale: (min, max) scale for target blocks
        target_aspect_ratio: (min, max) aspect ratio for target blocks
        context_scale: (min, max) scale for context block
    """

    def __init__(
        self,
        input_size: tuple[int, int] = (8, 8),
        num_targets: int = 4,
        target_scale: tuple[float, float] = (0.15, 0.25),
        target_aspect_ratio: tuple[float, float] = (0.75, 1.5),
        context_scale: tuple[float, float] = (0.75, 0.95),
    ):
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_targets = num_targets
        self.target_scale = target_scale
        self.target_aspect_ratio = target_aspect_ratio
        self.context_scale = context_scale

    def sample_block(
        self,
        scale_range: tuple[float, float],
        aspect_ratio_range: tuple[float, float],
    ) -> set[int]:
        """
        Sample a rectangular block of patches.

        Args:
            scale_range: (min, max) fraction of total patches
            aspect_ratio_range: (min, max) aspect ratio (width/height)

        Returns:
            Set of patch indices in the sampled block
        """
        # Sample scale (fraction of total patches)
        scale = random.uniform(*scale_range)
        num_patches_in_block = int(self.num_patches * scale)

        # Sample aspect ratio (width / height)
        aspect_ratio = random.uniform(*aspect_ratio_range)

        # Calculate block dimensions
        # height * width = num_patches_in_block
        # width / height = aspect_ratio
        # Therefore: height = sqrt(num_patches_in_block / aspect_ratio)
        block_height = int(round(math.sqrt(num_patches_in_block / aspect_ratio)))
        block_width = int(round(block_height * aspect_ratio))

        # Clamp to valid range [1, grid_size]
        block_height = max(1, min(block_height, self.height))
        block_width = max(1, min(block_width, self.width))

        # Sample top-left corner position
        top = random.randint(0, self.height - block_height)
        left = random.randint(0, self.width - block_width)

        # Get patch indices in the block
        indices = set()
        for row in range(top, top + block_height):
            for col in range(left, left + block_width):
                patch_idx = row * self.width + col
                indices.add(patch_idx)

        return indices

    def __call__(self) -> tuple[list[int], list[list[int]]]:
        """
        Generate masks for one sample.

        Returns:
            context_indices: Sorted list of patch indices for context
            target_indices_list: List of M sorted lists, each containing
                                patch indices for one target block
        """
        # Sample target blocks
        target_indices_list = []
        all_target_indices = set()

        for _ in range(self.num_targets):
            target_block = self.sample_block(
                self.target_scale, self.target_aspect_ratio
            )
            target_indices_list.append(target_block)
            all_target_indices.update(target_block)

        # Sample context block (large, square)
        context_block = self.sample_block(self.context_scale, (1.0, 1.0))

        # Remove target patches from context (ensure non-trivial prediction)
        context_indices = context_block - all_target_indices

        # Convert to sorted lists
        context_indices = sorted(list(context_indices))
        target_indices_list = [sorted(list(t)) for t in target_indices_list]

        return context_indices, target_indices_list
