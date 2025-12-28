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
        raise NotImplementedError("sample_block not yet implemented")

    def __call__(self) -> tuple[list[int], list[list[int]]]:
        """
        Generate masks for one sample.

        Returns:
            context_indices: Sorted list of patch indices for context
            target_indices_list: List of M sorted lists, each containing
                                patch indices for one target block
        """
        raise NotImplementedError("MultiBlockMaskGenerator.__call__ not yet implemented")
