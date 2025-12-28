"""
I-JEPA: Image-based Joint-Embedding Predictive Architecture

A self-supervised learning method that predicts representations of image patches
in embedding space rather than pixel space.
"""

from ijepa.models import IJEPA, ViTEncoder, Predictor

__all__ = ['IJEPA', 'ViTEncoder', 'Predictor']
