"""I-JEPA Evaluation Utilities"""

from ijepa.evaluation.linear_probe import (
    evaluate_linear_probe,
    extract_features,
    train_linear_classifier,
    LinearProbe,
)

from ijepa.evaluation.visualize import (
    visualize_features,
    compute_tsne,
    plot_tsne,
    extract_features_subset,
    CIFAR10_CLASSES,
)

__all__ = [
    'evaluate_linear_probe',
    'extract_features',
    'train_linear_classifier',
    'LinearProbe',
    'visualize_features',
    'compute_tsne',
    'plot_tsne',
    'extract_features_subset',
    'CIFAR10_CLASSES',
]
