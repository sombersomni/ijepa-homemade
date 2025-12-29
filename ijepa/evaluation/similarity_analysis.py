"""
Cosine similarity analysis between class centroids.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

from ijepa.evaluation.visualize import CIFAR10_CLASSES


def compute_class_centroids(
    features: np.ndarray,
    labels: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute centroid (mean) for each class.

    Args:
        features: (N, D) feature matrix
        labels: (N,) class labels
        normalize: If True, L2-normalize centroids

    Returns:
        centroids: (num_classes, D) centroid matrix
    """
    classes = np.unique(labels)
    centroids = []

    for cls in sorted(classes):
        mask = labels == cls
        centroid = features[mask].mean(axis=0)
        if normalize:
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids.append(centroid)

    return np.stack(centroids)


def compute_centroid_similarity_matrix(
    features: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity matrix between class centroids.

    Returns:
        similarity: (num_classes, num_classes) similarity matrix
    """
    centroids = compute_class_centroids(features, labels, normalize=True)
    return cosine_similarity(centroids)


def plot_similarity_matrix(
    similarity: np.ndarray,
    title: str = "Class Centroid Cosine Similarity",
    class_names: list[str] = None,
    figsize: tuple = (10, 8),
    save_path: str = None,
    show: bool = False,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> plt.Figure:
    """Plot heatmap of similarity matrix."""
    if class_names is None:
        class_names = CIFAR10_CLASSES

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        similarity,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        square=True,
    )

    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved similarity matrix to {save_path}")

    if show:
        plt.show()

    return fig


def compare_similarity_matrices(
    similarities: dict[str, np.ndarray],
    class_names: list[str] = None,
    figsize: tuple = (16, 6),
    save_path: str = None,
    show: bool = False,
) -> plt.Figure:
    """Plot multiple similarity matrices side-by-side."""
    if class_names is None:
        class_names = CIFAR10_CLASSES

    n_models = len(similarities)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    for i, (ax, (name, sim)) in enumerate(zip(axes, similarities.items())):
        sns.heatmap(
            sim,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            xticklabels=class_names,
            yticklabels=class_names,
            vmin=-1.0,
            vmax=1.0,
            ax=ax,
            square=True,
            cbar=(i == n_models - 1),  # Only show colorbar on last plot
        )
        ax.set_title(name, fontsize=12)
        ax.set_xticklabels(class_names, rotation=45, ha='right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison matrices to {save_path}")

    if show:
        plt.show()

    return fig
