"""
PCA visualization with variance explained analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from ijepa.evaluation.visualize import CIFAR10_CLASSES


def compute_pca(
    features: np.ndarray,
    n_components: int = 50,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Compute PCA on features.

    Args:
        features: (N, D) feature matrix
        n_components: Number of components to compute

    Returns:
        transformed: (N, n_components) PCA-transformed features
        variance_explained: (n_components,) variance ratio per component
        pca: Fitted PCA object
    """
    n_components = min(n_components, features.shape[1], features.shape[0])
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(features)
    return transformed, pca.explained_variance_ratio_, pca


def plot_pca_2d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    variance_explained: np.ndarray,
    title: str = "PCA Visualization",
    class_names: list[str] = None,
    figsize: tuple = (12, 10),
    save_path: str = None,
    show: bool = False,
) -> plt.Figure:
    """Plot first 2 PCA components colored by class."""
    if class_names is None:
        class_names = CIFAR10_CLASSES

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.get_cmap('tab10')

    for class_idx in range(len(class_names)):
        mask = labels == class_idx
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[cmap(class_idx)],
            label=class_names[class_idx],
            alpha=0.6,
            s=10,
        )

    var_pc1 = variance_explained[0] * 100
    var_pc2 = variance_explained[1] * 100
    ax.set_xlabel(f"PC1 ({var_pc1:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_pc2:.1f}% variance)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10, markerscale=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved PCA plot to {save_path}")

    if show:
        plt.show()

    return fig


def plot_variance_explained(
    variance_ratios: dict[str, np.ndarray],
    n_components: int = 20,
    figsize: tuple = (12, 5),
    save_path: str = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot cumulative variance explained for multiple models.

    Args:
        variance_ratios: Dict mapping model name to variance ratio array
        n_components: Number of components to show
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.get_cmap('tab10')
    width = 0.8 / len(variance_ratios)

    # Per-component variance
    for i, (name, ratios) in enumerate(variance_ratios.items()):
        n = min(n_components, len(ratios))
        x = np.arange(n) + i * width
        ax1.bar(
            x,
            ratios[:n] * 100,
            width=width,
            alpha=0.7,
            label=name,
            color=colors(i),
        )
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("Per-Component Variance")
    ax1.legend()
    ax1.set_xticks(np.arange(n_components) + width * (len(variance_ratios) - 1) / 2)
    ax1.set_xticklabels([str(i+1) for i in range(n_components)])

    # Cumulative variance
    for i, (name, ratios) in enumerate(variance_ratios.items()):
        n = min(n_components, len(ratios))
        cumulative = np.cumsum(ratios[:n]) * 100
        ax2.plot(range(1, n + 1), cumulative, marker='o', label=name, color=colors(i))

    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance Explained (%)")
    ax2.set_title("Cumulative Variance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved variance plot to {save_path}")

    if show:
        plt.show()

    return fig
