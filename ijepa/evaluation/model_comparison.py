"""
Side-by-side comparison utilities for different models.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

from ijepa.evaluation.visualize import compute_tsne, CIFAR10_CLASSES
from ijepa.evaluation.pca_visualization import compute_pca
from ijepa.evaluation.nearest_neighbor import NearestNeighborRetriever


def compare_tsne(
    model_features: Dict[str, tuple[np.ndarray, np.ndarray]],
    perplexity: float = 30.0,
    figsize: tuple = (20, 8),
    save_path: str = None,
    show: bool = False,
) -> plt.Figure:
    """
    Generate side-by-side t-SNE visualizations.

    Args:
        model_features: Dict mapping model name to (features, labels) tuple
        perplexity: t-SNE perplexity parameter
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display the plot
    """
    n_models = len(model_features)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    cmap = plt.cm.get_cmap('tab10')

    for ax, (name, (features, labels)) in zip(axes, model_features.items()):
        print(f"Computing t-SNE for {name}...")
        embeddings = compute_tsne(features, perplexity=perplexity)

        for class_idx in range(10):
            mask = labels == class_idx
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=[cmap(class_idx)],
                label=CIFAR10_CLASSES[class_idx],
                alpha=0.6,
                s=10,
            )

        ax.set_title(f"t-SNE: {name}", fontsize=12)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.grid(True, alpha=0.3)

    # Single legend for all
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='center right', fontsize=9)

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved t-SNE comparison to {save_path}")

    if show:
        plt.show()

    return fig


def compare_pca(
    model_features: Dict[str, tuple[np.ndarray, np.ndarray]],
    figsize: tuple = (20, 8),
    save_path: str = None,
    show: bool = False,
) -> plt.Figure:
    """Generate side-by-side PCA visualizations."""
    n_models = len(model_features)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    cmap = plt.cm.get_cmap('tab10')

    for ax, (name, (features, labels)) in zip(axes, model_features.items()):
        pca_embeddings, variance, _ = compute_pca(features, n_components=2)

        for class_idx in range(10):
            mask = labels == class_idx
            ax.scatter(
                pca_embeddings[mask, 0],
                pca_embeddings[mask, 1],
                c=[cmap(class_idx)],
                label=CIFAR10_CLASSES[class_idx],
                alpha=0.6,
                s=10,
            )

        var_str = f"PC1: {variance[0]*100:.1f}%, PC2: {variance[1]*100:.1f}%"
        ax.set_title(f"PCA: {name}\n{var_str}", fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='center right', fontsize=9)

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved PCA comparison to {save_path}")

    if show:
        plt.show()

    return fig


def generate_comparison_report(
    model_features: Dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: str = "comparison_results",
) -> dict:
    """
    Generate comprehensive comparison report.

    Returns metrics dict with:
        - PCA variance explained
        - Class separability (silhouette score)
        - NN retrieval precision
    """
    from sklearn.metrics import silhouette_score
    import os

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for name, (features, labels) in model_features.items():
        # PCA analysis
        _, variance, _ = compute_pca(features, n_components=50)

        # Silhouette score (class separability)
        sample_size = min(5000, len(features))
        sil_score = silhouette_score(
            features, labels,
            sample_size=sample_size,
            random_state=42,
        )

        # NN retrieval
        retriever = NearestNeighborRetriever(features, labels, metric='cosine')
        num_queries = min(1000, len(features) - 1)  # Ensure we don't query more than available
        nn_metrics = retriever.compute_retrieval_accuracy(k=5, num_queries=num_queries)

        results[name] = {
            'variance_explained_10pc': float(variance[:10].sum()),
            'variance_explained_50pc': float(variance[:min(50, len(variance))].sum()),
            'silhouette_score': float(sil_score),
            'nn_precision@5': float(nn_metrics['precision@k']),
            'embed_dim': int(features.shape[1]),
            'num_samples': int(features.shape[0]),
        }

        print(f"\n{name}:")
        print(f"  Embedding dim: {features.shape[1]}")
        print(f"  Variance in 10 PCs: {variance[:10].sum()*100:.1f}%")
        print(f"  Silhouette score: {sil_score:.3f}")
        print(f"  NN Precision@5: {nn_metrics['precision@k']*100:.1f}%")

    return results
