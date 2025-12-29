"""
Nearest neighbor retrieval for qualitative evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

from ijepa.evaluation.visualize import CIFAR10_CLASSES
from ijepa.data.cifar10 import CIFAR10_MEAN, CIFAR10_STD


class NearestNeighborRetriever:
    """Find nearest neighbors in embedding space."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        metric: str = 'cosine',
    ):
        """
        Args:
            features: (N, D) feature matrix (database)
            labels: (N,) class labels
            metric: Distance metric ('cosine', 'euclidean')
        """
        self.features = features
        self.labels = labels
        self.metric = metric

        # Normalize for cosine similarity
        if metric == 'cosine':
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            self.normalized_features = features / (norms + 1e-8)
        else:
            self.normalized_features = features

        self.nn = NearestNeighbors(n_neighbors=20, metric=metric)
        self.nn.fit(self.normalized_features)

    def find_neighbors(
        self,
        query_idx: int,
        k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for a query.

        Returns:
            indices: (k,) neighbor indices
            distances: (k,) distances
            labels: (k,) neighbor labels
        """
        query = self.normalized_features[query_idx:query_idx+1]
        distances, indices = self.nn.kneighbors(query, n_neighbors=k+1)

        # Exclude self (first result)
        indices = indices[0, 1:k+1]
        distances = distances[0, 1:k+1]
        neighbor_labels = self.labels[indices]

        return indices, distances, neighbor_labels

    def compute_retrieval_accuracy(
        self,
        k: int = 5,
        num_queries: int = 1000,
    ) -> dict:
        """
        Compute retrieval accuracy metrics.

        Returns:
            Dict with precision@k, per-class precision
        """
        np.random.seed(42)
        query_indices = np.random.choice(len(self.features), num_queries, replace=False)

        correct_at_k = 0
        per_class_correct = {i: 0 for i in range(10)}
        per_class_total = {i: 0 for i in range(10)}

        for idx in query_indices:
            true_label = self.labels[idx]
            _, _, neighbor_labels = self.find_neighbors(idx, k=k)

            # Precision@k: fraction of neighbors with same class
            matches = (neighbor_labels == true_label).sum()
            correct_at_k += matches
            per_class_correct[true_label] += matches
            per_class_total[true_label] += k

        overall_precision = correct_at_k / (num_queries * k)
        per_class_precision = {
            cls: per_class_correct[cls] / max(per_class_total[cls], 1)
            for cls in range(10)
        }

        return {
            'precision@k': overall_precision,
            'per_class_precision': per_class_precision,
            'k': k,
            'num_queries': num_queries,
        }


def visualize_nearest_neighbors(
    query_idx: int,
    neighbor_indices: np.ndarray,
    dataset: Dataset,
    query_label: int,
    neighbor_labels: np.ndarray,
    distances: np.ndarray,
    model_name: str = "",
    class_names: list[str] = None,
    figsize: tuple = (15, 3),
    save_path: str = None,
    show: bool = False,
) -> plt.Figure:
    """
    Visualize query image and its nearest neighbors.

    Args:
        query_idx: Index of query image
        neighbor_indices: Indices of neighbor images
        dataset: Dataset to retrieve raw images from
        query_label: True label of query
        neighbor_labels: Labels of neighbors
        distances: Distances to neighbors
        model_name: Name of model for title
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES

    k = len(neighbor_indices)
    fig, axes = plt.subplots(1, k + 1, figsize=figsize)

    # Denormalize CIFAR-10 for display
    cifar_mean = np.array(CIFAR10_MEAN)
    cifar_std = np.array(CIFAR10_STD)

    def denorm(img):
        img = img.numpy().transpose(1, 2, 0)
        img = img * cifar_std + cifar_mean
        return np.clip(img, 0, 1)

    # Query image
    query_img, _ = dataset[query_idx]
    axes[0].imshow(denorm(query_img))
    axes[0].set_title(f"Query\n{class_names[query_label]}", fontsize=10)
    axes[0].axis('off')
    # Blue border for query
    for spine in axes[0].spines.values():
        spine.set_edgecolor('blue')
        spine.set_linewidth(3)
        spine.set_visible(True)

    # Neighbors
    for i, (idx, label, dist) in enumerate(zip(neighbor_indices, neighbor_labels, distances)):
        img, _ = dataset[idx]
        axes[i + 1].imshow(denorm(img))

        color = 'green' if label == query_label else 'red'
        axes[i + 1].set_title(f"{class_names[label]}\nd={dist:.3f}", fontsize=9)
        axes[i + 1].axis('off')
        # Colored border based on match
        for spine in axes[i + 1].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
            spine.set_visible(True)

    plt.suptitle(f"Nearest Neighbors - {model_name}", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved NN visualization to {save_path}")

    if show:
        plt.show()

    return fig
