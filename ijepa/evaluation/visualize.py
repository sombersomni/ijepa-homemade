"""
Visualization utilities for I-JEPA learned representations.

Includes t-SNE visualization of encoder features.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from ijepa.models import ViTEncoder


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


@torch.no_grad()
def extract_features_subset(
    encoder: ViTEncoder,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract features from encoder for a subset of samples.

    Args:
        encoder: Frozen encoder
        dataloader: Data loader with images and labels
        device: Device to run on
        max_samples: Maximum number of samples to extract

    Returns:
        features: (N, embed_dim) feature vectors as numpy array
        labels: (N,) class labels as numpy array
    """
    encoder.eval()
    all_features = []
    all_labels = []
    total_samples = 0

    for images, labels in tqdm(dataloader, desc="Extracting features"):
        if total_samples >= max_samples:
            break

        images = images.to(device)

        # Get encoder output (B, num_patches, embed_dim)
        features = encoder(images)

        # Global average pooling over patches -> (B, embed_dim)
        features = features.mean(dim=1)

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())
        total_samples += len(labels)

    features = np.concatenate(all_features, axis=0)[:max_samples]
    labels = np.concatenate(all_labels, axis=0)[:max_samples]

    return features, labels


def compute_tsne(
    features: np.ndarray,
    perplexity: float = 30.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute t-SNE embedding of features.

    Args:
        features: (N, embed_dim) feature vectors
        perplexity: t-SNE perplexity parameter
        max_iter: Maximum number of iterations
        random_state: Random seed for reproducibility

    Returns:
        tsne_embeddings: (N, 2) 2D embeddings
    """
    print(f"Computing t-SNE with perplexity={perplexity}, max_iter={max_iter}...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
        init='pca',
        learning_rate='auto',
    )
    embeddings = tsne.fit_transform(features)
    print("t-SNE complete!")
    return embeddings


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] = None,
    title: str = "t-SNE Visualization of Learned Features",
    figsize: tuple = (12, 10),
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot t-SNE embeddings colored by class.

    Args:
        embeddings: (N, 2) t-SNE embeddings
        labels: (N,) class labels
        class_names: List of class names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES

    fig, ax = plt.subplots(figsize=figsize)

    # Use a colormap with distinct colors
    cmap = plt.cm.get_cmap('tab10')
    num_classes = len(class_names)

    for class_idx in range(num_classes):
        mask = labels == class_idx
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[cmap(class_idx)],
            label=class_names[class_idx],
            alpha=0.6,
            s=10,
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE dimension 2", fontsize=12)
    ax.legend(loc='best', fontsize=10, markerscale=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()

    return fig


def visualize_features(
    model,
    dataloader: DataLoader,
    device: torch.device = None,
    use_target_encoder: bool = True,
    max_samples: int = 5000,
    perplexity: float = 30.0,
    save_path: str = None,
    show: bool = True,
) -> tuple[np.ndarray, np.ndarray, plt.Figure]:
    """
    Full pipeline: extract features, compute t-SNE, and visualize.

    Args:
        model: I-JEPA model
        dataloader: Data loader with images and labels
        device: Device to use
        use_target_encoder: If True, use target encoder; else context encoder
        max_samples: Maximum samples for t-SNE (more = slower but better)
        perplexity: t-SNE perplexity
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        embeddings: (N, 2) t-SNE embeddings
        labels: (N,) class labels
        fig: matplotlib Figure
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select encoder
    encoder = model.target_encoder if use_target_encoder else model.context_encoder
    encoder_name = "Target" if use_target_encoder else "Context"

    print(f"\n{'='*60}")
    print(f"t-SNE Visualization ({encoder_name} Encoder)")
    print('='*60)

    # Extract features
    features, labels = extract_features_subset(
        encoder=encoder,
        dataloader=dataloader,
        device=device,
        max_samples=max_samples,
    )
    print(f"Extracted {len(features)} feature vectors of dim {features.shape[1]}")

    # Compute t-SNE
    embeddings = compute_tsne(features, perplexity=perplexity)

    # Plot
    title = f"t-SNE of I-JEPA {encoder_name} Encoder Features (CIFAR-10)"
    fig = plot_tsne(
        embeddings=embeddings,
        labels=labels,
        title=title,
        save_path=save_path,
        show=show,
    )

    return embeddings, labels, fig
