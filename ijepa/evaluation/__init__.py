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

from ijepa.evaluation.embedding_extractor import (
    LocalIJEPAExtractor,
    HuggingFaceIJEPAExtractor,
)

from ijepa.evaluation.pca_visualization import (
    compute_pca,
    plot_pca_2d,
    plot_variance_explained,
)

from ijepa.evaluation.similarity_analysis import (
    compute_class_centroids,
    compute_centroid_similarity_matrix,
    plot_similarity_matrix,
    compare_similarity_matrices,
)

from ijepa.evaluation.nearest_neighbor import (
    NearestNeighborRetriever,
    visualize_nearest_neighbors,
)

from ijepa.evaluation.model_comparison import (
    compare_tsne,
    compare_pca,
    generate_comparison_report,
)

__all__ = [
    # Linear probe
    'evaluate_linear_probe',
    'extract_features',
    'train_linear_classifier',
    'LinearProbe',
    # Visualization
    'visualize_features',
    'compute_tsne',
    'plot_tsne',
    'extract_features_subset',
    'CIFAR10_CLASSES',
    # Embedding extractors
    'LocalIJEPAExtractor',
    'HuggingFaceIJEPAExtractor',
    # PCA
    'compute_pca',
    'plot_pca_2d',
    'plot_variance_explained',
    # Similarity
    'compute_class_centroids',
    'compute_centroid_similarity_matrix',
    'plot_similarity_matrix',
    'compare_similarity_matrices',
    # Nearest neighbor
    'NearestNeighborRetriever',
    'visualize_nearest_neighbors',
    # Model comparison
    'compare_tsne',
    'compare_pca',
    'generate_comparison_report',
]
