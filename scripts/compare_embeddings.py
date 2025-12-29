"""
Compare HuggingFace I-JEPA with locally trained model.

Usage:
    PYTHONPATH=. ~/.pyenv/versions/3.11.9/bin/python scripts/compare_embeddings.py \
        --local-checkpoint checkpoints_300ep/checkpoint_latest.pt \
        --output-dir comparison_results
"""

import argparse
import json
import os

import torch
import matplotlib
matplotlib.use('Agg')
import torchvision
import torchvision.transforms as transforms

from ijepa.models import IJEPA
from ijepa.data import get_cifar10_loaders
from ijepa.data.cifar10 import CIFAR10_MEAN, CIFAR10_STD
from ijepa.evaluation.embedding_extractor import (
    LocalIJEPAExtractor,
    HuggingFaceIJEPAExtractor,
)
from ijepa.evaluation.model_comparison import (
    compare_tsne,
    compare_pca,
    generate_comparison_report,
)
from ijepa.evaluation.similarity_analysis import (
    compute_centroid_similarity_matrix,
    compare_similarity_matrices,
)
from ijepa.evaluation.pca_visualization import (
    plot_variance_explained,
    compute_pca,
)
from ijepa.evaluation.nearest_neighbor import (
    NearestNeighborRetriever,
    visualize_nearest_neighbors,
)


def main():
    parser = argparse.ArgumentParser(description='Compare I-JEPA model embeddings')
    parser.add_argument('--local-checkpoint', type=str,
                        default='checkpoints_300ep/checkpoint_latest.pt',
                        help='Path to local model checkpoint')
    parser.add_argument('--hf-model', type=str,
                        default='facebook/ijepa_vith14_1k',
                        help='HuggingFace model identifier')
    parser.add_argument('--output-dir', type=str,
                        default='comparison_results',
                        help='Directory to save results')
    parser.add_argument('--max-samples', type=int, default=5000,
                        help='Maximum samples for embedding extraction')
    parser.add_argument('--skip-hf', action='store_true',
                        help='Skip HuggingFace model (for testing)')
    parser.add_argument('--skip-tsne', action='store_true',
                        help='Skip t-SNE (slow)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load CIFAR-10 test data
    _, test_loader = get_cifar10_loaders(batch_size=64, num_workers=4)

    # Get raw dataset for NN visualization
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    )

    # ==================== Load Local Model ====================
    print("\n" + "="*60)
    print("Loading local I-JEPA model...")
    print("="*60)

    checkpoint = torch.load(args.local_checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    local_model = IJEPA(
        img_size=config.get('img_size', 32),
        patch_size=config.get('patch_size', 4),
        encoder_embed_dim=config.get('encoder_embed_dim', 192),
        encoder_depth=config.get('encoder_depth', 6),
        encoder_num_heads=config.get('encoder_num_heads', 6),
        predictor_embed_dim=config.get('predictor_embed_dim', 96),
        predictor_depth=config.get('predictor_depth', 4),
    ).to(device)
    local_model.load_state_dict(checkpoint['model_state_dict'])
    local_model.eval()
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown') + 1}")

    local_extractor = LocalIJEPAExtractor(local_model, device, use_target_encoder=True)

    # ==================== Extract Local Embeddings ====================
    print("\nExtracting local model embeddings...")
    local_features, local_labels = local_extractor.extract_embeddings(
        test_loader, max_samples=args.max_samples
    )
    print(f"Local features shape: {local_features.shape}")

    model_features = {
        local_extractor.model_name: (local_features, local_labels)
    }

    # ==================== Load HF Model (if not skipped) ====================
    if not args.skip_hf:
        print("\n" + "="*60)
        print(f"Loading HuggingFace model: {args.hf_model}")
        print("="*60)

        hf_extractor = HuggingFaceIJEPAExtractor(
            model_id=args.hf_model,
            device=device,
            use_float16=True,
        )

        print("\nExtracting HF model embeddings (this may take a while)...")
        hf_features, hf_labels = hf_extractor.extract_embeddings(
            test_loader, max_samples=args.max_samples
        )
        print(f"HF features shape: {hf_features.shape}")

        model_features[hf_extractor.model_name] = (hf_features, hf_labels)

    # ==================== Generate Visualizations ====================
    print("\n" + "="*60)
    print("Generating comparison visualizations...")
    print("="*60)

    # 1. t-SNE comparison (if not skipped)
    if not args.skip_tsne:
        print("\n1. t-SNE comparison...")
        compare_tsne(
            model_features,
            save_path=os.path.join(args.output_dir, 'tsne_comparison.png'),
        )
    else:
        print("\n1. Skipping t-SNE (use --skip-tsne=False to enable)")

    # 2. PCA comparison
    print("\n2. PCA comparison...")
    compare_pca(
        model_features,
        save_path=os.path.join(args.output_dir, 'pca_comparison.png'),
    )

    # 3. Variance explained
    print("\n3. Variance explained analysis...")
    variance_ratios = {}
    for name, (features, _) in model_features.items():
        _, var, _ = compute_pca(features, n_components=50)
        variance_ratios[name] = var

    plot_variance_explained(
        variance_ratios,
        save_path=os.path.join(args.output_dir, 'variance_explained.png'),
    )

    # 4. Similarity matrices
    print("\n4. Class centroid similarity...")
    similarities = {}
    for name, (features, labels) in model_features.items():
        similarities[name] = compute_centroid_similarity_matrix(features, labels)

    compare_similarity_matrices(
        similarities,
        save_path=os.path.join(args.output_dir, 'similarity_matrices.png'),
    )

    # 5. Nearest neighbor examples
    print("\n5. Nearest neighbor retrieval examples...")
    for name, (features, labels) in model_features.items():
        retriever = NearestNeighborRetriever(features, labels, metric='cosine')

        # Visualize a few examples from different classes
        safe_name = name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        # Use valid query indices based on dataset size
        n_samples = len(features)
        query_indices = [i for i in [0, 100, 500, 1000, 2000] if i < n_samples]
        for query_idx in query_indices:
            indices, distances, neighbor_labels = retriever.find_neighbors(query_idx, k=5)
            visualize_nearest_neighbors(
                query_idx=query_idx,
                neighbor_indices=indices,
                dataset=test_dataset,
                query_label=labels[query_idx],
                neighbor_labels=neighbor_labels,
                distances=distances,
                model_name=name,
                save_path=os.path.join(args.output_dir, f'nn_{safe_name}_query{query_idx}.png'),
            )

    # 6. Generate summary report
    print("\n6. Generating summary report...")
    results = generate_comparison_report(model_features, args.output_dir)

    # Save results as JSON
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print(f"Results saved to {args.output_dir}/")
    print("="*60)
    print("\nGenerated files:")
    for f in sorted(os.listdir(args.output_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
