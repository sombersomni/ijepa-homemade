"""
Visualize I-JEPA learned representations using t-SNE.

Usage:
    PYTHONPATH=. python scripts/visualize_tsne.py --checkpoint checkpoints_300ep/checkpoint_latest.pt
"""

import argparse
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving

from ijepa.models import IJEPA
from ijepa.data import get_cifar10_loaders
from ijepa.evaluation import visualize_features


def main():
    parser = argparse.ArgumentParser(description='Visualize I-JEPA features with t-SNE')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_300ep/checkpoint_latest.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--max-samples', type=int, default=5000,
                        help='Maximum samples for t-SNE')
    parser.add_argument('--perplexity', type=float, default=30.0,
                        help='t-SNE perplexity')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save figures')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    # Create model with same config
    model = IJEPA(
        img_size=config.get('img_size', 32),
        patch_size=config.get('patch_size', 4),
        encoder_embed_dim=config.get('encoder_embed_dim', 192),
        encoder_depth=config.get('encoder_depth', 6),
        encoder_num_heads=config.get('encoder_num_heads', 6),
        predictor_embed_dim=config.get('predictor_embed_dim', 96),
        predictor_depth=config.get('predictor_depth', 4),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown') + 1}")

    # Get test data loader
    _, test_loader = get_cifar10_loaders(
        batch_size=256,
        data_dir=config.get('data_dir', './data'),
        num_workers=4,
    )

    # Visualize target encoder
    print("\n" + "="*60)
    print("Generating t-SNE visualization...")
    print("="*60)

    target_save_path = f"{args.output_dir}/tsne_target_encoder.png"
    visualize_features(
        model=model,
        dataloader=test_loader,
        device=device,
        use_target_encoder=True,
        max_samples=args.max_samples,
        perplexity=args.perplexity,
        save_path=target_save_path,
        show=False,
    )

    # Also visualize context encoder for comparison
    context_save_path = f"{args.output_dir}/tsne_context_encoder.png"
    visualize_features(
        model=model,
        dataloader=test_loader,
        device=device,
        use_target_encoder=False,
        max_samples=args.max_samples,
        perplexity=args.perplexity,
        save_path=context_save_path,
        show=False,
    )

    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"  Target encoder: {target_save_path}")
    print(f"  Context encoder: {context_save_path}")
    print("="*60)


if __name__ == "__main__":
    main()
