"""
I-JEPA Training on CIFAR-10 with Linear Probe Evaluation
"""

import torch
from ijepa.training.train import train_ijepa
from ijepa.data import get_cifar10_loaders
from ijepa.evaluation.linear_probe import evaluate_linear_probe

if __name__ == "__main__":
    # Full training config
    config = {
        'img_size': 32,
        'patch_size': 4,
        'encoder_embed_dim': 192,
        'encoder_depth': 6,
        'encoder_num_heads': 6,
        'predictor_embed_dim': 96,
        'predictor_depth': 4,
        'batch_size': 256,
        'epochs': 300,
        'lr': 1e-3,
        'min_lr': 1e-5,
        'weight_decay': 0.05,
        'warmup_epochs': 30,
        'ema_momentum_start': 0.996,
        'ema_momentum_end': 1.0,
        'num_targets': 4,
        'target_scale': (0.15, 0.25),
        'context_scale': (0.75, 0.95),
        'data_dir': './data',
        'save_dir': './checkpoints_300ep',
    }

    print("=" * 60)
    print("I-JEPA Full Training - CIFAR-10")
    print("=" * 60)
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Encoder: depth={config['encoder_depth']}, dim={config['encoder_embed_dim']}")
    print(f"Predictor: depth={config['predictor_depth']}, dim={config['predictor_embed_dim']}")
    print("=" * 60)

    # Train I-JEPA
    model = train_ijepa(config)

    print("\n" + "=" * 60)
    print("Training complete! Starting linear probe evaluation...")
    print("=" * 60)

    # Linear probe evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data loaders for evaluation (no augmentation, just normalize)
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=256,
        data_dir=config['data_dir'],
        num_workers=4,
    )

    # Evaluate target encoder (main result)
    target_results = evaluate_linear_probe(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        probe_epochs=100,
        probe_lr=0.1,
        use_target_encoder=True,
    )

    # Also evaluate context encoder for comparison
    context_results = evaluate_linear_probe(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        probe_epochs=100,
        probe_lr=0.1,
        use_target_encoder=False,
    )

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Target Encoder:  {target_results['test_acc']:.2f}% test accuracy")
    print(f"Context Encoder: {context_results['test_acc']:.2f}% test accuracy")
    print("=" * 60)
