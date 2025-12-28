"""
I-JEPA Configuration for CIFAR-10

All hyperparameters for training I-JEPA on CIFAR-10 dataset.
Based on the design document specifications.
"""

CIFAR10_CONFIG = {
    # Data
    'image_size': 32,
    'patch_size': 4,
    'num_patches': 64,  # 8 x 8
    'in_channels': 3,

    # Context Encoder
    'encoder_embed_dim': 192,
    'encoder_depth': 6,
    'encoder_num_heads': 6,
    'mlp_ratio': 4.0,
    'dropout': 0.0,  # No dropout for pretraining

    # Predictor
    'predictor_embed_dim': 96,  # Narrow bottleneck
    'predictor_depth': 4,
    'predictor_num_heads': 6,

    # Masking (adjusted for 8x8 grid)
    'num_target_blocks': 4,
    'target_scale': (0.15, 0.25),  # Slightly larger for small grid
    'target_aspect_ratio': (0.75, 1.5),
    'context_scale': (0.75, 0.95),  # Slightly smaller to ensure non-trivial task

    # Training
    'batch_size': 256,
    'base_lr': 1e-3,
    'min_lr': 1e-5,
    'weight_decay': 0.05,
    'epochs': 100,
    'warmup_epochs': 10,

    # EMA
    'ema_momentum_start': 0.996,
    'ema_momentum_end': 1.0,

    # Evaluation
    'num_classes': 10,

    # Paths
    'data_dir': './data',
    'checkpoint_dir': './checkpoints',
}
