"""
I-JEPA Training and Evaluation Entry Point

Usage:
    python main.py --mode train
    python main.py --mode eval --checkpoint path/to/checkpoint.pt
"""

import argparse

from config import CIFAR10_CONFIG


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='I-JEPA for CIFAR-10')

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'eval'],
        default='train',
        help='Mode: train or eval'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for evaluation or resume training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Override config if needed
    config = CIFAR10_CONFIG.copy()
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    if args.mode == 'train':
        # TODO: Import and call train_ijepa
        # from ijepa.training.train import train_ijepa
        # train_ijepa(config, checkpoint_path=args.checkpoint)
        raise NotImplementedError("Training not yet implemented")

    elif args.mode == 'eval':
        if args.checkpoint is None:
            raise ValueError("Checkpoint path required for evaluation")
        # TODO: Import and call evaluate_linear_probe
        # from ijepa.evaluation.linear_probe import evaluate_linear_probe
        # evaluate_linear_probe(config, checkpoint_path=args.checkpoint)
        raise NotImplementedError("Evaluation not yet implemented")


if __name__ == '__main__':
    main()
