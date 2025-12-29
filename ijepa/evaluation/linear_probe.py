"""
Linear Probe Evaluation for I-JEPA

Evaluates learned representations by training a linear classifier
on frozen encoder features.
"""

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ijepa.models import ViTEncoder


class LinearProbe(nn.Module):
    """Linear classifier for evaluation."""

    def __init__(self, embed_dim: int, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@torch.no_grad()
def extract_features(
    encoder: ViTEncoder,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features from encoder for all samples in dataloader.

    Features are computed by average pooling all patch representations.

    Args:
        encoder: Frozen target encoder
        dataloader: Data loader with images and labels
        device: Device to run on

    Returns:
        features: (N, embed_dim) feature vectors
        labels: (N,) class labels
    """
    encoder.eval()
    all_features = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)

        # Get encoder output (B, num_patches, embed_dim)
        features = encoder(images)

        # Global average pooling over patches -> (B, embed_dim)
        features = features.mean(dim=1)

        all_features.append(features.cpu())
        all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


def train_linear_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    embed_dim: int,
    num_classes: int,
    epochs: int = 100,
    lr: float = 0.1,
    batch_size: int = 256,
    device: torch.device = None,
) -> nn.Linear:
    """
    Train a linear classifier on extracted features.

    Args:
        features: (N, embed_dim) training features
        labels: (N,) training labels
        embed_dim: Feature dimension
        num_classes: Number of output classes
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        device: Device to train on

    Returns:
        Trained linear classifier
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create probe and optimizer
    probe = LinearProbe(embed_dim, num_classes).to(device)
    optimizer = SGD(probe.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    criterion = nn.CrossEntropyLoss()

    # Create data loader
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        probe.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for feat, lbl in loader:
            feat, lbl = feat.to(device), lbl.to(device)

            optimizer.zero_grad()
            logits = probe(feat)
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == lbl).sum().item()
            total += lbl.size(0)

        if (epoch + 1) % 20 == 0:
            acc = 100.0 * correct / total
            print(f"  Probe Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}, Acc: {acc:.2f}%")

    return probe


@torch.no_grad()
def evaluate_classifier(
    probe: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """
    Evaluate classifier accuracy.

    Args:
        probe: Trained linear probe
        features: (N, embed_dim) features
        labels: (N,) labels
        device: Device to use
        batch_size: Batch size

    Returns:
        Accuracy (0-100%)
    """
    probe.eval()
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    for feat, lbl in loader:
        feat, lbl = feat.to(device), lbl.to(device)
        logits = probe(feat)
        preds = logits.argmax(dim=1)
        correct += (preds == lbl).sum().item()
        total += lbl.size(0)

    return 100.0 * correct / total


def evaluate_linear_probe(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device = None,
    probe_epochs: int = 100,
    probe_lr: float = 0.1,
    use_target_encoder: bool = True,
) -> dict:
    """
    Full linear probe evaluation pipeline.

    1. Extract features from train and test sets using frozen encoder
    2. Train linear classifier on train features
    3. Evaluate on test features

    Args:
        model: I-JEPA model
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to use
        probe_epochs: Epochs for linear probe training
        probe_lr: Learning rate for probe
        use_target_encoder: If True, use target encoder; else context encoder

    Returns:
        Dictionary with train_acc, test_acc
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select encoder
    encoder = model.target_encoder if use_target_encoder else model.context_encoder
    encoder_name = "target" if use_target_encoder else "context"

    print(f"\n{'='*60}")
    print(f"Linear Probe Evaluation ({encoder_name} encoder)")
    print('='*60)

    # Extract features
    print("\nExtracting training features...")
    train_features, train_labels = extract_features(encoder, train_loader, device)

    print("Extracting test features...")
    test_features, test_labels = extract_features(encoder, test_loader, device)

    print(f"\nTrain features: {train_features.shape}")
    print(f"Test features: {test_features.shape}")

    # Train linear probe
    embed_dim = train_features.shape[1]
    num_classes = len(torch.unique(train_labels))

    print(f"\nTraining linear probe ({probe_epochs} epochs, lr={probe_lr})...")
    probe = train_linear_classifier(
        features=train_features,
        labels=train_labels,
        embed_dim=embed_dim,
        num_classes=num_classes,
        epochs=probe_epochs,
        lr=probe_lr,
        device=device,
    )

    # Evaluate
    train_acc = evaluate_classifier(probe, train_features, train_labels, device)
    test_acc = evaluate_classifier(probe, test_features, test_labels, device)

    print(f"\n{'='*60}")
    print(f"Linear Probe Results ({encoder_name} encoder):")
    print(f"  Train Accuracy: {train_acc:.2f}%")
    print(f"  Test Accuracy:  {test_acc:.2f}%")
    print('='*60)

    return {
        'encoder': encoder_name,
        'train_acc': train_acc,
        'test_acc': test_acc,
    }
