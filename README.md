# I-JEPA: Image-based Joint-Embedding Predictive Architecture

A from-scratch PyTorch implementation of I-JEPA for self-supervised visual representation learning, trained on CIFAR-10.

## What is I-JEPA?

I-JEPA (Image-based Joint-Embedding Predictive Architecture) is a self-supervised learning method that learns visual representations by predicting the representations of image regions from context, rather than predicting pixels directly.

### Key Insight: Predict in Representation Space, Not Pixel Space

Unlike masked autoencoders (MAE) that reconstruct raw pixels, I-JEPA predicts **abstract representations**. This forces the model to learn semantic features rather than low-level details like textures and colors.

```
Traditional Masked Autoencoder:
  Image → Encoder → Decoder → Reconstructed Pixels
  Loss = ||pixels - reconstructed_pixels||²

I-JEPA:
  Image → Encoder → Predictor → Predicted Representations
  Loss = ||target_representations - predicted_representations||²
```

## Architecture Overview

I-JEPA consists of three main components:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           I-JEPA ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────────┘

                              Input Image
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
            ┌──────────────┐            ┌──────────────┐
            │   Context    │            │    Target    │
            │    Mask      │            │    Masks     │
            │  (visible)   │            │   (hidden)   │
            └──────┬───────┘            └──────┬───────┘
                   │                           │
                   ▼                           ▼
         ┌─────────────────┐          ┌─────────────────┐
         │     Context     │          │     Target      │
         │     Encoder     │          │     Encoder     │
         │   (trainable)   │          │   (EMA copy)    │
         └────────┬────────┘          └────────┬────────┘
                  │                            │
                  │    Context                 │    Target
                  │    Representations         │    Representations
                  │                            │
                  ▼                            │
    ┌─────────────────────────┐               │
    │  Concatenate with       │               │
    │  Mask Tokens +          │               │
    │  Positional Embeddings  │               │
    └────────────┬────────────┘               │
                 │                            │
                 ▼                            │
        ┌─────────────────┐                   │
        │    Predictor    │                   │
        │  (trainable)    │                   │
        └────────┬────────┘                   │
                 │                            │
                 │    Predicted               │
                 │    Representations         │
                 │                            │
                 └──────────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │     L2 Loss         │
                    │  (in repr. space)   │
                    └─────────────────────┘
```

## The Three Encoders Explained

### 1. Context Encoder (Trainable via Gradient Descent)
- Processes only the **visible patches** (context region)
- Standard Vision Transformer (ViT) architecture
- Parameters updated through backpropagation

### 2. Target Encoder (Updated via EMA)
- Processes the **full image** to produce target representations
- **Not trained with gradients** - updated as Exponential Moving Average of context encoder
- EMA momentum: 0.996 → 1.0 (increases during training)
- Formula: `target_params = momentum * target_params + (1 - momentum) * context_params`

### 3. Predictor (Trainable via Gradient Descent)
- Takes context representations + learnable mask tokens
- Predicts representations for the masked (target) regions
- Smaller and narrower than the encoders

## The Mask Token Mechanism: Key Innovation

This is what makes I-JEPA fundamentally different from reconstruction-based methods:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PREDICTOR INPUT CONSTRUCTION                        │
└─────────────────────────────────────────────────────────────────────────┘

Step 1: Get context representations from Context Encoder

        Context patches (e.g., 48 patches visible):
        [c₁, c₂, c₃, ..., c₄₈]  ← each is a 192-dim vector

Step 2: Create mask tokens for target positions

        Target positions (e.g., 16 patches to predict):
        [M, M, M, ..., M]  ← M is a SHARED learnable 192-dim vector
                            (same M repeated, but with different positions)

Step 3: Add positional embeddings to mask tokens

        [M + pos₅, M + pos₁₂, M + pos₂₃, ...]

        Now each mask token knows WHERE it needs to predict

Step 4: Concatenate context + positioned mask tokens

        Predictor Input = [c₁, c₂, ..., c₄₈, M+pos₅, M+pos₁₂, ...]
                          \_____context_____/ \____mask tokens____/

Step 5: Predictor transforms this into predictions

        The predictor uses self-attention to let mask tokens
        "query" the context tokens and predict target representations
```

### Why This Works (No Reconstruction Needed)

1. **Mask tokens are queries**: They ask "what should the representation be at position X?"
2. **Context provides answers**: Through self-attention, mask tokens gather information from visible context
3. **Positional embeddings are crucial**: Without them, the model wouldn't know which region to predict
4. **Shared mask token**: Using the same learnable vector (with different positions) forces the model to use position info

## Loss Function

The loss is simple but powerful:

```python
def ijepa_loss(predictions, targets):
    """
    L2 distance in representation space.

    predictions: List of predicted representations for each target block
    targets: List of target representations (from target encoder, no gradients)
    """
    total_loss = 0
    for pred, tgt in zip(predictions, targets):
        # Sum over embedding dimension, average over patches and batch
        patch_loss = (pred - tgt).pow(2).sum(dim=-1)  # L2 per patch
        total_loss += patch_loss.mean()
    return total_loss / len(predictions)
```

**Key properties:**
- Computed in **representation space**, not pixel space
- Target encoder is **frozen** (no gradients flow through it)
- Gradients only flow through context encoder and predictor
- Forces the model to learn **semantic** features, not pixel-level details

## Multi-Block Masking Strategy

I-JEPA uses a specific masking strategy with multiple target blocks:

```
┌────────────────────────────────────────┐
│  8x8 patch grid (CIFAR-10: 32x32, 4x4 patches) │
├────────────────────────────────────────┤
│                                        │
│    ████░░░░    ░░░░░░░░    ████ = Target Block 1 (predict)
│    ████░░░░    ░░░░████    ░░░░ = Context (visible)
│    ░░░░░░░░    ░░░░████
│    ░░░░░░░░    ░░░░░░░░    Target blocks: 15-25% of image each
│                            Context: 75-95% of image (minus targets)
│    4 target blocks total
│                                        │
└────────────────────────────────────────┘
```

**Why multiple blocks?**
- Harder prediction task → better representations
- Different scales and positions → more robust features
- Targets can overlap with context mask (context sees subset of what target encoder sees)

## Installation

```bash
# Clone the repository
git clone https://github.com/sombersomni/ijepa-homemade.git
cd ijepa-homemade

# Install dependencies (Python 3.11+ recommended)
pip install torch torchvision numpy tqdm pytest
```

## Quick Start: Training on CIFAR-10

### Run Training with Linear Probe Evaluation

```bash
# Set PYTHONPATH and run training
PYTHONPATH=. python scripts/train_cifar10.py
```

Or modify the config for a quick test:

```python
# scripts/train_cifar10.py
config = {
    'img_size': 32,
    'patch_size': 4,           # 8x8 = 64 patches per image
    'encoder_embed_dim': 192,
    'encoder_depth': 6,        # 6 transformer blocks
    'encoder_num_heads': 6,
    'predictor_embed_dim': 96, # Narrower than encoder
    'predictor_depth': 4,
    'batch_size': 256,
    'epochs': 100,             # Reduce for quick test (e.g., 10)
    'lr': 1e-3,
    'warmup_epochs': 10,
    'ema_momentum_start': 0.996,
    'ema_momentum_end': 1.0,
}
```

### Expected Training Behavior

```
Epoch   Loss    Notes
─────   ────    ─────
1       ~250    Initial random predictions
10      ~70     Learning basic structure
50      ~45     Capturing semantic features
100     ~37     Converged representations
```

The loss measures L2 distance in representation space - lower means the predictor can better anticipate target representations from context.

## What to Expect After Training

### Linear Probe Results (CIFAR-10)

After training, we evaluate by:
1. Freezing the target encoder
2. Extracting features (global average pooling over patches)
3. Training a linear classifier on frozen features

**Actual results with this implementation (100 epochs):**
- **Target Encoder: 70.10% test accuracy**
- **Context Encoder: 70.33% test accuracy**
- For comparison: supervised ViT on CIFAR-10 achieves ~85%+

### Why Accuracy Isn't Everything

The goal of I-JEPA is to learn **transferable representations**, not to achieve high accuracy on the pretraining dataset. Benefits emerge when:
- Fine-tuning on downstream tasks
- Using larger models and datasets
- Training for longer (original paper: 600 epochs on ImageNet)

### Inspecting Learned Representations

After training, you can extract features:

```python
import torch
from ijepa.models import IJEPA

# Load trained model
checkpoint = torch.load('checkpoints/checkpoint_latest.pt')
model = IJEPA(img_size=32, patch_size=4)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Extract features
with torch.no_grad():
    image = torch.randn(1, 3, 32, 32)  # Your image here
    features = model.target_encoder(image)  # (1, 64, 192)
    pooled = features.mean(dim=1)  # (1, 192) - global feature
```

## Project Structure

```
ijepa-homemade/
├── ijepa/
│   ├── models/
│   │   ├── vit.py          # Vision Transformer components
│   │   ├── predictor.py    # Predictor with mask tokens
│   │   └── ijepa.py        # Full I-JEPA model
│   ├── data/
│   │   ├── masking.py      # Multi-block mask generation
│   │   └── cifar10.py      # CIFAR-10 data loading
│   ├── training/
│   │   ├── train.py        # Training loop
│   │   ├── scheduler.py    # LR and momentum schedules
│   │   └── ema.py          # EMA update utility
│   └── evaluation/
│       └── linear_probe.py # Linear evaluation protocol
├── tests/                   # Unit tests (166 tests)
├── scripts/
│   └── train_cifar10.py    # Training script
└── checkpoints/            # Saved models (gitignored)
```

## Running Tests

```bash
# Run all tests
PYTHONPATH=. python -m pytest tests/ -v

# Run specific test file
PYTHONPATH=. python -m pytest tests/test_ijepa.py -v
```

## Key Differences from Other Self-Supervised Methods

| Method | Predicts | Learns |
|--------|----------|--------|
| MAE | Pixels | Low-level features (textures, edges) |
| SimCLR | Same image (contrastive) | Invariances (augmentation-dependent) |
| DINO | Class tokens | Global features |
| **I-JEPA** | **Representations** | **Semantic features without invariances** |

I-JEPA's advantage: learns semantic features without requiring hand-crafted augmentations or pixel-level reconstruction.

## References

- [I-JEPA Paper](https://arxiv.org/abs/2301.08243): "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (Assran et al., 2023)
- [Original Implementation](https://github.com/facebookresearch/ijepa)

## License

MIT License
