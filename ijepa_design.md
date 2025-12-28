# I-JEPA Implementation Design Document

## Overview

This document provides a complete specification for implementing I-JEPA (Image-based Joint-Embedding Predictive Architecture) from scratch, trainable on CIFAR-10 for educational purposes.

**Core Idea**: From a single context block of an image, predict the representations of various target blocks in the same image. Predictions happen in representation space, not pixel space.

---

## Architecture Components

### 1. Vision Transformer (ViT) Base

We need a simplified ViT that works well with CIFAR-10's 32×32 resolution.

```
Config for CIFAR-10:
- Image size: 32×32
- Patch size: 4×4 (gives us 8×8 = 64 patches)
- Embedding dimension: 192 (small model)
- Number of layers: 6
- Number of attention heads: 6
- MLP ratio: 4.0
- Dropout: 0.0 (for pretraining)
```

**Important**: I-JEPA does NOT use a [CLS] token. All representations are patch-level.

#### ViT Components:

```python
class PatchEmbed:
    """Convert image to patch embeddings"""
    # Input: (B, 3, 32, 32)
    # Output: (B, num_patches, embed_dim) = (B, 64, 192)
    # Implementation: Conv2d with kernel_size=patch_size, stride=patch_size

class PositionalEmbedding:
    """Learnable 2D positional embeddings"""
    # Shape: (1, num_patches, embed_dim) = (1, 64, 192)
    # These are ADDED to patch embeddings
    # Use nn.Parameter with normal initialization

class TransformerBlock:
    """Standard transformer block"""
    # LayerNorm -> MultiHeadAttention -> Residual
    # LayerNorm -> MLP -> Residual
    
class ViTEncoder:
    """Full encoder (used for context and target encoders)"""
    # patch_embed -> add positional embeddings -> transformer blocks -> final LayerNorm
```

---

### 2. The Three Networks

#### 2.1 Context Encoder (fθ)
- Standard ViT encoder
- **Only processes visible context patches** (not the full image)
- Trainable via gradient descent

#### 2.2 Target Encoder (f̄θ)  
- **Identical architecture** to context encoder
- **NOT trained with gradients**
- Updated via Exponential Moving Average (EMA) of context encoder:
  ```
  θ̄ = momentum * θ̄ + (1 - momentum) * θ
  ```
- Momentum starts at 0.996 and linearly increases to 1.0 during training

#### 2.3 Predictor (gφ)
- **Narrow/lightweight ViT** (bottleneck design)
- Embedding dimension: 96 (half of encoder's 192)
- Number of layers: 4 (fewer than encoder)
- Same number of attention heads as encoder: 6
- Takes context encoder output + mask tokens as input
- Outputs predictions for target patch representations

---

### 3. Mask Tokens (THE KEY MECHANISM)

This is the critical part that enables location-aware prediction.

#### How Mask Tokens Work:

```python
class MaskToken:
    """
    The mask token mechanism has TWO components:
    1. A SINGLE shared learnable vector (the "mask embedding")
    2. Positional embeddings (shared with or separate from encoder)
    
    For each target position we want to predict:
    mask_token_at_position_j = shared_mask_embedding + positional_embedding[j]
    """
    
    def __init__(self, embed_dim):
        # Single learnable vector, shared across ALL positions
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Initialize with small random values
        nn.init.normal_(self.mask_embedding, std=0.02)
    
    def create_mask_tokens(self, target_positions, positional_embeddings):
        """
        Args:
            target_positions: list of patch indices we want to predict [j1, j2, ...]
            positional_embeddings: (1, num_patches, embed_dim) tensor
        
        Returns:
            mask_tokens: (1, num_target_patches, embed_dim)
        """
        # Get positional embeddings for target positions
        pos_embeds = positional_embeddings[:, target_positions, :]  # (1, num_targets, embed_dim)
        
        # Add the shared mask embedding to each position's embedding
        mask_tokens = self.mask_embedding + pos_embeds  # Broadcasting: (1,1,D) + (1,T,D) = (1,T,D)
        
        return mask_tokens
```

#### Why This Design?

The predictor needs to know:
1. **What to predict**: The shared mask embedding tells the network "this is a position I need to generate a prediction for"
2. **Where to predict**: The positional embedding tells the network "this is the spatial location in the image"

The predictor receives:
- Context encoder outputs (with their positions)
- Mask tokens (with target positions)

It can attend between them and learn to predict what should be at the target locations based on the context.

---

### 4. Predictor Architecture Detail

```python
class Predictor:
    """
    Narrow ViT that predicts target representations from context + mask tokens
    """
    def __init__(self, context_dim, predictor_dim, num_layers, num_heads):
        # Project from context encoder dimension to narrow predictor dimension
        self.input_proj = nn.Linear(context_dim, predictor_dim)
        
        # Mask token (the shared learnable embedding)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Predictor's own positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, num_heads, mlp_ratio=4.0)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(predictor_dim)
        
        # Project back to context encoder dimension for loss computation
        self.output_proj = nn.Linear(predictor_dim, context_dim)
    
    def forward(self, context_output, context_positions, target_positions):
        """
        Args:
            context_output: (B, num_context_patches, context_dim) from context encoder
            context_positions: list of patch indices that are in context
            target_positions: list of patch indices we want to predict
        
        Returns:
            predictions: (B, num_target_patches, context_dim)
        """
        B = context_output.shape[0]
        
        # Project context to predictor dimension
        context_tokens = self.input_proj(context_output)  # (B, C, predictor_dim)
        
        # Add positional embeddings to context tokens
        context_pos = self.pos_embed[:, context_positions, :]  # (1, C, predictor_dim)
        context_tokens = context_tokens + context_pos
        
        # Create mask tokens for target positions
        target_pos = self.pos_embed[:, target_positions, :]  # (1, T, predictor_dim)
        mask_tokens = self.mask_token + target_pos  # (1, T, predictor_dim)
        mask_tokens = mask_tokens.expand(B, -1, -1)  # (B, T, predictor_dim)
        
        # Concatenate: [context_tokens, mask_tokens]
        x = torch.cat([context_tokens, mask_tokens], dim=1)  # (B, C+T, predictor_dim)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        # Extract only the predictions (the mask token positions)
        num_context = context_tokens.shape[1]
        predictions = x[:, num_context:, :]  # (B, T, predictor_dim)
        
        # Project back to context encoder dimension
        predictions = self.output_proj(predictions)  # (B, T, context_dim)
        
        return predictions
```

---

### 5. Masking Strategy (Multi-Block)

This is crucial for learning semantic representations.

#### Target Blocks:
- Sample M=4 (possibly overlapping) blocks
- Scale range: (0.15, 0.2) of image area
- Aspect ratio range: (0.75, 1.5)

#### Context Block:
- Sample 1 large block
- Scale range: (0.85, 1.0) of image area  
- Aspect ratio: 1.0 (square)
- **Remove overlap**: After sampling, remove any context patches that overlap with target blocks

```python
class MultiBlockMaskGenerator:
    """
    Generate context and target masks for I-JEPA
    """
    def __init__(
        self,
        input_size=(8, 8),  # Number of patches in each dimension
        num_targets=4,
        target_scale=(0.15, 0.2),
        target_aspect_ratio=(0.75, 1.5),
        context_scale=(0.85, 1.0),
    ):
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_targets = num_targets
        self.target_scale = target_scale
        self.target_aspect_ratio = target_aspect_ratio
        self.context_scale = context_scale
    
    def sample_block(self, scale_range, aspect_ratio_range):
        """
        Sample a rectangular block of patches
        
        Returns: set of patch indices
        """
        # Sample scale (fraction of total patches)
        scale = random.uniform(*scale_range)
        num_patches_in_block = int(self.num_patches * scale)
        
        # Sample aspect ratio
        aspect_ratio = random.uniform(*aspect_ratio_range)
        
        # Calculate block dimensions
        # height * width = num_patches_in_block
        # width / height = aspect_ratio
        # Therefore: height = sqrt(num_patches / aspect_ratio)
        block_height = int(round(math.sqrt(num_patches_in_block / aspect_ratio)))
        block_width = int(round(block_height * aspect_ratio))
        
        # Clamp to valid range
        block_height = max(1, min(block_height, self.height))
        block_width = max(1, min(block_width, self.width))
        
        # Sample top-left corner
        top = random.randint(0, self.height - block_height)
        left = random.randint(0, self.width - block_width)
        
        # Get patch indices in the block
        indices = set()
        for i in range(top, top + block_height):
            for j in range(left, left + block_width):
                indices.add(i * self.width + j)
        
        return indices
    
    def __call__(self):
        """
        Generate masks for one sample
        
        Returns:
            context_indices: list of patch indices for context
            target_indices_list: list of M sets, each containing patch indices for one target block
        """
        # Sample target blocks
        target_indices_list = []
        all_target_indices = set()
        for _ in range(self.num_targets):
            target_block = self.sample_block(self.target_scale, self.target_aspect_ratio)
            target_indices_list.append(target_block)
            all_target_indices.update(target_block)
        
        # Sample context block (large, square)
        context_block = self.sample_block(self.context_scale, (1.0, 1.0))
        
        # Remove target patches from context (ensure non-trivial prediction)
        context_indices = context_block - all_target_indices
        
        # Convert to sorted lists
        context_indices = sorted(list(context_indices))
        target_indices_list = [sorted(list(t)) for t in target_indices_list]
        
        return context_indices, target_indices_list
```

---

### 6. Forward Pass Flow

```
Input Image (B, 3, 32, 32)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Patch Embedding: (B, 64, 192)                                  │
│ (Full image converted to patch embeddings)                     │
└───────────────────────────────────────────────────────────────┘
        │
        ├──────────────────────────────┐
        ▼                              ▼
┌─────────────────────┐      ┌─────────────────────┐
│   Context Branch    │      │   Target Branch     │
├─────────────────────┤      ├─────────────────────┤
│ Select context      │      │ Select target       │
│ patches only        │      │ patches only        │
│ (B, ~50, 192)       │      │ (B, 64, 192)        │
├─────────────────────┤      ├─────────────────────┤
│ Context Encoder fθ  │      │ Target Encoder f̄θ  │
│ (trainable)         │      │ (EMA, no gradient)  │
├─────────────────────┤      ├─────────────────────┤
│ Output: sx          │      │ Output: sy          │
│ (B, ~50, 192)       │      │ (B, 64, 192)        │
└─────────────────────┘      └─────────────────────┘
        │                              │
        │                              ▼
        │                    ┌─────────────────────┐
        │                    │ Extract target      │
        │                    │ block patches       │
        │                    │ sy(1)...sy(M)       │
        │                    │ (B, ~12, 192) each  │
        │                    └─────────────────────┘
        │                              │
        ▼                              │
┌─────────────────────────────────┐    │
│         Predictor gφ            │    │
├─────────────────────────────────┤    │
│ Input:                          │    │
│  - context_output (B, ~50, 192) │    │
│  - context_positions [indices]  │    │
│  - target_positions [indices]   │    │
├─────────────────────────────────┤    │
│ Creates mask tokens internally: │    │
│  mask_token + pos_embed[target] │    │
├─────────────────────────────────┤    │
│ Output: ŝy                      │    │
│ (B, ~12, 192) per target block  │    │
└─────────────────────────────────┘    │
        │                              │
        └──────────────┬───────────────┘
                       ▼
              ┌─────────────────┐
              │   L2 Loss       │
              │ ||ŝy - sy||²    │
              │ (per patch,     │
              │  averaged)      │
              └─────────────────┘
```

---

### 7. Loss Function

```python
def ijepa_loss(predictions, targets):
    """
    Simple L2 loss between predicted and target representations
    
    Args:
        predictions: list of M tensors, each (B, num_patches_in_block, embed_dim)
        targets: list of M tensors, each (B, num_patches_in_block, embed_dim)
    
    Returns:
        scalar loss
    """
    total_loss = 0.0
    total_patches = 0
    
    for pred, tgt in zip(predictions, targets):
        # L2 loss per patch
        loss = (pred - tgt).pow(2).sum(dim=-1)  # (B, num_patches)
        total_loss += loss.sum()
        total_patches += loss.numel()
    
    return total_loss / total_patches
```

---

### 8. Training Loop Pseudocode

```python
def train_ijepa():
    # Initialize models
    context_encoder = ViTEncoder(embed_dim=192, depth=6, num_heads=6)
    target_encoder = copy.deepcopy(context_encoder)  # Start identical
    target_encoder.requires_grad_(False)  # No gradients for EMA
    
    predictor = Predictor(
        context_dim=192,
        predictor_dim=96,
        num_layers=4,
        num_heads=6
    )
    
    mask_generator = MultiBlockMaskGenerator()
    
    optimizer = AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=1e-3,
        weight_decay=0.05
    )
    
    # Training loop
    for epoch in range(num_epochs):
        for images, _ in dataloader:  # Labels ignored for self-supervised
            
            # Generate masks (per-sample)
            batch_context_indices = []
            batch_target_indices = []
            for _ in range(batch_size):
                ctx_idx, tgt_idx_list = mask_generator()
                batch_context_indices.append(ctx_idx)
                batch_target_indices.append(tgt_idx_list)
            
            # Embed all patches
            patch_embeddings = patch_embed(images)  # (B, 64, 192)
            
            # === Target Branch (no gradients) ===
            with torch.no_grad():
                target_output = target_encoder(patch_embeddings)  # (B, 64, 192)
                
                # Extract target block representations
                targets = []
                for m in range(num_target_blocks):
                    # Gather patches for m-th target block
                    target_patches = extract_patches(
                        target_output, 
                        [tgt[m] for tgt in batch_target_indices]
                    )
                    targets.append(target_patches)
            
            # === Context Branch (with gradients) ===
            # Extract only context patches before encoding (efficiency)
            context_patches = extract_patches(patch_embeddings, batch_context_indices)
            context_output = context_encoder(context_patches, batch_context_indices)
            
            # === Predictor ===
            predictions = []
            for m in range(num_target_blocks):
                target_positions = [tgt[m] for tgt in batch_target_indices]
                pred = predictor(context_output, batch_context_indices, target_positions)
                predictions.append(pred)
            
            # === Loss ===
            loss = ijepa_loss(predictions, targets)
            
            # === Backward ===
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # === EMA Update ===
            momentum = get_momentum_schedule(current_step)  # 0.996 -> 1.0
            update_ema(target_encoder, context_encoder, momentum)
    
    return target_encoder  # Use this for downstream tasks

def update_ema(target, source, momentum):
    """Exponential moving average update"""
    with torch.no_grad():
        for param_t, param_s in zip(target.parameters(), source.parameters()):
            param_t.data = momentum * param_t.data + (1 - momentum) * param_s.data
```

---

### 9. CIFAR-10 Specific Considerations

#### Patch Grid:
- 32×32 image with 4×4 patches = 8×8 = 64 patches
- This is much smaller than ImageNet (224/16 = 14×14 = 196 patches)

#### Adjusted Hyperparameters:
```python
CIFAR10_CONFIG = {
    # Data
    'image_size': 32,
    'patch_size': 4,
    'num_patches': 64,  # 8 x 8
    
    # Context Encoder
    'encoder_embed_dim': 192,
    'encoder_depth': 6,
    'encoder_num_heads': 6,
    
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
}
```

---

### 10. Evaluation

After pretraining, evaluate learned representations:

#### Linear Probe:
1. Freeze the target encoder
2. Average pool all patch representations to get image representation
3. Train a linear classifier on top

```python
def evaluate_linear_probe(target_encoder, train_loader, test_loader):
    target_encoder.eval()
    
    # Extract features
    def get_features(loader):
        features, labels = [], []
        with torch.no_grad():
            for images, targets in loader:
                patch_embeds = patch_embed(images)
                output = target_encoder(patch_embeds)
                # Average pool
                feature = output.mean(dim=1)  # (B, embed_dim)
                features.append(feature)
                labels.append(targets)
        return torch.cat(features), torch.cat(labels)
    
    train_features, train_labels = get_features(train_loader)
    test_features, test_labels = get_features(test_loader)
    
    # Train linear classifier
    classifier = nn.Linear(embed_dim, num_classes)
    # ... train for a few epochs ...
    
    # Evaluate
    accuracy = compute_accuracy(classifier, test_features, test_labels)
    return accuracy
```

---

### 11. File Structure

```
ijepa/
├── models/
│   ├── __init__.py
│   ├── vit.py           # PatchEmbed, TransformerBlock, ViTEncoder
│   ├── predictor.py     # Predictor with mask tokens
│   └── ijepa.py         # Full I-JEPA model combining all components
├── data/
│   ├── __init__.py
│   ├── masking.py       # MultiBlockMaskGenerator
│   └── cifar10.py       # DataLoader setup
├── training/
│   ├── __init__.py
│   ├── train.py         # Training loop
│   ├── scheduler.py     # LR and momentum schedulers
│   └── ema.py           # EMA update utilities
├── evaluation/
│   ├── __init__.py
│   └── linear_probe.py  # Linear evaluation
├── config.py            # All hyperparameters
└── main.py              # Entry point
```

---

### 12. Key Implementation Gotchas

1. **Mask tokens need position information**: The shared mask embedding alone is meaningless. It MUST be combined with positional embeddings.

2. **Context encoder only sees context patches**: For efficiency (like MAE), the context encoder should only process visible patches. The positional information is added BEFORE encoding.

3. **Target encoder sees full image**: Unlike context, the target encoder processes all patches, then we extract the target blocks from its output. This ensures high-quality semantic targets.

4. **No gradients through target encoder**: Always use `torch.no_grad()` and don't include target encoder in optimizer.

5. **Batch handling of variable masks**: Each sample in a batch may have different context/target indices. Handle this with padding or process samples individually.

6. **EMA momentum schedule**: Start at 0.996, linearly increase to 1.0. This is important for training stability.

7. **Weight decay schedule**: In the paper, they increase from 0.04 to 0.4. For CIFAR-10, simpler constant weight decay (0.05) should work.

---

### 13. Expected Training Behavior

- **Loss**: Should steadily decrease over epochs. The L2 loss between predicted and target representations should go from ~1.0+ to ~0.1-0.3.

- **Early epochs**: Predictor learns basic position-to-position mappings.

- **Later epochs**: Predictor learns semantic relationships (predicting meaningful content based on context).

- **Linear probe accuracy**: On CIFAR-10, expect 60-75% accuracy with linear probe (compared to ~95% supervised).

---

## Summary

The key insight of I-JEPA is predicting in representation space rather than pixel space. The mask token mechanism is simple but powerful:

```
mask_token_for_position_j = shared_learnable_embedding + positional_embedding[j]
```

This tells the predictor: "Predict what should be at position j, given the context." The predictor learns to use the context representation to make this prediction, forcing it to learn semantic relationships in the image.
