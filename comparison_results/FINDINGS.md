# I-JEPA Embedding Analysis: Local vs HuggingFace ViT-Huge

## Overview

This analysis compares embeddings from two I-JEPA models evaluated on CIFAR-10:
- **Local Model**: Custom I-JEPA trained from scratch (300 epochs, 192-dim embeddings)
- **HF Model**: `facebook/ijepa_vith14_1k` pretrained ViT-Huge (1280-dim embeddings)

## Quantitative Results

| Metric | Local I-JEPA | HF ViT-Huge | Ratio |
|--------|--------------|-------------|-------|
| Embedding Dimension | 192 | 1280 | 0.15x |
| Variance in 10 PCs | 30.2% | 49.9% | 0.61x |
| Variance in 50 PCs | 80.6% | 78.4% | 1.03x |
| Silhouette Score | 0.005 | 0.034 | 0.15x |
| NN Precision@5 | 49.8% | 68.5% | 0.73x |

## Key Findings

### 1. Representation Efficiency

The local model achieves **73% of the retrieval performance** (49.8% vs 68.5% precision@5) while using only **15% of the embedding dimensions** (192 vs 1280). This demonstrates that I-JEPA can learn meaningful representations even at small scale.

### 2. Variance Structure

- **HF model**: Concentrates 50% of variance in just 10 principal components, suggesting highly structured, low-dimensional representations hidden in the high-dimensional space
- **Local model**: More distributed variance (30% in 10 PCs), but reaches similar total variance (80%) by 50 PCs
- Both models capture ~80% of variance in 50 dimensions, suggesting effective dimensionality is similar

### 3. Class Separability

The silhouette score measures how well-separated class clusters are:
- HF model: 0.034 (better separation)
- Local model: 0.005 (classes more overlapping)

This 7x difference indicates the HF model learns more discriminative features, likely due to:
- Larger model capacity (ViT-Huge vs small ViT)
- Training on ImageNet (1M+ images) vs CIFAR-10 (50K images)
- Higher resolution input (224x224 vs 32x32)

### 4. What the Embeddings Learn

Based on the similarity matrices and t-SNE visualizations:

**Semantic groupings observed in both models:**
- Vehicles cluster together (airplane, automobile, ship, truck)
- Animals form related clusters (cat, dog, deer, horse)
- Frog forms a distinct cluster (unique appearance)

**Key differences:**
- HF model shows tighter, more separated clusters
- Local model has more overlap between semantically similar classes
- Both models struggle with fine-grained distinctions (cat vs dog, automobile vs truck)

### 5. Nearest Neighbor Retrieval Quality

At precision@5 = 49.8%, the local model correctly retrieves same-class images about half the time. This is:
- Significantly above random chance (10% for 10 classes)
- Indicates learned semantic similarity
- Shows room for improvement with longer training or larger models

## Conclusions

1. **I-JEPA works at small scale**: Even a tiny 192-dim encoder trained on CIFAR-10 learns semantically meaningful representations

2. **Scale matters**: The pretrained ViT-Huge significantly outperforms on all metrics, confirming the importance of model size and training data

3. **Efficient encoding**: Both models compress semantic information into a relatively small number of effective dimensions (~50 PCs capture 80% variance)

4. **Self-supervised learning discovers categories**: Without any labels during training, both models learn to group semantically similar images together

## Files Generated

```
comparison_results/
├── tsne_comparison.png      # t-SNE clustering visualization
├── pca_comparison.png       # PCA with variance explained
├── similarity_matrices.png  # Class centroid cosine similarity
├── variance_explained.png   # Per-component and cumulative variance
├── nn_*.png                 # Nearest neighbor retrieval examples
└── metrics.json             # Raw metrics data
```

## Reproduction

```bash
PYTHONPATH=. ~/.pyenv/versions/3.11.9/bin/python scripts/compare_embeddings.py \
    --local-checkpoint checkpoints_300ep/checkpoint_latest.pt \
    --hf-model facebook/ijepa_vith14_1k \
    --output-dir comparison_results \
    --max-samples 2000
```
