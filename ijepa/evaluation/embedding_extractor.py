"""
Unified embedding extraction for different I-JEPA model sources.

Provides a common interface for:
- Local ViTEncoder (32x32 CIFAR-10)
- HuggingFace IJepaModel (224x224 ImageNet)
"""

from typing import Protocol
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from ijepa.data.cifar10 import CIFAR10_MEAN, CIFAR10_STD


class EmbeddingExtractor(Protocol):
    """Protocol for embedding extraction from any model."""

    def extract_embeddings(
        self,
        dataloader: DataLoader,
        max_samples: int = 5000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings from images.

        Returns:
            features: (N, embed_dim) feature vectors
            labels: (N,) class labels
        """
        ...

    @property
    def embed_dim(self) -> int:
        """Return embedding dimension."""
        ...

    @property
    def model_name(self) -> str:
        """Return human-readable model name."""
        ...


class LocalIJEPAExtractor:
    """Extract embeddings from locally trained I-JEPA model."""

    def __init__(
        self,
        model,
        device: torch.device,
        use_target_encoder: bool = True,
    ):
        """
        Args:
            model: IJEPA model instance
            device: Device to run inference on
            use_target_encoder: If True, use target encoder; else context encoder
        """
        self.encoder = model.target_encoder if use_target_encoder else model.context_encoder
        self.device = device
        self._embed_dim = model.encoder_embed_dim
        self._name = f"Local I-JEPA ({'Target' if use_target_encoder else 'Context'})"
        self.encoder.eval()

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_name(self) -> str:
        return self._name

    @torch.no_grad()
    def extract_embeddings(
        self,
        dataloader: DataLoader,
        max_samples: int = 5000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract features using global average pooling over patches."""
        all_features = []
        all_labels = []
        total = 0

        for images, labels in tqdm(dataloader, desc=f"Extracting ({self._name})"):
            if total >= max_samples:
                break

            images = images.to(self.device)
            features = self.encoder(images)  # (B, num_patches, embed_dim)
            features = features.mean(dim=1)  # Global avg pool -> (B, embed_dim)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            total += len(labels)

        features = np.concatenate(all_features)[:max_samples]
        labels = np.concatenate(all_labels)[:max_samples]

        return features, labels


class HuggingFaceIJEPAExtractor:
    """Extract embeddings from HuggingFace I-JEPA model."""

    def __init__(
        self,
        model_id: str = "facebook/ijepa_vith14_1k",
        device: torch.device = None,
        use_float16: bool = True,
    ):
        """
        Args:
            model_id: HuggingFace model identifier
            device: Device to run inference on
            use_float16: Use float16 for memory efficiency
        """
        from transformers import AutoModel, AutoProcessor

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model with optional float16
        dtype = torch.float16 if use_float16 and self.device.type == 'cuda' else torch.float32
        print(f"Loading {model_id} with dtype={dtype}...")

        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)

        self._embed_dim = self.model.config.hidden_size
        self._name = f"HF I-JEPA ({model_id.split('/')[-1]})"
        self.model.eval()

        print(f"Loaded {self._name} with embed_dim={self._embed_dim}")

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def model_name(self) -> str:
        return self._name

    @torch.no_grad()
    def extract_embeddings(
        self,
        dataloader: DataLoader,
        max_samples: int = 5000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings from CIFAR-10 images.

        Note: Images are resized from 32x32 to 224x224 and renormalized
        using the HF processor.
        """
        from PIL import Image
        import torchvision.transforms.functional as TF

        all_features = []
        all_labels = []
        total = 0

        # CIFAR-10 denormalization constants
        cifar_mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
        cifar_std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

        for images, labels in tqdm(dataloader, desc=f"Extracting ({self._name})"):
            if total >= max_samples:
                break

            batch_features = []

            for img in images:
                # Denormalize CIFAR-10 image back to [0, 1]
                img = img * cifar_std + cifar_mean
                img = img.clamp(0, 1)

                # Convert to PIL and resize to 224x224
                pil_img = TF.to_pil_image(img)
                pil_img = pil_img.resize((224, 224), Image.BILINEAR)

                # Process with HF processor (handles normalization)
                inputs = self.processor(pil_img, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device, dtype=self.model.dtype)

                # Get embeddings
                outputs = self.model(pixel_values)
                # Use mean pooling over sequence (patches)
                features = outputs.last_hidden_state.mean(dim=1)  # (1, hidden_size)
                batch_features.append(features.float().cpu())

            batch_features = torch.cat(batch_features, dim=0).numpy()
            all_features.append(batch_features)
            all_labels.append(labels.numpy())
            total += len(labels)

        features = np.concatenate(all_features)[:max_samples]
        labels = np.concatenate(all_labels)[:max_samples]

        return features, labels
