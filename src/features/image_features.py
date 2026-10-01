"""Extraction d'embeddings d'images via ResNet18 pré-entraîné (gelé), packagé
comme un transformer scikit-learn pour s'intégrer au ColumnTransformer."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin

IMAGE_SIZE = 224
IMAGE_EMBEDDING_DIM = 512


class ImageEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Prend une colonne de chemins d'images, retourne une matrice d'embeddings (N, 512)."""

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _get_model(self) -> nn.Module:
        if self._model is None:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            resnet.fc = nn.Identity()  # embedding 512-d au lieu des 1000 classes ImageNet
            resnet.eval()
            resnet.to(self._device)
            self._model = resnet
        return self._model

    def fit(self, X, y=None):
        return self  # réseau gelé, rien à apprendre

    def transform(self, X) -> np.ndarray:
        paths = X.iloc[:, 0] if hasattr(X, "iloc") else X
        model = self._get_model()
        embeddings = []

        paths = list(paths)
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start:start + self.batch_size]
            tensors, valid_mask = [], []

            for path in batch_paths:
                loaded = False
                if path is not None and isinstance(path, str):
                    try:
                        image = Image.open(path).convert("RGB")
                        tensors.append(self._transform(image))
                        valid_mask.append(True)
                        loaded = True
                    except Exception:
                        pass
                if not loaded:
                    tensors.append(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))
                    valid_mask.append(False)

            batch_tensor = torch.stack(tensors).to(self._device)
            with torch.no_grad():
                batch_embeddings = model(batch_tensor).cpu().numpy()

            for i, is_valid in enumerate(valid_mask):
                if not is_valid:
                    batch_embeddings[i] = np.zeros(IMAGE_EMBEDDING_DIM, dtype=np.float32)

            embeddings.append(batch_embeddings)

        return np.vstack(embeddings) if embeddings else np.empty((0, IMAGE_EMBEDDING_DIM))

    def get_feature_names_out(self, input_features=None):
        return np.array([f"image_emb_{i}" for i in range(IMAGE_EMBEDDING_DIM)])