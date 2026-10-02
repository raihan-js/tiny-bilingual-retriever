"""Model loading and encoding for retrieval evaluation."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class Encoder:
    """Wrapper around a sentence-transformers model for retrieval."""

    def __init__(self, name: str, device: str = "cpu", batch_size: int = 64,
                 max_length: int = 512):
        self.name = name
        self.model = SentenceTransformer(name, device=device)
        self.batch_size = batch_size
        self.max_length = max_length
        self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Encode texts to embeddings."""
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Encode queries (with instruction prefix if needed)."""
        if "e5" in self.name.lower():
            texts = [f"query: {t}" for t in texts]
        return self.encode(texts)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Encode passages (with instruction prefix if needed)."""
        if "e5" in self.name.lower():
            texts = [f"passage: {t}" for t in texts]
        return self.encode(texts)

    def index_size_mb(self, n_passages: int) -> float:
        """Estimated index size in MB (float32)."""
        return n_passages * self.dim * 4 / (1024 * 1024)
