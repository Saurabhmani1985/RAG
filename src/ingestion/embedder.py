"""
src/ingestion/embedder.py
─────────────────────────
Text embedding for all chunk types.

All three chunk types (text, table, image) are embedded using the same
sentence-transformers model.  Image chunks use their VLM-generated text
summary as the embedding input — enabling unified semantic search across
modalities with a single vector index per collection.

Design decision: one unified embedding model rather than separate text +
CLIP embedders.  This simplifies the retrieval pipeline (single query
vector, single collection per document) at the cost of image retrieval
being mediated by the textual summary quality.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from src.config import get_settings
from src.ingestion.pdf_parser import ParsedChunk

_lock = threading.Lock()
_model = None


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                cfg = get_settings()
                _model = SentenceTransformer(cfg.embedding_model)
    return _model


class Embedder:
    """
    Produces dense vector embeddings for ParsedChunk objects.

    All chunk types share a single embedding space:
      - text chunks   → embed raw text
      - table chunks  → embed Markdown table (text model handles this well)
      - image chunks  → embed VLM-generated summary (key design choice)
    """

    def embed_chunks(
        self,
        chunks: list[ParsedChunk],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of chunks.

        Returns
        -------
        np.ndarray of shape (N, embedding_dim), L2-normalised.
        """
        if not chunks:
            return np.empty((0, get_settings().embedding_dim))

        model = _get_model()
        texts = [c.text for c in chunks]

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings  # shape (N, dim)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (dim,)."""
        model = _get_model()
        vec = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec[0]

    @property
    def dim(self) -> int:
        return get_settings().embedding_dim


# Singleton
_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
