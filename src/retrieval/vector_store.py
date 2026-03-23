"""
src/retrieval/vector_store.py
──────────────────────────────
ChromaDB persistent vector store wrapper.

Uses a SINGLE collection for all chunk types (text, table, image).
chunk_type is stored in metadata, enabling filtered retrieval.

Why ChromaDB over FAISS:
  - Native metadata filtering (filter by chunk_type, source, page)
  - Persistent storage out-of-the-box (no manual serialisation)
  - HTTP client available for production scaling
  - Active OSS project with good Python API
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from src.config import get_settings
from src.ingestion.pdf_parser import ParsedChunk

_lock = threading.Lock()
_client: Optional[chromadb.ClientAPI] = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                cfg = get_settings()
                _client = chromadb.PersistentClient(
                    path=str(cfg.chroma_persist_dir),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
    return _client


class VectorStore:
    """
    ChromaDB-backed vector store for multimodal RAG.

    All chunk types live in one collection per corpus.
    Metadata fields stored per chunk:
      - source       : PDF filename
      - page         : page number (int)
      - chunk_type   : "text" | "table" | "image"
      - chunk_id     : unique chunk identifier
    """

    def __init__(self, collection: Optional[str] = None):
        cfg = get_settings()
        self._default_collection = collection or cfg.default_collection
        self._client = _get_client()

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        chunks: list[ParsedChunk],
        embeddings: np.ndarray,
        collection: Optional[str] = None,
        batch_size: int = 256,
    ) -> None:
        """Upsert chunks and embeddings into ChromaDB."""
        col = self._get_or_create_collection(collection or self._default_collection)

        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [
            {
                "source": c.source,
                "page": c.page,
                "chunk_type": c.chunk_type,
                "chunk_id": c.chunk_id,
                **{k: v for k, v in c.metadata.items()
                   if isinstance(v, (str, int, float, bool))},
            }
            for c in chunks
        ]
        emb_list = embeddings.tolist()

        for i in range(0, len(ids), batch_size):
            sl = slice(i, i + batch_size)
            col.upsert(
                ids=ids[sl],
                embeddings=emb_list[sl],
                documents=docs[sl],
                metadatas=metas[sl],
            )

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 6,
        collection: Optional[str] = None,
        chunk_type_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Query the vector store.

        Returns a list of dicts:
        [{"chunk_id", "text", "chunk_type", "source", "page", "score"}, ...]
        """
        col = self._get_or_create_collection(collection or self._default_collection)

        if col.count() == 0:
            return []

        # Build optional where clause
        where: Optional[dict] = None
        conditions = []
        if chunk_type_filter:
            conditions.append({"chunk_type": {"$eq": chunk_type_filter}})
        if source_filter:
            conditions.append({"source": {"$eq": source_filter}})

        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        kwargs = dict(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, col.count()),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)

        output = []
        for cid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append(
                {
                    "chunk_id": cid,
                    "text": doc,
                    "chunk_type": meta.get("chunk_type", "text"),
                    "source": meta.get("source", ""),
                    "page": meta.get("page", 0),
                    "score": round(1.0 - dist, 4),   # distance → similarity
                    "metadata": meta,
                }
            )

        return output

    # ── Metadata / management ─────────────────────────────────────────────────

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.list_collections()]

    def collection_count(self, collection: Optional[str] = None) -> int:
        col = self._get_or_create_collection(collection or self._default_collection)
        return col.count()

    def list_indexed_documents(self, collection: Optional[str] = None) -> list[dict]:
        """Return unique (source, chunk counts) for all indexed documents."""
        col = self._get_or_create_collection(collection or self._default_collection)
        if col.count() == 0:
            return []

        # Retrieve all metadata to aggregate
        all_data = col.get(include=["metadatas"])
        source_stats: dict[str, dict] = {}

        for meta in all_data["metadatas"]:
            src = meta.get("source", "unknown")
            if src not in source_stats:
                source_stats[src] = {"text": 0, "table": 0, "image": 0}
            ct = meta.get("chunk_type", "text")
            source_stats[src][ct] = source_stats[src].get(ct, 0) + 1

        return [
            {
                "filename": src,
                "chunks": stats,
                "total_chunks": sum(stats.values()),
            }
            for src, stats in source_stats.items()
        ]

    def delete_document(self, source: str, collection: Optional[str] = None) -> int:
        """Delete all chunks for a given source filename. Returns deleted count."""
        col = self._get_or_create_collection(collection or self._default_collection)
        result = col.get(where={"source": {"$eq": source}}, include=["metadatas"])
        ids = result["ids"]
        if ids:
            col.delete(ids=ids)
        return len(ids)

    def delete_collection(self, collection: str) -> None:
        self._client.delete_collection(collection)

    def get_stats(self, collection: Optional[str] = None) -> dict:
        """Return collection statistics."""
        col_name = collection or self._default_collection
        try:
            col = self._get_or_create_collection(col_name)
            count = col.count()
        except Exception:
            count = 0
        return {
            "collection": col_name,
            "total_chunks": count,
            "all_collections": self.list_collections(),
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_or_create_collection(self, name: str) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )


# Singleton
_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
