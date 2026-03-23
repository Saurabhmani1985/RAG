"""
tests/test_retrieval.py
────────────────────────
Unit tests for the vector store and retriever.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-placeholder")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/test_chroma_retrieval")

TEST_COLLECTION = "test_retrieval_collection"


@pytest.fixture(autouse=True)
def clean_test_collection():
    """Clean up test collection before each test."""
    yield
    try:
        from src.retrieval.vector_store import get_vector_store
        store = get_vector_store()
        if TEST_COLLECTION in store.list_collections():
            store.delete_collection(TEST_COLLECTION)
    except Exception:
        pass


def _make_chunks(n: int = 5):
    from src.ingestion.pdf_parser import ParsedChunk
    chunks = []
    texts = [
        "Fault code P0087 Rail Pressure Build Fault. Turns CEL ON.",
        "IMV Inlet Metering Valve. Short circuit to positive. P0251.",
        "| P0087 | CE | Rail Pressure Build | Pressure does not build |",
        "ECU pin 40 connects to Rail Pressure Sensor signal terminal 1.",
        "Coolant temperature signal fault P0115. Torque reduction activated.",
    ]
    types = ["text", "text", "table", "image", "text"]
    for i in range(min(n, len(texts))):
        chunks.append(
            ParsedChunk(
                chunk_id=f"test_chunk_{i:03d}",
                source="test.pdf",
                page=i + 1,
                chunk_type=types[i],
                text=texts[i],
                metadata={"source": "test.pdf", "page": i + 1, "chunk_type": types[i]},
            )
        )
    return chunks


def test_vector_store_upsert_and_count():
    from src.ingestion.embedder import Embedder
    from src.retrieval.vector_store import VectorStore

    chunks = _make_chunks(5)
    embedder = Embedder()
    vecs = embedder.embed_chunks(chunks)

    store = VectorStore()
    store.upsert(chunks=chunks, embeddings=vecs, collection=TEST_COLLECTION)

    count = store.collection_count(TEST_COLLECTION)
    assert count == 5


def test_vector_store_query_returns_results():
    from src.ingestion.embedder import Embedder
    from src.retrieval.vector_store import VectorStore

    chunks = _make_chunks(5)
    embedder = Embedder()
    vecs = embedder.embed_chunks(chunks)

    store = VectorStore()
    store.upsert(chunks=chunks, embeddings=vecs, collection=TEST_COLLECTION)

    query_vec = embedder.embed_query("rail pressure fault P0087")
    results = store.query(query_vec, top_k=3, collection=TEST_COLLECTION)

    assert len(results) > 0
    assert len(results) <= 3
    for r in results:
        assert "chunk_id" in r
        assert "text" in r
        assert "score" in r
        assert 0.0 <= r["score"] <= 1.0


def test_vector_store_chunk_type_filter():
    from src.ingestion.embedder import Embedder
    from src.retrieval.vector_store import VectorStore

    chunks = _make_chunks(5)
    embedder = Embedder()
    vecs = embedder.embed_chunks(chunks)

    store = VectorStore()
    store.upsert(chunks=chunks, embeddings=vecs, collection=TEST_COLLECTION)

    query_vec = embedder.embed_query("ECU wiring diagram")
    results = store.query(
        query_vec, top_k=5,
        collection=TEST_COLLECTION,
        chunk_type_filter="image",
    )

    # All returned results must be image type
    for r in results:
        assert r["chunk_type"] == "image"


def test_vector_store_list_documents():
    from src.ingestion.embedder import Embedder
    from src.retrieval.vector_store import VectorStore

    chunks = _make_chunks(5)
    embedder = Embedder()
    vecs = embedder.embed_chunks(chunks)

    store = VectorStore()
    store.upsert(chunks=chunks, embeddings=vecs, collection=TEST_COLLECTION)

    docs = store.list_indexed_documents(TEST_COLLECTION)
    assert len(docs) == 1
    assert docs[0]["filename"] == "test.pdf"
    assert docs[0]["total_chunks"] == 5


def test_vector_store_delete_document():
    from src.ingestion.embedder import Embedder
    from src.retrieval.vector_store import VectorStore

    chunks = _make_chunks(5)
    embedder = Embedder()
    vecs = embedder.embed_chunks(chunks)

    store = VectorStore()
    store.upsert(chunks=chunks, embeddings=vecs, collection=TEST_COLLECTION)
    assert store.collection_count(TEST_COLLECTION) == 5

    removed = store.delete_document("test.pdf", collection=TEST_COLLECTION)
    assert removed == 5
    assert store.collection_count(TEST_COLLECTION) == 0
