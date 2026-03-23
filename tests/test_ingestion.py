"""
tests/test_ingestion.py
────────────────────────
Unit tests for the PDF parsing and chunking pipeline.
Tests run against the real sample PDF (no mocking needed for parsing).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-placeholder")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/test_chroma_ingest")

SAMPLE_PDF = Path(__file__).parent.parent / "sample_documents" / "Diagnostic_Document.pdf"


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Sample PDF not available")
def test_pdf_parser_extracts_text():
    from src.ingestion.pdf_parser import PDFParser

    parser = PDFParser(use_vision=False)  # skip VLM in unit tests
    result = parser.parse(SAMPLE_PDF)

    assert result.total_pages > 0
    assert len(result.text_chunks) > 0
    for chunk in result.text_chunks:
        assert chunk.chunk_type == "text"
        assert len(chunk.text) > 10
        assert chunk.page >= 1
        assert chunk.source == SAMPLE_PDF.name


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Sample PDF not available")
def test_pdf_parser_extracts_tables():
    from src.ingestion.pdf_parser import PDFParser

    parser = PDFParser(use_vision=False)
    result = parser.parse(SAMPLE_PDF)

    assert len(result.table_chunks) > 0
    for chunk in result.table_chunks:
        assert chunk.chunk_type == "table"
        # Tables should be Markdown-formatted
        assert "|" in chunk.text
        assert chunk.page >= 1


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Sample PDF not available")
def test_pdf_parser_extracts_images():
    from src.ingestion.pdf_parser import PDFParser

    # use_vision=False → placeholder caption instead of VLM call
    parser = PDFParser(use_vision=False)
    result = parser.parse(SAMPLE_PDF)

    assert len(result.image_chunks) > 0
    for chunk in result.image_chunks:
        assert chunk.chunk_type == "image"
        assert len(chunk.text) > 0
        assert "width" in chunk.metadata
        assert "height" in chunk.metadata


def test_chunker_splits_long_text():
    from src.ingestion.chunker import TextChunker
    from src.ingestion.pdf_parser import ParsedChunk

    long_text = "This is a sentence about fault diagnosis. " * 300  # ~12k chars
    chunk = ParsedChunk(
        chunk_id="test_001",
        source="test.pdf",
        page=1,
        chunk_type="text",
        text=long_text,
    )

    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    result = chunker.split([chunk])

    assert len(result) > 1
    for sub in result:
        assert sub.chunk_type == "text"
        assert sub.source == "test.pdf"
        assert sub.page == 1


def test_chunker_preserves_table_chunks():
    from src.ingestion.chunker import TextChunker
    from src.ingestion.pdf_parser import ParsedChunk

    table_chunk = ParsedChunk(
        chunk_id="tab_001",
        source="test.pdf",
        page=5,
        chunk_type="table",
        text="| Code | Fault | Description |\n| --- | --- | --- |\n| P0087 | Rail Pressure Build | ... |",
    )

    chunker = TextChunker()
    result = chunker.split([table_chunk])

    assert len(result) == 1
    assert result[0].chunk_id == "tab_001"


def test_chunker_preserves_image_chunks():
    from src.ingestion.chunker import TextChunker
    from src.ingestion.pdf_parser import ParsedChunk

    img_chunk = ParsedChunk(
        chunk_id="img_001",
        source="test.pdf",
        page=22,
        chunk_type="image",
        text="ECU pin 40 connects to Rail Pressure Sensor terminal 1 (Signal). Pin 58 to terminal 2 (Ground). Pin 77 to terminal 3 (Vref).",
    )

    chunker = TextChunker()
    result = chunker.split([img_chunk])

    assert len(result) == 1
    assert result[0].chunk_type == "image"


def test_embedder_produces_correct_shape():
    from src.ingestion.pdf_parser import ParsedChunk
    from src.ingestion.embedder import Embedder

    chunks = [
        ParsedChunk("c1", "test.pdf", 1, "text", "Fault code P0087 rail pressure build error."),
        ParsedChunk("c2", "test.pdf", 2, "table", "| P0087 | CE | Rail Pressure Build |"),
        ParsedChunk("c3", "test.pdf", 3, "image", "ECU wiring diagram showing pin 40 to rail sensor."),
    ]

    embedder = Embedder()
    vecs = embedder.embed_chunks(chunks)

    assert vecs.shape == (3, embedder.dim)
    # Vectors should be L2-normalised (norm ≈ 1.0)
    import numpy as np
    norms = np.linalg.norm(vecs, axis=1)
    assert all(abs(n - 1.0) < 1e-4 for n in norms)


def test_embedder_query_shape():
    from src.ingestion.embedder import Embedder

    embedder = Embedder()
    vec = embedder.embed_query("What is the recovery mode for P0087?")

    assert vec.shape == (embedder.dim,)
