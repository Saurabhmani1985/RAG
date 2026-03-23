"""
src/ingestion/pipeline.py
──────────────────────────
End-to-end ingestion pipeline orchestrator.

Flow:
  PDF file
    │
    ├─ PDFParser ──────────────────────────────────────────────────┐
    │   ├── Text   → text chunks                                   │
    │   ├── Tables → table chunks (Markdown)                       │
    │   └── Images → VLM summarise → image chunks (text summaries) │
    │                                                              │
    ├─ TextChunker ─ split long text chunks                        │
    │                                                              │
    ├─ Embedder ─── embed ALL chunk types (unified embedding space) │
    │                                                              │
    └─ VectorStore ─ upsert to ChromaDB                           ─┘

The pipeline exposes a synchronous `run()` and an async `run_async()`
wrapper so it can be called from FastAPI background tasks.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.ingestion.chunker import TextChunker
from src.ingestion.embedder import get_embedder
from src.ingestion.pdf_parser import PDFParser, ParsedChunk
from src.retrieval.vector_store import get_vector_store

_executor = ThreadPoolExecutor(max_workers=2)


@dataclass
class IngestionResult:
    filename: str
    collection: str
    total_pages: int
    chunk_counts: dict[str, int]   # {"text": N, "table": N, "image": N, "total": N}
    processing_time_s: float
    errors: list[str]

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "collection": self.collection,
            "total_pages": self.total_pages,
            "chunk_counts": self.chunk_counts,
            "processing_time_s": round(self.processing_time_s, 2),
            "errors": self.errors,
        }


class IngestionPipeline:
    """
    Orchestrates the full PDF → ChromaDB ingestion pipeline.

    Usage:
        pipeline = IngestionPipeline()
        result = pipeline.run(Path("sample_documents/Diagnostic_Document.pdf"))
    """

    def __init__(self, collection: Optional[str] = None, use_vision: bool = True):
        from src.config import get_settings
        cfg = get_settings()
        self.collection = collection or cfg.default_collection
        self.parser = PDFParser(use_vision=use_vision)
        self.chunker = TextChunker()
        self.embedder = get_embedder()
        self.store = get_vector_store()

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self, pdf_path: Path) -> IngestionResult:
        """Synchronous ingestion. Blocks until complete."""
        pdf_path = Path(pdf_path)
        t0 = time.perf_counter()

        # 1. Parse
        parse_result = self.parser.parse(pdf_path)

        # 2. Chunk (text only; tables/images pass through)
        all_raw = (
            parse_result.text_chunks
            + parse_result.table_chunks
            + parse_result.image_chunks
        )
        chunked = self.chunker.split(all_raw)

        # 3. Embed
        embeddings = self.embedder.embed_chunks(chunked, show_progress=True)

        # 4. Upsert to vector store
        self.store.upsert(
            collection=self.collection,
            chunks=chunked,
            embeddings=embeddings,
        )

        elapsed = time.perf_counter() - t0

        # Count final chunk types
        counts: dict[str, int] = {"text": 0, "table": 0, "image": 0}
        for c in chunked:
            counts[c.chunk_type] = counts.get(c.chunk_type, 0) + 1
        counts["total"] = sum(counts.values())

        return IngestionResult(
            filename=pdf_path.name,
            collection=self.collection,
            total_pages=parse_result.total_pages,
            chunk_counts=counts,
            processing_time_s=elapsed,
            errors=parse_result.errors,
        )

    async def run_async(self, pdf_path: Path) -> IngestionResult:
        """Async wrapper — runs blocking ingestion in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.run, pdf_path)
