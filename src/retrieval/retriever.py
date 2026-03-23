"""
src/retrieval/retriever.py
───────────────────────────
RAG retrieval: embed query → search ChromaDB → assemble context → generate.

The Retriever class handles:
  1. Query embedding
  2. Vector similarity search (with optional chunk-type filters)
  3. Result deduplication and ranking
  4. Context assembly for the LLM
  5. LLM generation via the custom prompt template

The generate() method returns a fully structured QueryResult.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from src.config import get_settings
from src.ingestion.embedder import get_embedder
from src.models.llm import get_llm
from src.retrieval.vector_store import get_vector_store


@dataclass
class SourceReference:
    filename: str
    page: int
    chunk_type: str   # "text" | "table" | "image"
    score: float
    excerpt: str      # first 200 chars of the chunk text


@dataclass
class QueryResult:
    question: str
    answer: str
    sources: list[SourceReference]
    retrieval_stats: dict
    latency_ms: float
    model: str

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [
                {
                    "filename": s.filename,
                    "page": s.page,
                    "chunk_type": s.chunk_type,
                    "score": s.score,
                    "excerpt": s.excerpt,
                }
                for s in self.sources
            ],
            "retrieval_stats": self.retrieval_stats,
            "latency_ms": self.latency_ms,
            "model": self.model,
        }


class Retriever:
    """
    Full RAG pipeline: retrieve → assemble context → generate.

    Usage:
        retriever = Retriever()
        result = retriever.query("What does fault code P0087 mean?")
    """

    def __init__(self, collection: Optional[str] = None):
        cfg = get_settings()
        self.collection = collection or cfg.default_collection
        self.embedder = get_embedder()
        self.store = get_vector_store()
        self.llm = get_llm()
        self.cfg = cfg

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
        chunk_type_filter: Optional[str] = None,
    ) -> QueryResult:
        """
        Execute the full RAG pipeline for a natural language question.

        Parameters
        ----------
        question          : Natural language question
        top_k             : Number of chunks to retrieve (default from config)
        collection        : Override default collection name
        chunk_type_filter : Restrict to "text", "table", or "image" chunks only
        """
        t0 = time.perf_counter()
        top_k = top_k or self.cfg.default_top_k
        col = collection or self.collection

        # ── 1. Embed query ────────────────────────────────────────────────────
        query_vec = self.embedder.embed_query(question)

        # ── 2. Retrieve chunks ────────────────────────────────────────────────
        raw_results = self.store.query(
            query_embedding=query_vec,
            top_k=top_k,
            collection=col,
            chunk_type_filter=chunk_type_filter,
        )

        # ── 3. Assemble context string ────────────────────────────────────────
        context_str = self.llm.format_context(raw_results)

        # ── 4. Generate answer ────────────────────────────────────────────────
        llm_response = self.llm.generate(question=question, context=context_str)

        latency_ms = (time.perf_counter() - t0) * 1000

        # ── 5. Build source references ────────────────────────────────────────
        sources = [
            SourceReference(
                filename=r["source"],
                page=r["page"],
                chunk_type=r["chunk_type"],
                score=r["score"],
                excerpt=r["text"][:300].replace("\n", " "),
            )
            for r in raw_results
        ]

        # Retrieval breakdown by type
        type_counts: dict[str, int] = {}
        for r in raw_results:
            ct = r["chunk_type"]
            type_counts[ct] = type_counts.get(ct, 0) + 1

        return QueryResult(
            question=question,
            answer=llm_response.answer,
            sources=sources,
            retrieval_stats={
                "total_retrieved": len(raw_results),
                "by_type": type_counts,
                "collection": col,
                "top_k_requested": top_k,
                "llm_input_tokens": llm_response.input_tokens,
                "llm_output_tokens": llm_response.output_tokens,
                "llm_latency_ms": llm_response.latency_ms,
            },
            latency_ms=round(latency_ms, 1),
            model=llm_response.model,
        )


# Singleton
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
