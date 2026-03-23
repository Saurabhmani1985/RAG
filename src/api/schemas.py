"""
src/api/schemas.py
───────────────────
Pydantic v2 request and response schemas for every API endpoint.
Strict typing ensures the OpenAPI/Swagger docs are accurate and complete.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════════════════════════
# /health
# ══════════════════════════════════════════════════════════════════════════════

class ComponentStatus(BaseModel):
    status: Literal["ok", "unavailable"]
    detail: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    uptime_seconds: float
    indexed_documents: int
    total_chunks: int
    index_size_mb: float
    components: dict[str, ComponentStatus]

    model_config = {"json_schema_extra": {
        "example": {
            "status": "ok",
            "version": "1.0.0",
            "uptime_seconds": 142.3,
            "indexed_documents": 1,
            "total_chunks": 312,
            "index_size_mb": 4.2,
            "components": {
                "vector_store": {"status": "ok", "detail": "ChromaDB connected"},
                "embedding_model": {"status": "ok", "detail": "all-MiniLM-L6-v2 loaded"},
                "llm": {"status": "ok", "detail": "claude-sonnet-4-6"},
                "vision_model": {"status": "ok", "detail": "claude-opus-4-5"},
            },
        }
    }}


# ══════════════════════════════════════════════════════════════════════════════
# /ingest
# ══════════════════════════════════════════════════════════════════════════════

class ChunkCounts(BaseModel):
    text: int = Field(..., description="Number of text chunks")
    table: int = Field(..., description="Number of table chunks")
    image: int = Field(..., description="Number of image (VLM-summarised) chunks")
    total: int = Field(..., description="Total chunks indexed")


class IngestResponse(BaseModel):
    status: Literal["success", "partial", "error"]
    filename: str
    collection: str
    total_pages: int
    chunk_counts: ChunkCounts
    processing_time_s: float
    warnings: list[str] = Field(default_factory=list, description="Non-fatal parsing issues")
    message: str = "Document successfully ingested."

    model_config = {"json_schema_extra": {
        "example": {
            "status": "success",
            "filename": "Diagnostic_Document.pdf",
            "collection": "diagnostic_rag",
            "total_pages": 86,
            "chunk_counts": {"text": 248, "table": 41, "image": 23, "total": 312},
            "processing_time_s": 87.4,
            "warnings": [],
            "message": "Document successfully ingested.",
        }
    }}


# ══════════════════════════════════════════════════════════════════════════════
# /query
# ══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural language question to answer from the indexed documents.",
        examples=["What should I check when fault code P0087 appears?"],
    )
    collection: str = Field(
        "diagnostic_rag",
        description="ChromaDB collection to query. Must have been previously ingested.",
    )
    top_k: int = Field(
        6,
        ge=1,
        le=20,
        description="Number of chunks to retrieve from the vector store.",
    )
    chunk_type_filter: Optional[Literal["text", "table", "image"]] = Field(
        None,
        description="Restrict retrieval to a specific chunk type. Leave null for all types.",
    )

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        return v.strip()


class SourceReference(BaseModel):
    filename: str = Field(..., description="Source PDF filename")
    page: int = Field(..., description="Page number within the PDF (1-indexed)")
    chunk_type: Literal["text", "table", "image"] = Field(
        ..., description="Type of the retrieved chunk"
    )
    score: float = Field(..., description="Cosine similarity score (0–1, higher = more relevant)")
    excerpt: str = Field(..., description="First 300 characters of the chunk text")


class RetrievalStats(BaseModel):
    total_retrieved: int
    by_type: dict[str, int]
    collection: str
    top_k_requested: int
    llm_input_tokens: int
    llm_output_tokens: int
    llm_latency_ms: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceReference]
    retrieval_stats: RetrievalStats
    latency_ms: float
    model: str

    model_config = {"json_schema_extra": {
        "example": {
            "question": "What does fault code P0087 mean and how do I fix it?",
            "answer": "Fault code `P0087` is the **Rail Pressure Build Fault**...",
            "sources": [
                {
                    "filename": "Diagnostic_Document.pdf",
                    "page": 28,
                    "chunk_type": "text",
                    "score": 0.91,
                    "excerpt": "CODE FAULT NAME P0087 Rail Pressure Build Error...",
                }
            ],
            "retrieval_stats": {
                "total_retrieved": 6,
                "by_type": {"text": 4, "table": 1, "image": 1},
                "collection": "diagnostic_rag",
                "top_k_requested": 6,
                "llm_input_tokens": 1842,
                "llm_output_tokens": 387,
                "llm_latency_ms": 2140.5,
            },
            "latency_ms": 2341.2,
            "model": "claude-sonnet-4-6",
        }
    }}


# ══════════════════════════════════════════════════════════════════════════════
# /documents
# ══════════════════════════════════════════════════════════════════════════════

class DocumentInfo(BaseModel):
    filename: str
    chunks: dict[str, int]   # {"text": N, "table": N, "image": N}
    total_chunks: int


class DocumentsResponse(BaseModel):
    collection: str
    document_count: int
    documents: list[DocumentInfo]


# ══════════════════════════════════════════════════════════════════════════════
# /delete
# ══════════════════════════════════════════════════════════════════════════════

class DeleteRequest(BaseModel):
    filename: str = Field(..., description="Exact filename to delete from the index.")
    collection: str = Field("diagnostic_rag")


class DeleteResponse(BaseModel):
    status: Literal["deleted", "not_found"]
    filename: str
    chunks_removed: int
    message: str


# ══════════════════════════════════════════════════════════════════════════════
# Error
# ══════════════════════════════════════════════════════════════════════════════

class ErrorResponse(BaseModel):
    detail: str
    error_type: str = "InternalError"
