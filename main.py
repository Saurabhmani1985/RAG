"""
main.py
────────
FastAPI application entry point for the Multimodal RAG system.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Or directly:
    python main.py
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("multimodal_rag")


# ── Startup / shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise heavy resources once at startup (warm embedding model cache,
    connect ChromaDB) so the first API call is not delayed.
    """
    log.info("═" * 60)
    log.info("  Multimodal RAG System — starting up")
    log.info("═" * 60)

    # Pre-warm embedding model (downloads from HuggingFace Hub on first run)
    try:
        from src.ingestion.embedder import get_embedder
        emb = get_embedder()
        log.info(f"  ✓ Embedding model ready  (dim={emb.dim})")
    except Exception as exc:
        log.warning(f"  ✗ Embedding model not ready: {exc}")

    # Pre-connect ChromaDB
    try:
        from src.retrieval.vector_store import get_vector_store
        store = get_vector_store()
        stats = store.get_stats()
        log.info(
            f"  ✓ ChromaDB ready  "
            f"({stats['total_chunks']} chunks in '{stats['collection']}')"
        )
    except Exception as exc:
        log.warning(f"  ✗ ChromaDB not ready: {exc}")

    log.info("  ✓ API server ready")
    log.info("═" * 60)

    yield  # ← application runs here

    log.info("Multimodal RAG System — shutting down")


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multimodal RAG System",
    description=(
        "## Multimodal Retrieval-Augmented Generation\n\n"
        "Ingests PDF documents containing **text**, **tables**, and **images**, "
        "builds a searchable vector index, and generates grounded answers via Claude.\n\n"
        "### Key features\n"
        "- Images are summarised by **Claude Vision (VLM)** before embedding\n"
        "- Unified semantic search across all chunk types (text, table, image)\n"
        "- Custom RAG prompt template for diagnostic domain reasoning\n"
        "- Source references with filename, page, chunk type, and relevance score\n\n"
        "### Quick start\n"
        "1. `POST /ingest` — upload `Diagnostic_Document.pdf`\n"
        "2. `POST /query` — ask a question\n"
        "3. `GET /documents` — list indexed documents\n"
        "4. `GET /health` — check system status"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",          # Swagger UI  ← required endpoint
    redoc_url="/redoc",        # ReDoc alternative
    openapi_url="/openapi.json",
    contact={
        "name": "Multimodal RAG",
        "url": "https://github.com/your-org/multimodal-rag",
    },
    license_info={"name": "MIT"},
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    log.info(
        f"{request.method} {request.url.path} → "
        f"{response.status_code} [{elapsed_ms} ms]"
    )
    return response


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "error_type": type(exc).__name__,
        },
    )


# ── Include routes ────────────────────────────────────────────────────────────
app.include_router(router, prefix="")


# ── Root redirect to docs ─────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse(
        content={
            "message": "Multimodal RAG System",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.config import get_settings
    cfg = get_settings()
    uvicorn.run(
        "main:app",
        host=cfg.host,
        port=cfg.port,
        reload=False,
        log_level=cfg.log_level.lower(),
        access_log=False,   # handled by our middleware
    )
