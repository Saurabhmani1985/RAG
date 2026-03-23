"""
tests/test_api.py
──────────────────
FastAPI endpoint integration tests using httpx AsyncClient.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# Set minimal env before importing app
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-placeholder")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/test_chroma")

from main import app  # noqa: E402

SAMPLE_PDF = Path(__file__).parent.parent / "sample_documents" / "Diagnostic_Document.pdf"


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


# ── /health ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_200(client):
    resp = await client.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_response_schema(client):
    resp = await client.get("/health")
    data = resp.json()
    assert "status" in data
    assert "version" in data
    assert "uptime_seconds" in data
    assert "indexed_documents" in data
    assert "total_chunks" in data
    assert "components" in data


# ── /docs ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_swagger_ui_accessible(client):
    resp = await client.get("/docs")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_openapi_json_accessible(client):
    resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    # Verify all required endpoints are documented
    paths = schema["paths"]
    assert "/health" in paths
    assert "/ingest" in paths
    assert "/query" in paths
    assert "/documents" in paths
    assert "/delete" in paths


# ── /ingest (validation) ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_rejects_non_pdf(client):
    resp = await client.post(
        "/ingest",
        files={"file": ("test.txt", b"hello world", "text/plain")},
        data={"collection": "test"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_ingest_rejects_empty_file(client):
    resp = await client.post(
        "/ingest",
        files={"file": ("empty.pdf", b"", "application/pdf")},
        data={"collection": "test"},
    )
    assert resp.status_code == 400


# ── /query (validation) ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_query_requires_question(client):
    resp = await client.post("/query", json={})
    assert resp.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_query_short_question_rejected(client):
    resp = await client.post("/query", json={"question": "ab"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_query_empty_collection_returns_404(client):
    resp = await client.post(
        "/query",
        json={
            "question": "What does P0087 mean?",
            "collection": "nonexistent_collection_xyz",
        },
    )
    assert resp.status_code == 404


# ── /documents ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_documents_returns_200(client):
    resp = await client.get("/documents?collection=test_collection_xyz")
    assert resp.status_code == 200
    data = resp.json()
    assert "documents" in data
    assert "document_count" in data


# ── /delete ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_nonexistent_returns_not_found(client):
    resp = await client.request(
        "DELETE",
        "/delete",
        json={
            "filename": "nonexistent_file.pdf",
            "collection": "test",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "not_found"
    assert data["chunks_removed"] == 0


# ── Root ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_root_redirects_to_info(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "docs" in data
