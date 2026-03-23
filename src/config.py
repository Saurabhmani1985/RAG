"""
src/config.py
─────────────
All configuration via pydantic-settings.
Values loaded from .env file or environment variables.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Anthropic ────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(..., description="Anthropic API key")
    vision_model: str = Field("claude-opus-4-5")
    llm_model: str = Field("claude-sonnet-4-6")
    max_tokens_response: int = Field(2048, ge=256, le=8192)
    temperature: float = Field(0.1, ge=0.0, le=1.0)

    # ── Embedding ────────────────────────────────────────────────────────────
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = Field(384)  # MiniLM-L6-v2 produces 384-d vectors

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_persist_dir: Path = Field(Path("./data/chroma"))
    default_collection: str = Field("diagnostic_rag")

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(800, ge=100, le=4000)
    chunk_overlap: int = Field(100, ge=0, le=500)

    # ── Retrieval ────────────────────────────────────────────────────────────
    default_top_k: int = Field(6, ge=1, le=20)

    # ── Server ───────────────────────────────────────────────────────────────
    host: str = Field("0.0.0.0")
    port: int = Field(8000)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO")

    @field_validator("chroma_persist_dir", mode="before")
    @classmethod
    def make_dir(cls, v):
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
