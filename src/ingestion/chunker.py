"""
src/ingestion/chunker.py
─────────────────────────
Splits long text chunks into smaller overlapping windows.

Table chunks and image-summary chunks are kept whole (they are
already self-contained units).  Only raw text chunks are split.
"""

from __future__ import annotations

import re
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings
from src.ingestion.pdf_parser import ParsedChunk


class TextChunker:
    """
    Applies recursive character splitting to text chunks.
    Table and image chunks pass through unchanged.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        cfg = get_settings()
        # LangChain splitter uses character counts; 1 token ≈ 4 chars
        size = (chunk_size or cfg.chunk_size) * 4
        overlap = (chunk_overlap or cfg.chunk_overlap) * 4

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            keep_separator=True,
            length_function=len,
        )

    def split(self, chunks: list[ParsedChunk]) -> list[ParsedChunk]:
        """
        Apply chunking.  Returns a new list — input is not mutated.
        Text chunks that exceed chunk_size are split;
        table and image chunks are returned as-is.
        """
        result: list[ParsedChunk] = []

        for chunk in chunks:
            if chunk.chunk_type != "text":
                result.append(chunk)
                continue

            sub_texts = self._splitter.split_text(chunk.text)

            if len(sub_texts) <= 1:
                result.append(chunk)
                continue

            for sub_idx, sub_text in enumerate(sub_texts):
                sub_text = sub_text.strip()
                if not sub_text:
                    continue

                new_id = f"{chunk.chunk_id}_sub{sub_idx:03d}"
                new_meta = dict(chunk.metadata)
                new_meta["sub_index"] = sub_idx
                new_meta["parent_chunk_id"] = chunk.chunk_id

                result.append(
                    ParsedChunk(
                        chunk_id=new_id,
                        source=chunk.source,
                        page=chunk.page,
                        chunk_type="text",
                        text=sub_text,
                        metadata=new_meta,
                    )
                )

        return result
