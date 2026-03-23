"""
src/ingestion/pdf_parser.py
────────────────────────────
Multimodal PDF parser.

Extracts three chunk types from every PDF page:
  ┌─────────────┬───────────────────────────────────────────────────┐
  │ Chunk type  │ Extraction method                                  │
  ├─────────────┼───────────────────────────────────────────────────┤
  │ text        │ PyMuPDF (fitz) page.get_text()                    │
  │ table       │ pdfplumber page.extract_tables() → Markdown        │
  │ image       │ PyMuPDF page.get_images() → PIL → VLM summary      │
  └─────────────┴───────────────────────────────────────────────────┘

Images are processed through the VLM *during parsing* so that the
resulting chunk.text field contains a rich natural-language description
ready for embedding.  Raw pixel data is NOT stored in the index.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

from src.models.vision import get_vision_model

# Minimum image area (px²) to process — skips decorative dots, rules, etc.
MIN_IMAGE_AREA = 80 * 80


# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParsedChunk:
    """A single extracted unit from a PDF — text, table, or image."""
    chunk_id: str            # unique deterministic ID
    source: str              # PDF filename
    page: int                # 1-indexed page number
    chunk_type: str          # "text" | "table" | "image"
    text: str                # embeddable text content
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page,
            "chunk_type": self.chunk_type,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class ParseResult:
    """Aggregated result of parsing one PDF."""
    filename: str
    total_pages: int
    text_chunks: list[ParsedChunk]
    table_chunks: list[ParsedChunk]
    image_chunks: list[ParsedChunk]
    errors: list[str] = field(default_factory=list)

    @property
    def all_chunks(self) -> list[ParsedChunk]:
        return self.text_chunks + self.table_chunks + self.image_chunks

    @property
    def counts(self) -> dict[str, int]:
        return {
            "text": len(self.text_chunks),
            "table": len(self.table_chunks),
            "image": len(self.image_chunks),
            "total": len(self.all_chunks),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Parser
# ══════════════════════════════════════════════════════════════════════════════

class PDFParser:
    """
    Parses a PDF file into multimodal chunks.

    Example:
        parser = PDFParser()
        result = parser.parse(Path("sample_documents/Diagnostic_Document.pdf"))
        for chunk in result.all_chunks:
            print(chunk.chunk_type, chunk.page, chunk.text[:80])
    """

    def __init__(self, use_vision: bool = True):
        self._use_vision = use_vision
        self._vision = get_vision_model() if use_vision else None

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self, pdf_path: Path) -> ParseResult:
        """Parse a PDF, returning all extracted chunks."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        filename = pdf_path.name
        text_chunks: list[ParsedChunk] = []
        table_chunks: list[ParsedChunk] = []
        image_chunks: list[ParsedChunk] = []
        errors: list[str] = []
        total_pages = 0

        # Open with both libraries simultaneously
        with fitz.open(str(pdf_path)) as fitz_doc, \
             pdfplumber.open(str(pdf_path)) as plumber_doc:

            total_pages = len(fitz_doc)

            for page_idx in range(total_pages):
                page_num = page_idx + 1
                fitz_page = fitz_doc[page_idx]
                plumber_page = plumber_doc.pages[page_idx]

                # ── Text ──────────────────────────────────────────────────
                try:
                    txt = self._extract_text(fitz_page, filename, page_num,
                                             len(text_chunks))
                    text_chunks.extend(txt)
                except Exception as exc:
                    errors.append(f"Page {page_num} text: {exc}")

                # ── Tables ────────────────────────────────────────────────
                try:
                    tabs = self._extract_tables(plumber_page, filename, page_num,
                                                len(table_chunks))
                    table_chunks.extend(tabs)
                except Exception as exc:
                    errors.append(f"Page {page_num} tables: {exc}")

                # ── Images ────────────────────────────────────────────────
                try:
                    imgs = self._extract_images(fitz_doc, fitz_page, filename,
                                                page_num, len(image_chunks))
                    image_chunks.extend(imgs)
                except Exception as exc:
                    errors.append(f"Page {page_num} images: {exc}")

        return ParseResult(
            filename=filename,
            total_pages=total_pages,
            text_chunks=text_chunks,
            table_chunks=table_chunks,
            image_chunks=image_chunks,
            errors=errors,
        )

    # ── Text extraction ───────────────────────────────────────────────────────

    def _extract_text(
        self,
        page: fitz.Page,
        source: str,
        page_num: int,
        offset: int,
    ) -> list[ParsedChunk]:
        """Extract and clean raw text from a PDF page."""
        raw = page.get_text("text")
        cleaned = self._clean_text(raw)

        if len(cleaned) < 30:
            return []

        chunk_id = f"text_{source}_{page_num:04d}_{offset:04d}"
        return [
            ParsedChunk(
                chunk_id=chunk_id,
                source=source,
                page=page_num,
                chunk_type="text",
                text=cleaned,
                metadata={
                    "source": source,
                    "page": page_num,
                    "chunk_type": "text",
                    "char_count": len(cleaned),
                },
            )
        ]

    # ── Table extraction ──────────────────────────────────────────────────────

    def _extract_tables(
        self,
        page: pdfplumber.page.Page,
        source: str,
        page_num: int,
        offset: int,
    ) -> list[ParsedChunk]:
        """Extract tables from a page and convert to Markdown."""
        tables = page.extract_tables()
        chunks = []

        for t_idx, raw_table in enumerate(tables):
            if not self._table_is_valid(raw_table):
                continue

            md = self._table_to_markdown(raw_table)
            if not md.strip():
                continue

            chunk_id = f"table_{source}_{page_num:04d}_{offset + t_idx:04d}"
            n_rows = len(raw_table)
            n_cols = len(raw_table[0]) if raw_table else 0

            chunks.append(
                ParsedChunk(
                    chunk_id=chunk_id,
                    source=source,
                    page=page_num,
                    chunk_type="table",
                    text=md,
                    metadata={
                        "source": source,
                        "page": page_num,
                        "chunk_type": "table",
                        "rows": n_rows,
                        "cols": n_cols,
                    },
                )
            )

        return chunks

    # ── Image extraction + VLM summarisation ──────────────────────────────────

    def _extract_images(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        source: str,
        page_num: int,
        offset: int,
    ) -> list[ParsedChunk]:
        """
        Extract images from a PDF page, pass each through the VLM,
        and return chunks whose .text field is the VLM-generated summary.
        """
        image_list = page.get_images(full=True)
        chunks = []
        img_idx = 0

        for xref_info in image_list:
            xref = xref_info[0]
            try:
                pil_img = self._xref_to_pil(doc, xref)
            except Exception:
                continue

            if pil_img is None:
                continue

            w, h = pil_img.size
            if w * h < MIN_IMAGE_AREA:
                continue

            # ── VLM summarisation ─────────────────────────────────────────
            context_hint = (
                f"Page {page_num} of {source}. "
                f"Image dimensions: {w}×{h} pixels."
            )

            if self._use_vision and self._vision:
                summary = self._vision.summarise(pil_img, context=context_hint)
            else:
                summary = (
                    f"[Image on page {page_num} of {source}, {w}×{h}px. "
                    "VLM summarisation disabled.]"
                )

            chunk_id = f"image_{source}_{page_num:04d}_{offset + img_idx:04d}"
            chunks.append(
                ParsedChunk(
                    chunk_id=chunk_id,
                    source=source,
                    page=page_num,
                    chunk_type="image",
                    text=summary,   # ← VLM description, not raw pixels
                    metadata={
                        "source": source,
                        "page": page_num,
                        "chunk_type": "image",
                        "width": w,
                        "height": h,
                        "vlm_model": (
                            self._vision._model if self._vision else "none"
                        ),
                    },
                )
            )
            img_idx += 1

        return chunks

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _xref_to_pil(doc: fitz.Document, xref: int) -> Optional[Image.Image]:
        """Decode a PDF image by its xref to a PIL Image."""
        try:
            base = doc.extract_image(xref)
            data = base["image"]
            img = Image.open(io.BytesIO(data))
            if img.mode not in ("RGB", "RGBA", "L"):
                img = img.convert("RGB")
            return img
        except Exception:
            return None

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalise whitespace and strip non-ASCII control characters."""
        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\xA0-\uFFFF]", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _table_is_valid(table: list[list]) -> bool:
        if not table or len(table) < 2:
            return False
        if not table[0] or len(table[0]) < 2:
            return False
        non_empty = sum(
            1 for row in table for cell in row if cell and str(cell).strip()
        )
        return non_empty >= 4

    @staticmethod
    def _table_to_markdown(table: list[list]) -> str:
        """Convert pdfplumber table (list-of-lists) to GitHub Markdown."""
        cleaned: list[list[str]] = []
        for row in table:
            cleaned.append(
                [re.sub(r"\s+", " ", str(c)).strip() if c else "" for c in row]
            )

        if not cleaned:
            return ""

        n_cols = max(len(r) for r in cleaned)
        for row in cleaned:
            while len(row) < n_cols:
                row.append("")

        lines = [
            "| " + " | ".join(cleaned[0]) + " |",
            "| " + " | ".join(["---"] * n_cols) + " |",
        ]
        for row in cleaned[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
