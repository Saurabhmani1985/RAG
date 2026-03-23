#!/usr/bin/env python3
"""
scripts/ingest_pdf.py
──────────────────────
Command-line interface for ingesting PDF documents into the RAG system.
Useful for batch ingestion, CI pipelines, or pre-populating the vector
store before starting the FastAPI server.

Usage:
    # Ingest the sample document (vision enabled)
    python scripts/ingest_pdf.py sample_documents/Diagnostic_Document.pdf

    # Ingest into a specific collection, skip VLM image processing
    python scripts/ingest_pdf.py path/to/manual.pdf --collection my_docs --no-vision

    # Ingest all PDFs in a folder
    python scripts/ingest_pdf.py docs/ --collection fleet_manuals

    # Dry-run: parse only, no indexing
    python scripts/ingest_pdf.py sample_documents/Diagnostic_Document.pdf --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def print_banner():
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║           Multimodal RAG — PDF Ingestion CLI                 ║\n"
        "╚══════════════════════════════════════════════════════════════╝"
    )


def print_result_table(result) -> None:
    """Pretty-print ingestion stats."""
    counts = result.chunk_counts
    width = 58

    print(f"\n{'─' * width}")
    print(f"  ✓  Ingestion complete")
    print(f"{'─' * width}")
    print(f"  File          : {result.filename}")
    print(f"  Collection    : {result.collection}")
    print(f"  Pages         : {result.total_pages}")
    print(f"{'─' * width}")
    print(f"  Chunks        : {counts.get('total', 0):>6}  total")
    print(f"    ├── text    : {counts.get('text', 0):>6}")
    print(f"    ├── table   : {counts.get('table', 0):>6}")
    print(f"    └── image   : {counts.get('image', 0):>6}  (VLM-summarised)")
    print(f"{'─' * width}")
    print(f"  Time          : {result.processing_time_s:.1f}s")

    if result.errors:
        print(f"\n  ⚠  {len(result.errors)} warning(s):")
        for e in result.errors[:5]:
            print(f"     • {e}")
        if len(result.errors) > 5:
            print(f"     … and {len(result.errors) - 5} more")

    print(f"{'─' * width}\n")


def ingest_file(pdf_path: Path, collection: str, use_vision: bool, dry_run: bool) -> bool:
    """Ingest a single PDF. Returns True on success."""
    if not pdf_path.exists():
        print(f"  ✗  File not found: {pdf_path}", file=sys.stderr)
        return False

    if not pdf_path.suffix.lower() == ".pdf":
        print(f"  ✗  Not a PDF: {pdf_path}", file=sys.stderr)
        return False

    print(f"\n  → Processing: {pdf_path.name}")
    print(f"     Collection : {collection}")
    print(f"     Vision VLM : {'enabled' if use_vision else 'disabled (--no-vision)'}")

    if dry_run:
        # Parse only, no embedding or indexing
        from src.ingestion.pdf_parser import PDFParser

        parser = PDFParser(use_vision=False)
        t0 = time.perf_counter()
        result = parser.parse(pdf_path)
        elapsed = time.perf_counter() - t0

        print(f"\n  [DRY RUN — no data written to vector store]")
        print(f"  Pages   : {result.total_pages}")
        print(f"  Text    : {len(result.text_chunks)} raw chunks")
        print(f"  Tables  : {len(result.table_chunks)} chunks")
        print(f"  Images  : {len(result.image_chunks)} found")
        if result.errors:
            print(f"  Errors  : {len(result.errors)}")
        print(f"  Parsed in {elapsed:.2f}s\n")
        return True

    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(collection=collection, use_vision=use_vision)
    result = pipeline.run(pdf_path)
    print_result_table(result)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the Multimodal RAG vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a PDF file or directory of PDFs.",
    )
    parser.add_argument(
        "--collection",
        default="diagnostic_rag",
        help="ChromaDB collection name (default: diagnostic_rag)",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Skip Claude Vision image summarisation (faster, no image retrieval).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse only — do not embed or write to the vector store.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=Path(".env"),
        help="Path to .env file (default: .env in current directory)",
    )

    args = parser.parse_args()

    # Load environment
    if args.env.exists():
        from dotenv import load_dotenv
        load_dotenv(args.env)
        print(f"  Loaded environment from: {args.env}")
    else:
        print(f"  ⚠  No .env file found at {args.env}. Relying on shell environment.")

    print_banner()

    # Collect PDF files to process
    target = args.path
    if target.is_dir():
        pdfs = sorted(target.glob("*.pdf"))
        if not pdfs:
            print(f"\n  ✗  No PDF files found in: {target}", file=sys.stderr)
            sys.exit(1)
        print(f"\n  Found {len(pdfs)} PDF(s) in {target}")
    elif target.is_file():
        pdfs = [target]
    else:
        print(f"\n  ✗  Path does not exist: {target}", file=sys.stderr)
        sys.exit(1)

    # Process each file
    success_count = 0
    for pdf_path in pdfs:
        ok = ingest_file(
            pdf_path=pdf_path,
            collection=args.collection,
            use_vision=not args.no_vision,
            dry_run=args.dry_run,
        )
        if ok:
            success_count += 1

    print(f"  Done: {success_count}/{len(pdfs)} file(s) ingested.")

    if success_count < len(pdfs):
        sys.exit(1)


if __name__ == "__main__":
    main()
