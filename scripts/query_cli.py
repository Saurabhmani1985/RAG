#!/usr/bin/env python3
"""
scripts/query_cli.py
─────────────────────
Interactive command-line interface for querying the RAG system.
Useful for testing without running the full FastAPI server.

Usage:
    # Interactive REPL
    python scripts/query_cli.py

    # Single question
    python scripts/query_cli.py --question "What does P0087 mean?"

    # Filter to table chunks only
    python scripts/query_cli.py --question "List faults in STOP_ENGINE group" --type table

    # Adjust top-k
    python scripts/query_cli.py --top-k 10
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path


CHUNK_TYPE_COLORS = {
    "text":  "\033[94m",   # blue
    "table": "\033[93m",   # yellow
    "image": "\033[95m",   # magenta
}
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"


def print_result(result, show_sources: bool = True) -> None:
    print(f"\n{BOLD}{'─' * 70}{RESET}")
    print(f"{BOLD}Answer:{RESET}")
    print()

    # Word-wrap the answer
    for line in result.answer.split("\n"):
        if line.strip():
            print(textwrap.fill(line, width=70, initial_indent="  ", subsequent_indent="  "))
        else:
            print()

    if show_sources and result.sources:
        print(f"\n{BOLD}Sources ({len(result.sources)}):{RESET}")
        for i, src in enumerate(result.sources, 1):
            color = CHUNK_TYPE_COLORS.get(src.filename, "")  # fallback
            ct_color = CHUNK_TYPE_COLORS.get(src.chunk_type, "")
            print(
                f"  [{i}] {ct_color}{src.chunk_type.upper():6}{RESET} "
                f"page {src.page:>3}  score {src.score:.3f}  "
                f"\033[90m{src.filename}\033[0m"
            )
            excerpt = src.excerpt[:100].replace("\n", " ")
            print(f"       \033[90m{excerpt}…\033[0m")

    rs = result.retrieval_stats
    print(f"\n{BOLD}Stats:{RESET}")
    print(f"  Retrieved: {rs['total_retrieved']} chunks  {rs.get('by_type', {})}")
    print(f"  Latency  : {result.latency_ms:.0f}ms total  ({rs.get('llm_latency_ms', 0):.0f}ms LLM)")
    print(f"  Model    : {result.model}")
    print(f"{'─' * 70}")


def run_query(question: str, collection: str, top_k: int, chunk_type: str | None) -> None:
    from src.retrieval.retriever import get_retriever

    print(f"\n  {BOLD}Query:{RESET} {question[:80]}")
    print(f"  Collection: {collection}  |  top_k: {top_k}  |  filter: {chunk_type or 'all'}")
    print("  Retrieving and generating…", end="", flush=True)

    try:
        retriever = get_retriever()
        result = retriever.query(
            question=question,
            top_k=top_k,
            collection=collection,
            chunk_type_filter=chunk_type,
        )
        print(f"\r  {GREEN}✓{RESET} Generated in {result.latency_ms:.0f}ms")
        print_result(result)
    except Exception as exc:
        print(f"\r  {RED}✗ Error: {exc}{RESET}")
        raise


def interactive_repl(collection: str, top_k: int, chunk_type: str | None) -> None:
    print(f"\n{BOLD}Multimodal RAG — Interactive Query{RESET}")
    print(f"Collection: {collection}  |  top_k: {top_k}  |  filter: {chunk_type or 'all'}")
    print("Type your question and press Enter. Type 'quit' or Ctrl-C to exit.\n")

    while True:
        try:
            question = input(f"{GREEN}❯{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        try:
            run_query(question, collection, top_k, chunk_type)
        except Exception:
            pass  # Error already printed; continue REPL


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool for querying the Multimodal RAG system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--question", "-q", type=str, help="Single question (omit for interactive mode)")
    parser.add_argument("--collection", default="diagnostic_rag")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--type", choices=["text", "table", "image"],
        default=None, dest="chunk_type",
        help="Filter retrieval to a specific chunk type",
    )
    parser.add_argument("--env", type=Path, default=Path(".env"))

    args = parser.parse_args()

    # Load environment
    if args.env.exists():
        from dotenv import load_dotenv
        load_dotenv(args.env)

    if args.question:
        run_query(args.question, args.collection, args.top_k, args.chunk_type)
    else:
        interactive_repl(args.collection, args.top_k, args.chunk_type)


if __name__ == "__main__":
    main()
