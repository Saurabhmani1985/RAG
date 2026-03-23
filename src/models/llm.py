"""
src/models/llm.py
──────────────────
LLM wrapper using Claude for RAG answer generation.

Contains the custom prompt template that structures retrieved context
(text chunks, table chunks, image summaries) into a coherent prompt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM PROMPT TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an expert automotive diagnostic engineer specialising in \
common-rail diesel engine ECU (Engine Control Unit) fault diagnosis and repair.

You have access to retrieved excerpts from a comprehensive ECU Diagnostic Manual covering:
- Fault codes (P-codes / DTCs) with descriptions and lamp status (CE, MIL, WIF)
- Sensor wiring diagrams and ECU pin connections
- Recovery modes activated by each fault
- Step-by-step diagnostic procedures (what to check, how to rectify)
- Engine specifications and component technical details
- High-pressure and low-pressure fuel system diagnostics

## Instructions for answering:
1. Base your answer EXCLUSIVELY on the provided context. Do not invent information.
2. If a fault code is relevant, always state: the code, fault name, lamp status, and recovery mode.
3. For diagnostic procedures, present checks as a numbered list in the correct order.
4. If a wiring diagram or image description is provided, reference it explicitly.
5. If a table is in the context, extract and present the relevant rows.
6. If the answer is NOT in the context, state: "This information is not present in the retrieved sections of the diagnostic manual."
7. Use technical terminology appropriate for a qualified mechanic.
8. Format fault codes in backticks, e.g. `P0087`.

## Response format:
- Start with a direct answer to the question (1–2 sentences).
- Then provide supporting detail (fault description, checks, wiring, etc.).
- End with a "Sources" note listing which pages/chunks informed the answer."""

RAG_PROMPT_TEMPLATE = """## Retrieved Context

{context}

---

## Question

{question}

## Answer:"""


@dataclass
class LLMResponse:
    answer: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str


class LLMClient:
    """
    Claude-based LLM client for RAG generation.

    Usage:
        llm = LLMClient()
        response = llm.generate(question="What does P0087 mean?", context=ctx_str)
    """

    def __init__(self):
        cfg = get_settings()
        self._client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        self._model = cfg.llm_model
        self._max_tokens = cfg.max_tokens_response
        self._temperature = cfg.temperature

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(self, question: str, context: str) -> LLMResponse:
        """
        Generate an answer using the RAG prompt template.

        Parameters
        ----------
        question : str
            The user's natural language question.
        context  : str
            Pre-assembled context string from retrieved chunks.

        Returns
        -------
        LLMResponse with the generated answer and token usage.
        """
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        t0 = time.perf_counter()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        return LLMResponse(
            answer=response.content[0].text.strip(),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=round(latency_ms, 1),
            model=response.model,
        )

    def format_context(self, chunks: list[dict]) -> str:
        """
        Convert a list of retrieved chunk dicts into the context string
        that gets injected into the RAG prompt template.

        Each chunk dict has: {text, chunk_type, source, page, score}
        """
        if not chunks:
            return "No relevant context was retrieved."

        sections: list[str] = []

        # Group by chunk type for clarity
        text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
        table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
        image_chunks = [c for c in chunks if c["chunk_type"] == "image"]

        if text_chunks:
            sections.append("### Text Excerpts")
            for i, c in enumerate(text_chunks, 1):
                sections.append(
                    f"[Excerpt {i} | {c['source']} | Page {c['page']} | "
                    f"Relevance: {c['score']:.2f}]\n{c['text']}"
                )

        if table_chunks:
            sections.append("\n### Table Data")
            for i, c in enumerate(table_chunks, 1):
                sections.append(
                    f"[Table {i} | {c['source']} | Page {c['page']} | "
                    f"Relevance: {c['score']:.2f}]\n{c['text']}"
                )

        if image_chunks:
            sections.append("\n### Diagram / Image Descriptions (VLM-generated)")
            for i, c in enumerate(image_chunks, 1):
                sections.append(
                    f"[Diagram {i} | {c['source']} | Page {c['page']} | "
                    f"Relevance: {c['score']:.2f}]\n{c['text']}"
                )

        return "\n\n".join(sections)


# ── Singleton ─────────────────────────────────────────────────────────────────
_llm: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm
