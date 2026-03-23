"""
src/models/vision.py
────────────────────
Vision Language Model (VLM) wrapper using Claude's vision capability.

Pipeline:
  PIL Image → base64 encode → Claude Vision API → text summary

This satisfies the requirement that "images must be processed through a
VLM to generate text summaries before embedding."  We never embed raw
pixel data; we embed the VLM-generated description.
"""

from __future__ import annotations

import base64
import io
import threading
import time
from typing import Optional

import anthropic
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings

_lock = threading.Lock()


# ── Prompt for image summarisation ───────────────────────────────────────────
IMAGE_SUMMARY_PROMPT = """You are analysing a page from a diesel engine ECU diagnostic manual.

Describe this image in detail, covering:
1. What type of diagram or figure this is (e.g., wiring diagram, circuit schematic, table, flowchart, sensor diagram)
2. The specific components shown (ECU pin numbers, sensor names, relay labels, connector labels)
3. The connections or relationships depicted (which ECU pin connects to which sensor/actuator terminal)
4. Any labels, numbers, or annotations visible in the image
5. The diagnostic or technical significance of what is shown

Be precise and technical. Your description will be used to answer fault diagnosis questions,
so include all numeric pin numbers, component names, and connection details you can see.
If the image contains a table or structured data, extract and list its contents.
If it is a circuit diagram, trace and describe the electrical connections.

Output a coherent paragraph (or structured list if a table) that captures all technical details."""


class VisionModel:
    """
    Wraps Claude's vision API to summarise images extracted from PDFs.

    Usage:
        vm = VisionModel()
        summary = vm.summarise(pil_image, context="Page 22, AMF Sensor Signal Fault")
    """

    def __init__(self):
        cfg = get_settings()
        self._client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        self._model = cfg.vision_model
        self._call_count = 0
        self._total_time = 0.0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        reraise=True,
    )
    def summarise(self, image: Image.Image, context: str = "") -> str:
        """
        Call Claude Vision to generate a text summary of a PIL image.

        Parameters
        ----------
        image   : PIL.Image.Image
        context : Optional hint (e.g., page number, fault name) prepended to prompt

        Returns
        -------
        str — rich textual description of the image
        """
        b64 = self._pil_to_b64(image)
        media_type = "image/png"

        user_prompt = IMAGE_SUMMARY_PROMPT
        if context:
            user_prompt = f"Context: {context}\n\n{IMAGE_SUMMARY_PROMPT}"

        t0 = time.perf_counter()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
        )
        elapsed = time.perf_counter() - t0

        with _lock:
            self._call_count += 1
            self._total_time += elapsed

        return response.content[0].text.strip()

    def summarise_batch(
        self,
        images: list[Image.Image],
        contexts: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Summarise multiple images sequentially (respects API rate limits).
        Returns list of summaries in the same order as `images`.
        """
        if contexts is None:
            contexts = [""] * len(images)

        summaries = []
        for img, ctx in zip(images, contexts):
            try:
                summary = self.summarise(img, ctx)
            except Exception as exc:
                # Graceful degradation: store error message rather than crashing
                summary = f"[Image description unavailable: {exc}]"
            summaries.append(summary)

        return summaries

    @property
    def stats(self) -> dict:
        return {
            "calls": self._call_count,
            "total_time_s": round(self._total_time, 2),
            "avg_time_s": round(self._total_time / max(self._call_count, 1), 2),
        }

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pil_to_b64(image: Image.Image) -> str:
        """Convert PIL image to base64 PNG string."""
        # Convert to RGB (handles RGBA, CMYK, palette modes)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        # Resize if very large (Claude vision accepts up to ~2000px, but costs more)
        max_dim = 1568
        w, h = image.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Singleton ─────────────────────────────────────────────────────────────────
_vision_model: Optional[VisionModel] = None


def get_vision_model() -> VisionModel:
    global _vision_model
    if _vision_model is None:
        _vision_model = VisionModel()
    return _vision_model
