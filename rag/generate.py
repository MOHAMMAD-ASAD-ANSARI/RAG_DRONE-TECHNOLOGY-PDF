"""Answer generation with Gemini (free-tier friendly: only top-k chunks are sent)."""
from __future__ import annotations

import time
from typing import Any, Iterator

from . import config as cfg

PROMPT = """You are a precise and reliable AI assistant in a Retrieval-Augmented Generation system.
Answer the user's question using ONLY the context below.

RULES
1. Use only the provided context. Never use outside knowledge or guess.
2. Answer the actual question; ignore irrelevant context; combine relevant pieces coherently.
3. Stay faithful to the source. Paraphrase; do not copy large passages unless a quote is requested.
4. If the answer cannot be determined from the context, say exactly:
   "I don't have enough information in the provided context to answer that."
5. If the context conflicts, say so and explain what each part states.
6. Be concise. Use bullet points or numbered lists when they help.
7. Keep important details: numbers, dates, conditions, definitions, exceptions.
8. Do not mention "the context", "the documents" or the retrieval process.

<context>
{context}
</context>

<question>
{question}
</question>

Answer based strictly on the context above."""


def build_context(hits: list[dict[str, Any]]) -> str:
    return "\n\n---\n\n".join(h["text"] for h in hits)


def stream_answer(question: str, hits: list[dict[str, Any]]) -> Iterator[str]:
    if not cfg.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
    from google import genai

    client = genai.Client(api_key=cfg.GEMINI_API_KEY)
    prompt = PROMPT.format(context=build_context(hits), question=question)
    for attempt in range(3):
        started = False
        try:
            for chunk in client.models.generate_content_stream(model=cfg.GEN_MODEL, contents=prompt):
                if chunk.text:
                    started = True
                    yield chunk.text
            return
        except Exception as exc:  # noqa: BLE001
            busy = any(t in str(exc) for t in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))
            if started or not busy or attempt == 2:
                raise RuntimeError(f"Gemini request failed: {exc}") from exc
            time.sleep(5 * (attempt + 1))  # free-tier rate limit: back off and retry
