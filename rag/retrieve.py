"""Retrieval = Qdrant search (candidate pool) + optional cross-encoder rerank."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from . import config as cfg
from . import store


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(cfg.RERANK_MODEL)


def rerank(query: str, hits: list[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
    if not hits:
        return hits
    scores = _cross_encoder().predict([(query, h["text"]) for h in hits])
    out = sorted(({**h, "rerank_score": float(s)} for h, s in zip(hits, scores)),
                 key=lambda h: h["rerank_score"], reverse=True)
    return out[:top_k] if top_k else out


def search(query: str, top_k: int = 5, candidates: int = 30, mode: str = "hybrid",
           use_rerank: bool = True, source: str | None = None) -> list[dict[str, Any]]:
    """Fetch `candidates` from Qdrant, rerank them, return the best `top_k`."""
    pool = max(candidates, top_k) if use_rerank else top_k
    hits = store.search(query, limit=pool, mode=mode, prefetch=max(pool, 30), source=source)
    return rerank(query, hits, top_k) if use_rerank else hits[:top_k]
