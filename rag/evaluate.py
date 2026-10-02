"""Retrieval metrics. Retrieves once per question and scores every K from the same ranking."""
from __future__ import annotations

import json
import math
import re
from typing import Any

from .retrieve import search


def _gt_ids(raw: list) -> set[int]:
    """Accepts 0-based ints (notebook style) or 'chunk_001' labels (1-based)."""
    out = set()
    for x in raw:
        if isinstance(x, int):
            out.add(x)
        elif isinstance(x, str) and (m := re.fullmatch(r"chunk_(\d+)", x.strip())):
            out.add(int(m.group(1)) - 1)
        elif isinstance(x, str) and x.strip().isdigit():
            out.add(int(x))
    return out


def hit_rate(ids, gt, k): return int(bool(set(ids[:k]) & gt))
def recall(ids, gt, k): return len(set(ids[:k]) & gt) / len(gt) if gt else 0.0
def precision(ids, gt, k): return sum(i in gt for i in ids[:k]) / k if k else 0.0


def mrr(ids, gt, k):
    for rank, i in enumerate(ids[:k], start=1):
        if i in gt:
            return 1 / rank
    return 0.0


def ndcg(ids, gt, k):
    dcg = sum(1 / math.log2(r + 2) for r, i in enumerate(ids[:k]) if i in gt)
    ideal = sum(1 / math.log2(r + 2) for r in range(min(len(gt), k)))  # standard ideal ranking
    return dcg / ideal if ideal else 0.0


def run_eval(dataset_path: str, ks=(1, 3, 5, 10, 20, 30), mode="hybrid",
             use_rerank=False, source: str | None = None) -> dict[int, dict[str, float]]:
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    if not dataset:
        raise ValueError("Evaluation dataset is empty.")
    pool = max(ks)
    acc = {k: {"Hit@K": 0.0, "Precision@K": 0.0, "Recall@K": 0.0, "MRR": 0.0, "nDCG@K": 0.0} for k in ks}

    for item in dataset:
        gt = _gt_ids(item["ground_truth_chunks"])
        hits = search(item["question"], top_k=pool, candidates=max(pool, 30),
                      mode=mode, use_rerank=use_rerank, source=source)
        ids = [h["chunk_id"] for h in hits]
        for k in ks:
            a = acc[k]
            a["Hit@K"] += hit_rate(ids, gt, k)
            a["Precision@K"] += precision(ids, gt, k)
            a["Recall@K"] += recall(ids, gt, k)
            a["MRR"] += mrr(ids, gt, k)
            a["nDCG@K"] += ndcg(ids, gt, k)
    n = len(dataset)
    return {k: {m: v / n for m, v in a.items()} for k, a in acc.items()}


def print_results(title: str, results: dict[int, dict[str, float]]) -> None:
    print(f"\n=== {title} ===")
    for k, scores in results.items():
        print(f"K={k:<3} " + "  ".join(f"{m}={v:.4f}" for m, v in scores.items()))
