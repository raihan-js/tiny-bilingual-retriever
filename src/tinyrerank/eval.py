"""Retrieval evaluation: nDCG@k, Recall@k, and CPU latency.

All metrics are computed on a fixed subsample so models are compared against
each other on identical data. Absolute nDCG values are inflated by subsampling;
only relative comparisons are meaningful.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RetrievalResult:
    """Metrics for one model on one eval set."""

    model: str
    eval_set: str
    n_queries: int
    n_passages: int
    ndcg_at_10: float
    recall_at_100: float
    latency_p50_ms: float
    latency_p95_ms: float
    index_mb: float
    extra: dict = field(default_factory=dict)


def dcg(relevances: np.ndarray, k: int = 10) -> float:
    """Discounted cumulative gain at k."""
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(relevances * discounts))


def ndcg_at_k(ranked_relevances: list[np.ndarray], k: int = 10) -> float:
    """Mean nDCG@k across queries."""
    if not ranked_relevances:
        return 0.0
    scores = []
    for rel in ranked_relevances:
        ideal = np.sort(rel)[::-1]
        idcg = dcg(ideal, k)
        scores.append(dcg(rel, k) / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores))


def recall_at_k(ranked_relevances: list[np.ndarray], k: int = 100) -> float:
    """Mean Recall@k: fraction of relevant passages retrieved in top-k."""
    if not ranked_relevances:
        return 0.0
    scores = []
    for rel in ranked_relevances:
        n_rel = int(np.sum(rel > 0))
        if n_rel == 0:
            continue
        scores.append(float(np.sum(rel[:k] > 0)) / n_rel)
    return float(np.mean(scores)) if scores else 0.0


def compute_metrics(ranked_relevances: list[np.ndarray],
                    latencies_ms: list[float] | None = None) -> dict:
    """Compute nDCG@10, Recall@100, and latency percentiles."""
    out = {
        "ndcg_at_10": ndcg_at_k(ranked_relevances, 10),
        "recall_at_100": recall_at_k(ranked_relevances, 100),
    }
    if latencies_ms:
        arr = np.asarray(latencies_ms)
        out["latency_p50_ms"] = float(np.percentile(arr, 50))
        out["latency_p95_ms"] = float(np.percentile(arr, 95))
    return out


def cosine_search(query_embeddings: np.ndarray, passage_embeddings: np.ndarray,
                  k: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force cosine similarity search. Returns (indices, scores)."""
    q = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-12)
    p = passage_embeddings / (np.linalg.norm(passage_embeddings, axis=1, keepdims=True) + 1e-12)
    scores = q @ p.T
    k = min(k, scores.shape[1])
    indices = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    top_scores = np.take_along_axis(scores, indices, axis=1)
    order = np.argsort(-top_scores, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    top_scores = np.take_along_axis(top_scores, order, axis=1)
    return indices, top_scores


def timed_search(query_embeddings: np.ndarray, passage_embeddings: np.ndarray,
                 k: int = 100) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Cosine search with per-query latency measurement."""
    latencies = []
    all_indices = []
    all_scores = []
    for i in range(len(query_embeddings)):
        t0 = time.perf_counter()
        idx, sc = cosine_search(query_embeddings[i:i + 1], passage_embeddings, k)
        latencies.append((time.perf_counter() - t0) * 1000)
        all_indices.append(idx[0])
        all_scores.append(sc[0])
    return np.array(all_indices), np.array(all_scores), latencies


def build_relevance_judgments(q_ids: list, ranked_indices: np.ndarray,
                             passage_ids: list,
                             qid_to_relevant: dict[str, set[str]]) -> list[np.ndarray]:
    """Build per-query relevance arrays from ranked indices and q-p relevance."""
    judgments = []
    for i, qid in enumerate(q_ids):
        relevant = qid_to_relevant.get(qid, set())
        rel = np.array([1.0 if passage_ids[pidx] in relevant else 0.0
                        for pidx in ranked_indices[i]])
        judgments.append(rel)
    return judgments
