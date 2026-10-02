#!/usr/bin/env python3
"""Hybrid fusion: RRF of MeCab-segmented BM25 and dense student embeddings.

Usage:
  PYTHONPATH=src python scripts/run_fusion.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tinyrerank.eval import (build_relevance_judgments, compute_metrics,
                             timed_search)
from tinyrerank.models import Encoder
from tinyrerank.data import load_jaqara, load_opus_en_ja

RESULTS = Path("data/results")


def mecab_segment(text: str) -> str:
    """Segment Japanese text into space-separated words using MeCab."""
    import fugashi
    tagger = fugashi.Tagger()
    return " ".join(word.surface for word in tagger(text))


def bm25_search(query_texts: list[str], passage_texts: list[str],
               k: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """BM25 search with MeCab segmentation for Japanese."""
    import bm25s

    # Segment passages
    seg_passages = [mecab_segment(t) for t in passage_texts]
    # Segment queries
    seg_queries = [mecab_segment(t) for t in query_texts]

    # Index
    tokens = bm25s.tokenize(seg_passages, stopwords=None)
    retriever = bm25s.BM25()
    retriever.index(tokens)

    # Search
    query_tokens = bm25s.tokenize(seg_queries, stopwords=None)
    results = retriever.retrieve(query_tokens, k=min(k, len(passage_texts)))
    indices = results.documents
    scores = results.scores
    return indices, scores


def rrf_fuse(bm25_indices: np.ndarray, dense_indices: np.ndarray,
            k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion of BM25 and dense rankings."""
    n = len(bm25_indices)
    fused = []
    for i in range(n):
        scores = {}
        for rank, idx in enumerate(bm25_indices[i]):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(dense_indices[i]):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        ranked = sorted(scores.keys(), key=lambda x: -scores[x])
        fused.append(ranked[:100])
    return np.array(fused)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--n-passages", type=int, default=5000)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from tinyrerank.data import load_jaqara, load_opus_en_ja

    # Load eval sets
    print("Loading eval sets...", flush=True)
    ja_q, ja_p, ja_rel = load_jaqara(args.n_queries, args.n_passages)
    en_q, en_p, en_rel = load_opus_en_ja(args.n_queries, args.n_passages)

    # Load student
    student = Encoder(str(Path("data/models/student-matryoshka")),
                      device=args.device)

    results = []
    for eval_name, queries, passages, rel in [
        ("ja_ja", ja_q, ja_p, ja_rel),
        ("en_ja", en_q, en_p, en_rel),
    ]:
        print(f"\n=== {eval_name} ===", flush=True)
        q_texts = [q["text"] for q in queries]
        p_texts = [p["text"] for p in passages]
        p_ids = [p["id"] for p in passages]

        # Dense
        print("  Dense retrieval...", flush=True)
        q_emb = student.encode_queries(q_texts)
        p_emb = student.encode_passages(p_texts)
        dense_idx, _, dense_lat = timed_search(q_emb, p_emb, k=100)
        dense_judgments = build_relevance_judgments(
            [q["id"] for q in queries], dense_idx, p_ids, rel)
        dense_metrics = compute_metrics(dense_judgments, dense_lat)
        print(f"  Dense: nDCG@10={dense_metrics['ndcg_at_10']:.4f} "
              f"R@100={dense_metrics['recall_at_100']:.4f}", flush=True)

        # BM25
        print("  BM25 (MeCab)...", flush=True)
        bm25_idx, _ = bm25_search(q_texts, p_texts, k=100)
        bm25_judgments = build_relevance_judgments(
            [q["id"] for q in queries], bm25_idx, p_ids, rel)
        bm25_metrics = compute_metrics(bm25_judgments)
        print(f"  BM25: nDCG@10={bm25_metrics['ndcg_at_10']:.4f} "
              f"R@100={bm25_metrics['recall_at_100']:.4f}", flush=True)

        # RRF fusion
        print(f"  RRF fusion (k={args.rrf_k})...", flush=True)
        fused_idx = rrf_fuse(bm25_idx, dense_idx, k=args.rrf_k)
        fused_judgments = build_relevance_judgments(
            [q["id"] for q in queries], fused_idx, p_ids, rel)
        fused_metrics = compute_metrics(fused_judgments)
        print(f"  RRF: nDCG@10={fused_metrics['ndcg_at_10']:.4f} "
              f"R@100={fused_metrics['recall_at_100']:.4f}", flush=True)

        results.append({
            "eval_set": eval_name,
            "dense": dense_metrics,
            "bm25": bm25_metrics,
            "rrf": fused_metrics,
        })

    # Summary
    print(f"\n{'eval':8s} {'method':8s} {'nDCG@10':>8s} {'R@100':>8s}")
    for r in results:
        for method in ["dense", "bm25", "rrf"]:
            m = r[method]
            print(f"{r['eval_set']:8s} {method:8s} {m['ndcg_at_10']:8.4f} {m['recall_at_100']:8.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "fusion.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {RESULTS / 'fusion.json'}")


if __name__ == "__main__":
    main()
