#!/usr/bin/env python3
"""Run all baselines on JA-JA (JQaRA) and EN-JA (opus-100) eval sets.

Usage:
  PYTHONPATH=src python scripts/run_baselines.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tinyrerank.eval import (RetrievalResult, build_relevance_judgments,
                             compute_metrics, timed_search)
from tinyrerank.models import Encoder

MODELS = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-small",
    "cl-nagoya/ruri-v3-30m",
    "sbintuitions/modernbert-ja-30m",
]

RESULTS = Path("data/results")


def load_jaqara(n_queries: int = 1000, n_passages: int = 10000, seed: int = 0):
    """Load JQaRA dev split for JA-JA monolingual eval."""
    from datasets import load_dataset
    ds = load_dataset("hotchpotch/JQaRA", split="dev")
    # Group by q_id
    from collections import defaultdict
    q_to_passages = defaultdict(list)
    for row in ds:
        q_to_passages[row["q_id"]].append(row)
    # Sample queries with at least 1 relevant passage
    rng = np.random.default_rng(seed)
    qids = [qid for qid, rows in q_to_passages.items()
            if any(r["label"] > 0 for r in rows)]
    selected = rng.choice(qids, size=min(n_queries, len(qids)), replace=False)
    queries, passages, qid_to_rel = [], [], {}
    for qid in selected:
        rows = q_to_passages[qid]
        for r in rows:
            pid = r["passage_row_id"]
            if pid not in [p["id"] for p in passages]:
                passages.append({"id": pid, "text": r["text"], "title": r["title"]})
        queries.append({"id": qid, "text": rows[0]["question"]})
        qid_to_rel[qid] = {r["passage_row_id"] for r in rows if r["label"] > 0}
    # Subsample passages if too many
    if len(passages) > n_passages:
        keep = set()
        for qid in selected:
            keep.update(qid_to_rel[qid])
        rng2 = np.random.default_rng(seed + 1)
        rest = [p for p in passages if p["id"] not in keep]
        n_rest = max(0, n_passages - len(keep))
        if n_rest < len(rest):
            idx = rng2.choice(len(rest), size=n_rest, replace=False)
            rest = [rest[i] for i in idx]
        passages = [p for p in passages if p["id"] in keep] + rest
    return queries, passages, qid_to_rel


def load_opus_en_ja(n_queries: int = 1000, n_passages: int = 10000, seed: int = 0):
    """Create EN-JA cross-lingual eval from opus-100.

    Each EN query is relevant to exactly 1 JA passage (its translation).
    """
    from datasets import load_dataset
    ds = load_dataset("Helsinki-NLP/opus-100", "en-ja", split="train")
    rng = np.random.default_rng(seed)
    # Sample passage indices
    n_total = len(ds)
    p_idx = rng.choice(n_total, size=min(n_passages, n_total), replace=False)
    p_idx_set = set(p_idx)
    # Sample query indices (must have their passage in the collection)
    q_candidates = [i for i in range(n_total) if i in p_idx_set]
    q_idx = rng.choice(q_candidates, size=min(n_queries, len(q_candidates)), replace=False)
    passages = [{"id": f"p{i}", "text": ds[int(i)]["translation"]["ja"]} for i in p_idx]
    queries = [{"id": f"q{i}", "text": ds[int(i)]["translation"]["en"]} for i in q_idx]
    qid_to_rel = {f"q{i}": {f"p{i}"} for i in q_idx}
    return queries, passages, qid_to_rel


def run_eval(encoder: Encoder, queries: list, passages: list,
             qid_to_rel: dict, eval_set: str) -> RetrievalResult:
    """Run retrieval eval for one encoder."""
    print(f"  encoding {len(queries)} queries...", flush=True)
    q_emb = encoder.encode_queries([q["text"] for q in queries])
    print(f"  encoding {len(passages)} passages...", flush=True)
    p_emb = encoder.encode_passages([p["text"] for p in passages])
    print(f"  searching...", flush=True)
    indices, scores, latencies = timed_search(q_emb, p_emb, k=100)
    p_ids = [p["id"] for p in passages]
    judgments = build_relevance_judgments(
        [q["id"] for q in queries], indices, p_ids, qid_to_rel)
    metrics = compute_metrics(judgments, latencies)
    return RetrievalResult(
        model=encoder.name,
        eval_set=eval_set,
        n_queries=len(queries),
        n_passages=len(passages),
        ndcg_at_10=metrics["ndcg_at_10"],
        recall_at_100=metrics["recall_at_100"],
        latency_p50_ms=metrics["latency_p50_ms"],
        latency_p95_ms=metrics["latency_p95_ms"],
        index_mb=encoder.index_size_mb(len(passages)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--n-queries", type=int, default=1000)
    ap.add_argument("--n-passages", type=int, default=10000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    # Load eval sets
    print("Loading JQaRA (JA-JA)...", flush=True)
    ja_queries, ja_passages, ja_rel = load_jaqara(args.n_queries, args.n_passages)
    print(f"  {len(ja_queries)} queries, {len(ja_passages)} passages", flush=True)

    print("Loading opus-100 (EN-JA)...", flush=True)
    en_queries, en_passages, en_rel = load_opus_en_ja(args.n_queries, args.n_passages)
    print(f"  {len(en_queries)} queries, {len(en_passages)} passages", flush=True)

    results = []
    for model_name in args.models:
        print(f"\n=== {model_name} ===", flush=True)
        enc = Encoder(model_name, device=args.device)
        print("  JA-JA eval:", flush=True)
        r = run_eval(enc, ja_queries, ja_passages, ja_rel, "ja_ja")
        results.append(r)
        print(f"    nDCG@10={r.ndcg_at_10:.4f} Recall@100={r.recall_at_100:.4f} "
              f"p50={r.latency_p50_ms:.1f}ms", flush=True)
        print("  EN-JA eval:", flush=True)
        r = run_eval(enc, en_queries, en_passages, en_rel, "en_ja")
        results.append(r)
        print(f"    nDCG@10={r.ndcg_at_10:.4f} Recall@100={r.recall_at_100:.4f} "
              f"p50={r.latency_p50_ms:.1f}ms", flush=True)

    # Summary
    print(f"\n{'model':35s} {'eval':8s} {'nDCG@10':>8s} {'R@100':>8s} {'p50_ms':>8s} {'MB':>8s}")
    for r in results:
        print(f"{r.model:35s} {r.eval_set:8s} {r.ndcg_at_10:8.4f} {r.recall_at_100:8.4f} "
              f"{r.latency_p50_ms:8.1f} {r.index_mb:8.1f}")

    (RESULTS / "baselines.json").write_text(json.dumps(
        [r.__dict__ for r in results], indent=2))
    print(f"\nSaved {RESULTS / 'baselines.json'}")


if __name__ == "__main__":
    main()
