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
from tinyrerank.data import load_jaqara, load_opus_en_ja

MODELS = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-small",
    "cl-nagoya/ruri-v3-30m",
    "sbintuitions/modernbert-ja-30m",
]

RESULTS = Path("data/results")


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
