#!/usr/bin/env python3
"""Quantisation: int8 and binary quantisation of student embeddings.

Measures quality degradation at each quantisation level.

Usage:
  PYTHONPATH=src python scripts/quantize.py
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


def quantize_int8(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize embeddings to int8. Returns (quantized, min, max)."""
    emin = embeddings.min(axis=0, keepdims=True)
    emax = embeddings.max(axis=0, keepdims=True)
    scale = (emax - emin) / 255.0
    scale = np.maximum(scale, 1e-8)
    quantized = np.round((embeddings - emin) / scale).astype(np.uint8)
    return quantized, emin, emax


def dequantize_int8(quantized: np.ndarray, emin: np.ndarray, emax: np.ndarray) -> np.ndarray:
    """Dequantize int8 embeddings."""
    scale = (emax - emin) / 255.0
    return quantized.astype(np.float32) * scale + emin


def quantize_binary(embeddings: np.ndarray) -> np.ndarray:
    """Binarize embeddings (sign of each element)."""
    return (embeddings > 0).astype(np.float32) * 2 - 1


def evaluate_quantization(encoder: Encoder, queries: list, passages: list,
                          qid_to_rel: dict, eval_set: str) -> dict:
    """Evaluate original, int8, and binary quantisation."""
    q_emb = encoder.encode_queries([q["text"] for q in queries])
    p_emb = encoder.encode_passages([p["text"] for p in passages])
    p_ids = [p["id"] for p in passages]

    results = {}

    # Original (float32)
    idx, _, lat = timed_search(q_emb, p_emb, k=100)
    judgments = build_relevance_judgments([q["id"] for q in queries], idx, p_ids, qid_to_rel)
    results["float32"] = compute_metrics(judgments, lat)

    # int8
    q_int8, q_min, q_max = quantize_int8(q_emb)
    p_int8, p_min, p_max = quantize_int8(p_emb)
    q_deint8 = dequantize_int8(q_int8, q_min, q_max)
    p_deint8 = dequantize_int8(p_int8, p_min, p_max)
    idx, _, lat = timed_search(q_deint8, p_deint8, k=100)
    judgments = build_relevance_judgments([q["id"] for q in queries], idx, p_ids, qid_to_rel)
    results["int8"] = compute_metrics(judgments, lat)

    # Binary
    q_bin = quantize_binary(q_emb)
    p_bin = quantize_binary(p_emb)
    idx, _, lat = timed_search(q_bin, p_bin, k=100)
    judgments = build_relevance_judgments([q["id"] for q in queries], idx, p_ids, qid_to_rel)
    results["binary"] = compute_metrics(judgments, lat)

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--n-passages", type=int, default=5000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # Load eval sets
    ja_q, ja_p, ja_rel = load_jaqara(args.n_queries, args.n_passages)
    en_q, en_p, en_rel = load_opus_en_ja(args.n_queries, args.n_passages)

    # Load student
    student = Encoder(str(Path("data/models/student-matryoshka")),
                      device=args.device)

    all_results = {}
    for eval_name, queries, passages, rel in [
        ("ja_ja", ja_q, ja_p, ja_rel),
        ("en_ja", en_q, en_p, en_rel),
    ]:
        print(f"\n=== {eval_name} ===", flush=True)
        r = evaluate_quantization(student, queries, passages, rel, eval_name)
        all_results[eval_name] = r
        for method in ["float32", "int8", "binary"]:
            m = r[method]
            print(f"  {method:8s}: nDCG@10={m['ndcg_at_10']:.4f} R@100={m['recall_at_100']:.4f}")

    # Summary
    print(f"\n{'eval':8s} {'method':8s} {'nDCG@10':>8s} {'R@100':>8s} {'MB':>8s}")
    for eval_name, r in all_results.items():
        for method in ["float32", "int8", "binary"]:
            m = r[method]
            n = args.n_passages
            if method == "float32":
                mb = n * 256 * 4 / (1024 * 1024)
            elif method == "int8":
                mb = n * 256 * 1 / (1024 * 1024)
            else:
                mb = n * 256 * 0.125 / (1024 * 1024)
            print(f"{eval_name:8s} {method:8s} {m['ndcg_at_10']:8.4f} {m['recall_at_100']:8.4f} {mb:8.1f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "quantization.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {RESULTS / 'quantization.json'}")


if __name__ == "__main__":
    main()
