#!/usr/bin/env python3
"""How much does train/eval overlap matter? The distillation pairs and the EN-JA eval are both
seed-0 samples of the same OPUS-100 en-ja train split, so about 5% of eval queries were seen in training.
This re-scores a model on all queries and on the queries whose pair was NOT in the training sample.

  PYTHONPATH=src python scripts/leakage_check.py --model data/models/student-distilled
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tinyrerank.data import load_opus_en_ja
from tinyrerank.eval import build_relevance_judgments, compute_metrics, timed_search
from tinyrerank.models import Encoder

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--n-queries", type=int, default=500)
ap.add_argument("--n-passages", type=int, default=5000)
ap.add_argument("--train-pairs", type=int, default=50000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/leakage_check.json")
args = ap.parse_args()

N_TOTAL = 1_000_000  # len(opus-100 en-ja train)
train_idx = set(int(i) for i in np.random.default_rng(args.seed).choice(N_TOTAL, size=args.train_pairs, replace=False))
queries, passages, qid_to_rel = load_opus_en_ja(args.n_queries, args.n_passages, seed=args.seed)
seen = np.array([int(q["id"][1:]) in train_idx for q in queries])
print(f"{seen.sum()}/{len(queries)} queries were in the training sample", flush=True)

enc = Encoder(args.model)
q_emb = enc.encode_queries([q["text"] for q in queries])
p_emb = enc.encode_passages([p["text"] for p in passages])
indices, _, lat = timed_search(q_emb, p_emb, k=100)
judg = build_relevance_judgments([q["id"] for q in queries], indices, [p["id"] for p in passages], qid_to_rel)
res = {
    "model": args.model, "n_queries": len(queries), "n_seen_in_training": int(seen.sum()),
    "all": compute_metrics(judg, lat)["ndcg_at_10"],
    "unseen_only": compute_metrics([j for j, s in zip(judg, seen) if not s], [l for l, s in zip(lat, seen) if not s])["ndcg_at_10"],
    "seen_only": compute_metrics([j for j, s in zip(judg, seen) if s], [l for l, s in zip(lat, seen) if s])["ndcg_at_10"],
}
print(json.dumps(res, indent=2))
out = Path(args.out)
out.parent.mkdir(exist_ok=True)
prev = json.loads(out.read_text()) if out.exists() else []
out.write_text(json.dumps(prev + [res], indent=2))
