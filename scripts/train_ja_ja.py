#!/usr/bin/env python3
"""Train student on JA-JA pairs from JQaRA to improve monolingual retrieval.

Usage:
  PYTHONPATH=src python scripts/train_ja_ja.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("data")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--n-queries", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from collections import defaultdict
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer, losses

    np.random.seed(args.seed)

    # Load JQaRA
    ds = load_dataset("hotchpotch/JQaRA", split="dev")
    q_to_data = defaultdict(lambda: {"question": "", "pos": [], "neg": []})
    for row in ds:
        qid = row["q_id"]
        q_to_data[qid]["question"] = row["question"]
        if row["label"] > 0:
            q_to_data[qid]["pos"].append(row["text"])
        else:
            q_to_data[qid]["neg"].append(row["text"])

    # Filter queries with at least 1 positive
    valid = {qid: d for qid, d in q_to_data.items() if len(d["pos"]) > 0}
    rng = np.random.default_rng(args.seed)
    qids = list(valid.keys())
    selected = rng.choice(qids, size=min(args.n_queries, len(qids)), replace=False)

    # Build training pairs: (query, positive)
    from sentence_transformers import InputExample
    train_examples = []
    for qid in selected:
        d = valid[qid]
        for pos_text in d["pos"]:
            train_examples.append(InputExample(texts=[d["question"], pos_text]))

    print(f"Loaded {len(train_examples)} JA-JA training pairs from {len(selected)} queries",
          flush=True)

    # Load student
    student = SentenceTransformer(str(DATA_DIR / "models" / "student-matryoshka"),
                                  device=args.device)

    # Train with MultipleNegativesRankingLoss
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(student)

    student.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        show_progress_bar=True,
    )

    # Save
    out_dir = DATA_DIR / "models" / "student-ja-en"
    out_dir.mkdir(parents=True, exist_ok=True)
    student.save(str(out_dir))
    print(f"Saved to {out_dir}", flush=True)

    meta = {"source": "JQaRA", "n_queries": len(selected),
            "n_pairs": len(train_examples), "epochs": args.epochs,
            "lr": args.lr, "seed": args.seed}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
