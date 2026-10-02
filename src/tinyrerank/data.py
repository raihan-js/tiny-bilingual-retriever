"""Dataset loading for retrieval evaluation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def load_jaqara(n_queries: int = 1000, n_passages: int = 10000, seed: int = 0):
    """Load JQaRA dev split for JA-JA monolingual eval."""
    from datasets import load_dataset
    ds = load_dataset("hotchpotch/JQaRA", split="dev")
    q_to_passages = defaultdict(list)
    for row in ds:
        q_to_passages[row["q_id"]].append(row)
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
    """Create EN-JA cross-lingual eval from opus-100."""
    from datasets import load_dataset
    ds = load_dataset("Helsinki-NLP/opus-100", "en-ja", split="train")
    rng = np.random.default_rng(seed)
    n_total = len(ds)
    p_idx = rng.choice(n_total, size=min(n_passages, n_total), replace=False)
    p_idx_set = set(p_idx)
    q_candidates = [i for i in range(n_total) if i in p_idx_set]
    q_idx = rng.choice(q_candidates, size=min(n_queries, len(q_candidates)), replace=False)
    passages = [{"id": f"p{i}", "text": ds[int(i)]["translation"]["ja"]} for i in p_idx]
    queries = [{"id": f"q{i}", "text": ds[int(i)]["translation"]["en"]} for i in q_idx]
    qid_to_rel = {f"q{i}": {f"p{i}"} for i in q_idx}
    return queries, passages, qid_to_rel
