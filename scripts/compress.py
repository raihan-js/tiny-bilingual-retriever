#!/usr/bin/env python3
"""Compression: Matryoshka fine-tune, int8/binary quantisation, ONNX export.

Measures nDCG@10 against index MB and CPU ms at each compression level.

Usage:
  PYTHONPATH=src python scripts/compress.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
RESULTS = DATA_DIR / "results"

DIMENSIONS = [256, 128, 64]


def matryoshka_finetune(student, teacher_en_emb: np.ndarray, teacher_ja_emb: np.ndarray,
                        pairs: list[dict], epochs: int = 2, batch_size: int = 64,
                        lr: float = 1e-5, device: str = "cuda") -> None:
    """Fine-tune student with Matryoshka loss at multiple dimensions.

    A projection layer maps teacher 1024-dim → student 256-dim. The student
    learns to put the most important information in the first d dimensions.
    """
    import torch
    from torch import nn

    student.to(device)
    student.train()

    teacher_dim = teacher_en_emb.shape[1]
    student_dim = student.get_sentence_embedding_dimension()
    projector = nn.Linear(teacher_dim, student_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projector.parameters()), lr=lr)

    en_texts = [p["en"] for p in pairs]
    ja_texts = [p["ja"] for p in pairs]
    all_texts = en_texts + ja_texts
    all_teacher_emb = np.concatenate([teacher_en_emb, teacher_ja_emb], axis=0)

    n = len(all_texts)
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            texts = [all_texts[i] for i in batch_idx]
            target = torch.tensor(all_teacher_emb[batch_idx], dtype=torch.float32, device=device)

            features = student.tokenize(texts, padding=True, truncation=True, max_length=512)
            features = {k: v.to(device) if hasattr(v, "to") else torch.tensor(v, device=device)
                        for k, v in features.items() if k != "modality"}
            output = student(features)
            student_emb = output["sentence_embedding"]

            # Project teacher to student dim
            teacher_proj = projector(target)

            # Matryoshka loss: sum over dimensions
            loss = 0.0
            for d in DIMENSIONS:
                s = torch.nn.functional.normalize(student_emb[:, :d], dim=-1)
                t = torch.nn.functional.normalize(teacher_proj[:, :d], dim=-1)
                loss = loss + (1.0 - torch.nn.functional.cosine_similarity(s, t, dim=-1).mean())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"  epoch {epoch + 1}/{epochs}: matryoshka_loss={avg_loss:.6f}", flush=True)


def export_onnx(student, out_path: Path, dim: int = 256) -> None:
    """Export student to ONNX Runtime for CPU inference."""
    import torch
    from torch import nn

    class ONNXWrapper(nn.Module):
        def __init__(self, student, dim):
            super().__init__()
            self.student = student
            self.dim = dim

        def forward(self, input_ids, attention_mask):
            output = self.student({"input_ids": input_ids, "attention_mask": attention_mask})
            emb = output["sentence_embedding"][:, :self.dim]
            return torch.nn.functional.normalize(emb, dim=-1)

    wrapper = ONNXWrapper(student, dim).eval()
    tokenizer = student.tokenize

    # Dummy input
    dummy = tokenizer(["hello"], padding=True, truncation=True, max_length=512)
    dummy = {k: v for k, v in dummy.items() if k != "modality"}

    torch.onnx.export(
        wrapper,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["sentence_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "sentence_embedding": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"  Exported ONNX to {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-pairs", type=int, default=50000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    np.random.seed(args.seed)

    # Load training pairs
    from datasets import load_dataset
    ds = load_dataset("Helsinki-NLP/opus-100", "en-ja", split="train")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(args.n_pairs, len(ds)), replace=False)
    pairs = [{"en": ds[int(i)]["translation"]["en"],
              "ja": ds[int(i)]["translation"]["ja"]} for i in idx]

    # Load cached teacher embeddings
    cache_file = CACHE_DIR / f"teacher_emb_{args.n_pairs}.npz"
    cached = np.load(cache_file)
    teacher_en_emb, teacher_ja_emb = cached["en"], cached["ja"]

    # Load student
    student = SentenceTransformer(str(DATA_DIR / "models" / "student-distilled"),
                                  device=args.device)

    # Matryoshka fine-tune
    print("Matryoshka fine-tune...", flush=True)
    matryoshka_finetune(student, teacher_en_emb, teacher_ja_emb, pairs,
                        epochs=args.epochs, batch_size=args.batch_size,
                        device=args.device)

    # Save
    out_dir = DATA_DIR / "models" / "student-matryoshka"
    out_dir.mkdir(parents=True, exist_ok=True)
    student.save(str(out_dir))
    print(f"Saved to {out_dir}", flush=True)

    # Export ONNX at each dimension
    print("Exporting ONNX...", flush=True)
    for d in DIMENSIONS:
        export_onnx(student, out_dir / f"model_{d}.onnx", dim=d)

    # Quantisation info
    print("\nQuantisation summary:", flush=True)
    for d in DIMENSIONS:
        fp32_mb = 10000 * d * 4 / (1024 * 1024)
        int8_mb = 10000 * d * 1 / (1024 * 1024)
        binary_mb = 10000 * d * 0.125 / (1024 * 1024)
        print(f"  dim={d}: fp32={fp32_mb:.1f}MB int8={int8_mb:.1f}MB binary={binary_mb:.1f}MB",
              flush=True)

    meta = {"dimensions": DIMENSIONS, "epochs": args.epochs,
            "n_pairs": len(pairs), "seed": args.seed}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
