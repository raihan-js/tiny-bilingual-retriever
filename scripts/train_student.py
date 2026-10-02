#!/usr/bin/env python3
"""Distill bge-m3 into modernbert-ja-30m using EN-JA sentence pairs.

The student learns to map both English and Japanese sentences to the same
embedding space as the teacher. Loss is MSE between student and teacher
embeddings.

Teacher embeddings are computed once on GPU and cached to disk.

Usage:
  PYTHONPATH=src python scripts/train_student.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

TEACHER = "BAAI/bge-m3"
STUDENT = "sbintuitions/modernbert-ja-30m"
DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"


def compute_teacher_embeddings(teacher, pairs: list[dict], batch_size: int = 256,
                               device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """Compute teacher embeddings for EN and JA sentences."""
    en_texts = [p["en"] for p in pairs]
    ja_texts = [p["ja"] for p in pairs]
    print(f"  Computing teacher embeddings for {len(pairs)} pairs...", flush=True)
    en_emb = teacher.encode(en_texts, batch_size=batch_size, show_progress_bar=True,
                            convert_to_numpy=True, normalize_embeddings=True, device=device)
    ja_emb = teacher.encode(ja_texts, batch_size=batch_size, show_progress_bar=True,
                            convert_to_numpy=True, normalize_embeddings=True, device=device)
    return en_emb, ja_emb


def train_student(student, pairs: list[dict], teacher_en_emb: np.ndarray,
                  teacher_ja_emb: np.ndarray, epochs: int = 3,
                  batch_size: int = 64, lr: float = 2e-5,
                  device: str = "cuda") -> None:
    """Train student with MSE loss on both EN and JA sentences.

    A projection layer bridges the dimension gap (student 256 → teacher 1024).
    """
    import torch
    from torch import nn

    # Ensure output directory exists
    (DATA_DIR / "models" / "student-distilled").mkdir(parents=True, exist_ok=True)

    student.to(device)
    student.train()

    # Projection layer: student_dim → teacher_dim
    student_dim = student.get_sentence_embedding_dimension()
    teacher_dim = teacher_en_emb.shape[1]
    projector = nn.Linear(student_dim, teacher_dim).to(device)

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

            # Encode with student + projection
            features = student.tokenize(texts, padding=True, truncation=True, max_length=512)
            features = {k: v.to(device) if hasattr(v, "to") else torch.tensor(v, device=device)
                        for k, v in features.items() if k != "modality"}
            output = student(features)
            student_emb = projector(output["sentence_embedding"])

            # Cosine similarity loss (more appropriate for retrieval than MSE)
            student_emb = torch.nn.functional.normalize(student_emb, dim=-1)
            target = torch.nn.functional.normalize(target, dim=-1)
            loss = 1.0 - torch.nn.functional.cosine_similarity(student_emb, target, dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"  epoch {epoch + 1}/{epochs}: loss={avg_loss:.6f}", flush=True)

    # Save the projector for inference
    import torch
    torch.save(projector.state_dict(), DATA_DIR / "models" / "student-distilled" / "projector.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--n-pairs", type=int, default=100000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", choices=["embed", "train", "all"], default="all")
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
    print(f"Loaded {len(pairs)} training pairs", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"teacher_emb_{args.n_pairs}.npz"

    if args.step in ("embed", "all"):
        print(f"Loading teacher: {TEACHER}", flush=True)
        teacher = SentenceTransformer(TEACHER, device=args.device)
        if cache_file.exists():
            print(f"Loading cached teacher embeddings from {cache_file}", flush=True)
            cached = np.load(cache_file)
            teacher_en_emb, teacher_ja_emb = cached["en"], cached["ja"]
        else:
            teacher_en_emb, teacher_ja_emb = compute_teacher_embeddings(
                teacher, pairs, batch_size=32, device=args.device)
            np.savez(cache_file, en=teacher_en_emb, ja=teacher_ja_emb)
            print(f"Cached teacher embeddings to {cache_file}", flush=True)
        del teacher
        import torch
        torch.cuda.empty_cache()
    else:
        cached = np.load(cache_file)
        teacher_en_emb, teacher_ja_emb = cached["en"], cached["ja"]

    if args.step in ("train", "all"):
        print(f"Loading student: {STUDENT}", flush=True)
        student = SentenceTransformer(STUDENT, device=args.device)

        # Train student
        print("Training student...", flush=True)
        train_student(student, pairs, teacher_en_emb, teacher_ja_emb,
                      epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, device=args.device)

        # Save
        out_dir = DATA_DIR / "models" / "student-distilled"
        out_dir.mkdir(parents=True, exist_ok=True)
        student.save(str(out_dir))
        print(f"Saved student to {out_dir}", flush=True)

        meta = {"teacher": TEACHER, "student": STUDENT, "n_pairs": len(pairs),
                "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                "seed": args.seed}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
