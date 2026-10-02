# AGENTS.md — tiny-bilingual-retriever

Portfolio project for Raihan Sikder. Target roles: Noeon Research (Senior ML Engineer, LLMOps), PayPay Card, Money Forward, Treasure AI, Citadel AI.

## Project: Tiny Bilingual Retriever

Distill BAAI/bge-m3 (568M) into sbintuitions/modernbert-ja-30m (30M) for English-Japanese cross-lingual retrieval on CPU. Add Matryoshka truncation, binary quantisation, and MeCab-segmented BM25 fusion.

## Why this project

Tokyo companies with global customers need English queries to find Japanese documents. The method is known (multilingual knowledge distillation, Reimers & Gurevych 2020), so the contribution is the specific small JA-EN student, the compression study, and an honest cost table (quality kept vs index MB vs CPU ms).

## Stack

Python, sentence-transformers, BAAI/bge-m3 (teacher), sbintuitions/modernbert-ja-30m (student), fugashi + unidic-lite (MeCab), bm25s, onnxruntime.

## Compute

One RTX 3060 12GB. Teacher embeddings for ~1M pairs computed once in fp16 (few hours) and cached. 30M student trains in a few hours. No rented GPU.

## Datasets

- Training: Helsinki-NLP/opus-100 (en-ja, 1M pairs)
- Eval JA-JA: hotchpotch/JQaRA (dev, 86,850 items)
- Eval EN-JA: custom from opus-100 (EN queries, JA passages)

## Milestones

1. **Baselines and eval harness** (5d) — nDCG@10, Recall@100, CPU latency for all models
2. **Distillation** (6d) — 30M student trained to match bge-m3
3. **Compression** (4d) — Matryoshka, int8/binary, ONNX CPU
4. **Hybrid fusion and release** (5d) — RRF of BM25 + student, HF model, write-up

## Conventions

- Python 3.10+, pytest for all eval metrics.
- All models compared on the same fixed subsample.
- Absolute nDCG inflated by subsampling; only relative comparisons meaningful.
- Release model and code, not raw pairs.

## Development

```bash
cd tiny-bilingual-retriever
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,bm25]"
pytest tests/ -v

# Run baselines
PYTHONPATH=src python scripts/run_baselines.py

# Train student
PYTHONPATH=src python scripts/train_student.py

# Compress
PYTHONPATH=src python scripts/compress.py

# Hybrid fusion
PYTHONPATH=src python scripts/run_fusion.py
```

## Current status

- 14 tests passing
- Baselines complete: bge-m3, e5-small, ruri-v3-30m, modernbert-ja-30m
- Distillation complete: 50k pairs, 3 epochs, cosine similarity loss
- **EN-JA: student captures 71% of teacher quality at 1/19th the index size**
- Pending: compression (Matryoshka, int8/binary, ONNX), fusion, release
