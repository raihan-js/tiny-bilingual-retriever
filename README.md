# Tiny Bilingual Retriever

Distill BAAI/bge-m3 (568M) into sbintuitions/modernbert-ja-30m (30M) for English-Japanese cross-lingual retrieval on CPU.

## The problem

Tokyo companies with global customers need English queries to find Japanese documents. A 568M model is too big for CPU serving. Can a 30M student match the teacher's cross-lingual quality?

## The approach

1. **Distill** bge-m3 into modernbert-ja-30m using EN-JA sentence pairs
2. **Compress** with Matryoshka truncation, int8/binary quantisation, ONNX Runtime
3. **Fuse** with MeCab-segmented BM25 via RRF
4. **Measure** quality kept vs index MB vs CPU ms

## Results

### Baselines (500 queries, 5,000 passages)

| Model | JA-JA nDCG@10 | EN-JA nDCG@10 | p50 | Index MB |
|---|---|---|---|---|
| cl-nagoya/ruri-v3-30m | 0.9604 | 0.5418 | 4.9ms | 4.9 |
| BAAI/bge-m3 (teacher) | 0.9421 | 0.6742 | 9.7ms | 19.5 |
| intfloat/multilingual-e5-small | 0.9113 | 0.4795 | 6.0ms | 7.3 |
| sbintuitions/modernbert-ja-30m | 0.3223 | 0.0373 | 3.6ms | 4.9 |

### After distillation (50k pairs, 3 epochs, cosine similarity loss)

| Model | JA-JA nDCG@10 | EN-JA nDCG@10 | vs teacher | Index MB |
|---|---|---|---|---|
| BAAI/bge-m3 (teacher) | 0.9421 | 0.6742 | 100% | 19.5 |
| **student-distilled** | 0.2179 | **0.4809** | **71%** | **4.9** |

**Key finding:** The student captures 71% of the teacher's EN-JA quality at 1/19th the index size. JA-JA is worse than the untrained baseline — expected, since training is on EN-JA pairs only.

## Usage

```bash
# Baselines
PYTHONPATH=src python scripts/run_baselines.py

# Train student
PYTHONPATH=src python scripts/train_student.py

# Compress
PYTHONPATH=src python scripts/compress.py

# Hybrid fusion
PYTHONPATH=src python scripts/run_fusion.py
```

## Limitations

- EN-JA eval is synthetic (opus-100 pairs), not a standard benchmark
- Absolute nDCG inflated by subsampling; only relative comparisons meaningful
- Student starts with no retrieval training; distillation quality depends on teacher

## License

MIT
