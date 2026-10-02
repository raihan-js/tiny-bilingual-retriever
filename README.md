# Tiny Bilingual Retriever

Distill BAAI/bge-m3 (568M) into sbintuitions/modernbert-ja-30m (30M) for English-Japanese cross-lingual retrieval on CPU.

## The problem

Tokyo companies with global customers need English queries to find Japanese documents. A 568M model is too big for CPU serving. Can a 30M student match the teacher's cross-lingual quality?

## The approach

1. **Distill** bge-m3 into modernbert-ja-30m using EN-JA sentence pairs
2. **Compress** with Matryoshka truncation, int8/binary quantisation, ONNX Runtime
3. **Fuse** with MeCab-segmented BM25 via RRF
4. **Measure** quality kept vs index MB vs CPU ms

## Baselines (500 queries, 5,000 passages)

### JA-JA monolingual (JQaRA)

| Model | nDCG@10 | p50 | Index MB |
|---|---|---|---|
| cl-nagoya/ruri-v3-30m | 0.9604 | 4.9ms | 4.9 |
| BAAI/bge-m3 | 0.9421 | 9.7ms | 19.5 |
| intfloat/multilingual-e5-small | 0.9113 | 6.0ms | 7.3 |
| sbintuitions/modernbert-ja-30m | 0.3223 | 3.6ms | 4.9 |

### EN-JA cross-lingual (opus-100)

| Model | nDCG@10 | p50 | Index MB |
|---|---|---|---|
| BAAI/bge-m3 | 0.6742 | 10.1ms | 19.5 |
| cl-nagoya/ruri-v3-30m | 0.5418 | 5.2ms | 4.9 |
| intfloat/multilingual-e5-small | 0.4795 | 5.9ms | 7.3 |
| sbintuitions/modernbert-ja-30m | 0.0373 | 4.1ms | 4.9 |

**Key finding:** modernbert-ja-30m has no retrieval training — 18× gap to bge-m3 on EN-JA. Distillation should close this.

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
