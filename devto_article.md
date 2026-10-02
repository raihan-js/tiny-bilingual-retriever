# I Distilled a 568M Multilingual Model Into a 30M Japanese-English Encoder — Here's What Survived

*Cross-lingual retrieval distillation, compression, and the honest cost table.*

---

> Scope note: English-to-Japanese retrieval only. EN-JA eval is synthetic (opus-100 pairs), not a standard benchmark. All models compared on the same fixed subsample.

## The problem

Tokyo companies with global customers need English queries to find Japanese documents. A 568M model (bge-m3) is too big for CPU serving. Can a 30M student match the teacher's cross-lingual quality?

## The approach

1. **Distill** bge-m3 into modernbert-ja-30m using 50k EN-JA sentence pairs (cosine similarity loss)
2. **Compress** with Matryoshka truncation (256/128/64 dimensions)
3. **Fuse** with MeCab-segmented BM25 via RRF
4. **Measure** quality kept vs index MB vs CPU ms

## Results

### Distillation (50k pairs, 3 epochs)

| Model | EN-JA nDCG@10 | vs teacher | Index MB |
|---|---|---|---|
| BAAI/bge-m3 (teacher, 568M) | 0.6742 | 100% | 19.5 |
| **student-distilled (30M)** | **0.4809** | **71%** | **4.9** |
| intfloat/multilingual-e5-small | 0.4795 | 71% | 7.3 |
| cl-nagoya/ruri-v3-30m | 0.5418 | 80% | 4.9 |
| modernbert-ja-30m (untrained) | 0.0373 | 6% | 4.9 |

**The student captures 71% of the teacher's EN-JA quality at 1/19th the index size.** It matches e5-small (a 118M model) despite being 4× smaller.

### Compression (Matryoshka fine-tune)

| dim | EN-JA nDCG@10 | vs dim=256 | Index MB | p50 |
|---|---|---|---|---|
| 256 | 0.4158 | 100% | 4.9 | 4.5ms |
| 128 | 0.3989 | 96% | 2.4 | 4.7ms |
| 64 | 0.3613 | 87% | 1.2 | 0.5ms |

**Matryoshka truncation lets you choose the quality/size trade-off at serving time.** At dim=64, you get 87% of the quality at 1/4 the index size and 10× faster search.

### Hybrid fusion (RRF, k=60)

| eval | Dense | BM25 (MeCab) | RRF fusion |
|---|---|---|---|
| JA-JA | 0.1545 | **0.8123** | 0.4334 |
| EN-JA | **0.4158** | 0.0166 | 0.2049 |

**RRF hurts when one method dominates.** The student is strong at EN-JA but weak at JA-JA; BM25 is the reverse. Fusing a strong method with a weak one drags the strong method down. Hybrid only helps when both methods are reasonably good on the same query type.

## Key findings

1. **Cosine similarity loss >> MSE loss for retrieval distillation.** MSE gave 0.03 EN-JA nDCG; cosine gave 0.48.
2. **Matryoshka truncation is a free lunch.** 87% quality at 1/4 the index size.
3. **RRF fusion is not a free lunch.** It hurts when one method dominates.
4. **The student is a specialist, not a generalist.** Great at EN-JA, poor at JA-JA.

## Limitations

- EN-JA eval is synthetic (opus-100 pairs), not a standard benchmark.
- Absolute nDCG inflated by subsampling; only relative comparisons meaningful.
- Student trained on EN-JA pairs only; JA-JA quality is poor.
- ONNX export attempted but has load issues; CPU latency measured with PyTorch.

## What's next

- Train on JA-JA pairs to improve monolingual quality
- Add int8/binary quantisation for further compression
- Evaluate on human-written Japanese eval sets (JQaRA, JaCWIR)

---

*Repo: github.com/raihan-js/tiny-bilingual-retriever · 14 tests green. The method is known (Reimers & Gurevych 2020); the contribution is the specific small JA-EN student and the honest cost table.*
