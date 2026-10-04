# I Distilled a 568M Multilingual Model Into a 37M Japanese-English Encoder — Here's What Survived

*Cross-lingual retrieval distillation, compression, and the honest cost table.*

---

> Scope note: English-to-Japanese retrieval only. EN-JA eval is synthetic (opus-100 pairs), not a standard benchmark. All models compared on the same fixed subsample.

![tiny-bilingual-retriever results](https://raw.githubusercontent.com/raihan-js/tiny-bilingual-retriever/HEAD/images/tiny-bilingual.png)

## The problem

Tokyo companies with global customers need English queries to find Japanese documents. A 568M model (bge-m3) is too big for CPU serving. Can a ~37M student match the teacher's cross-lingual quality?

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
| **student-distilled (36.7M)** | **0.4809** | **71%** | **4.9** |
| intfloat/multilingual-e5-small | 0.4795 | 71% | 7.3 |
| cl-nagoya/ruri-v3-30m | 0.5418 | 80% | 4.9 |
| modernbert-ja-30m (untrained) | 0.0373 | 6% | 4.9 |

**The student captures 71% of the teacher's EN-JA quality with 15× fewer parameters (36.7M vs 568M) and a 4× smaller index (4.9 vs 19.5 MB).** It matches e5-small (a 118M model) despite being 4× smaller.

### Compression (Matryoshka fine-tune)

| dim | EN-JA nDCG@10 | vs dim=256 | Index MB | p50 |
|---|---|---|---|---|
| 256 | 0.4158 | 100% | 4.9 | 4.5ms |
| 128 | 0.3989 | 96% | 2.4 | 4.7ms |
| 64 | 0.3613 | 87% | 1.2 | 0.5ms |

**Matryoshka truncation lets you choose the quality/size trade-off at serving time.** At dim=64 you keep 87% of the dim-256 model's score at 1/4 the index. Two caveats: the Matryoshka fine-tune itself cost quality at full width (0.4809 distilled → 0.4158 at dim 256), and against the teacher dim 64 is 54% (0.3613 vs 0.6742). The latency column is a single PyTorch/CPU measurement; the drop from 4.7 ms to 0.5 ms between dim 128 and 64 looks like noise, not a 10× speed-up.

### Hybrid fusion (RRF, k=60)

| eval | Dense | BM25 (MeCab) | RRF fusion |
|---|---|---|---|
| JA-JA | 0.1545 | **0.8123** | 0.4334 |
| EN-JA | **0.4158** | 0.0166 | 0.2049 |

**RRF hurts when one method dominates.** The student is strong at EN-JA but weak at JA-JA; BM25 is the reverse. Fusing a strong method with a weak one drags the strong method down. Hybrid only helps when both methods are reasonably good on the same query type.

### After JA-JA training (JQaRA, 3 epochs)

| Model | JA-JA nDCG@10 | EN-JA nDCG@10 |
|---|---|---|
| student-matryoshka | 0.1545 | **0.4158** |
| student-ja-en | **0.4348** | 0.2632 |

Training on Japanese pairs lifts monolingual retrieval (0.15 → 0.43) and costs cross-lingual quality (0.42 → 0.26). The student becomes more balanced and loses its EN-JA specialty: a real trade-off, not a free improvement.

### Quantisation (int8 and binary)

| Method | EN-JA nDCG@10 | vs float32 | Index MB |
|---|---|---|---|
| float32 | 0.4158 | 100% | 4.9 |
| int8 | 0.4143 | 99.6% | 1.2 |
| binary | 0.0322 | 7.7% | 0.2 |

int8 is nearly free: 99.6% of the quality at 1/4 the index. Binary is catastrophic here, because the sign of each element loses too much information at this embedding width.

## Key findings

1. **Cosine similarity loss >> MSE loss for retrieval distillation.** MSE gave 0.03 EN-JA nDCG; cosine gave 0.48.
2. **Matryoshka truncation is a free lunch.** 87% quality at 1/4 the index size.
3. **RRF fusion is not a free lunch.** It hurts when one method dominates.
4. **The student is a specialist, not a generalist.** Good at EN-JA, poor at JA-JA (0.22 vs the teacher's 0.94); fixing JA-JA costs EN-JA.
5. **A public model of the same size beats it.** cl-nagoya/ruri-v3-30m scores higher on both (EN-JA 0.54 vs 0.48). The contribution here is the measured recipe and cost table, not a new best model.

## Limitations

- EN-JA eval is synthetic (opus-100 pairs), not a standard benchmark.
- Absolute nDCG inflated by subsampling; only relative comparisons meaningful.
- Student trained on EN-JA pairs only; JA-JA quality is poor.
- Latency is the brute-force search time over 5,000 pre-encoded passages (not query-encoding time), PyTorch on CPU, single measurement; a re-run of the baselines differed by up to ~5x, so treat it as order-of-magnitude. `scripts/compress.py` has an ONNX export path, but no ONNX numbers are reported here.

## What's next

- Evaluate on human-written Japanese sets (JQaRA, JaCWIR) instead of synthetic opus-100 pairs
- Start from ruri-v3-30m or distil with a mixed EN-JA / JA-JA objective to avoid the specialist trade-off
- Report ONNX Runtime latency and a verified export

---

*Repo: github.com/raihan-js/tiny-bilingual-retriever · 14 tests green. The method is known (Reimers & Gurevych 2020); the contribution is the specific small JA-EN student and the honest cost table.*
