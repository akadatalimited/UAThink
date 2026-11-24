# UAThink reference model — D1024-H16 (FP32)

## Architecture

- Input: hashed character n-grams (`n = 3, 4`), `D = 1024` buckets
  (FNV-1a 32-bit + splitmix32-lite avalanche, binary presence).
- `EmbeddingBag(1024 x 64, mean)` over the hashed features (FP32).
- `FC 64 -> 16` + ReLU, `FC 16 -> 5` + softmax (FP32, AVX2/FMA matvec,
  scalar x87 exponential with max-subtraction for stability).
- Classes: `0=crawler 1=human_like 2=known_bot 3=scanner 4=script_tool`.
- Storage: `models/uathink_d1024h16_v2.bin`, 266,708 bytes
  (64-byte `UATH v1` header + 66,661 FP32 weights).

## Training methodology (reference weights)

- Online SGD with class-balanced oversampling of minority classes,
  deterministic order (fixed seed), 12 epochs.
- Learning-rate schedule per epoch: `0.005` (ep 0–3), `0.0025` (ep 4–7),
  `0.00125` (ep 8+); momentum `0`. Deterministic init (LCG).
- Numerical reference implementation kept bit-identical between the native
  forward pass and an independent FP32 oracle on the same weights.

## Measured metrics (held-out grouped-template splits, disjoint templates)

| Split | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| eval | 0.9971 | 0.6474 | per-class F1: crawler 0.667, human_like 0.999, known_bot 0.857, scanner 0.000 (support 4), script_tool 0.714 |
| unseen | 0.9972 | 0.6575 | unseen templates, same protocol |

- Calibration (ECE, 10 bins): ~0.005 on eval.
- Abstention at confidence `< 0.9`: ~2% of eval rows (route to review /
  behavioural layer rather than defaulting to `human_like`).

## Intended use and limits

- Scores **User-Agent identity claims** only. A `human_like` verdict means
  "this string looks like a human client", not "this request is benign".
- `scanner` recall is data-limited (tiny support); treat scanner output as a
  weak signal pending more labeled scanner traffic.
- Hostile automation that mimics mainstream browsers is expected to score
  `human_like` here by design — that is the behavioural layer's job, using
  request dynamics this model never sees.
- Quantisation (INT8) is future work; FP32 is the validated numerical
  reference. Validate any quantised build against this model's outputs
  before use.
