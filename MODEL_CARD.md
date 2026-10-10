---
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
  - dispatcher
  - attention-free
  - subquadratic
  - causal-lm
datasets:
  - HuggingFaceFW/fineweb-edu
language:
  - en
---

# Dispatcher-Hybrid 292M

A causal language model whose mixer is a **gated dyadic shift-and-sum** rather
than self-attention, in 16 of 18 layers; the remaining 2 layers are
sliding-window (1024) attention. Mixing is O(N log N) and comparison-free in
the tree layers. Revival of [arXiv:2105.03994](https://arxiv.org/abs/2105.03994).

Code and reproduction scripts: https://github.com/fractalego/dispatcher-hybrid-2

## Model details

| | |
|---|---|
| parameters | 292.2M (tied embeddings) |
| layers | 18 — tree in all but layers 8 and 17 (1-indexed), which are attention |
| d_model / heads / FFN | 1024 / 16 / 2816 (SwiGLU) |
| context | 4096 (12 dyadic levels) |
| attention window | 1024, RoPE; tree layers use no positional encoding |
| tokenizer | GPT-2 BPE (50257) |
| training data | FineWeb-Edu (`sample-100BT`), 94.9B tokens, single epoch |
| precision | released in fp32; evaluate under bf16 autocast |

Note on dtype: these weights were trained and evaluated in fp32 with bf16
autocast. Casting the *weights* to bf16 is not recommended — for models this
size that cast is measurably lossy.

## Results

Measured in fp32 with a single harness. Held-out CE is on 400 FineWeb-Edu
documents from a stream position training never reached.

| metric | value |
|---|---|
| held-out cross-entropy | 2.5543 (ppl 12.86) |
| LAMBADA (openai) acc / ppl | .347 / 15.83 |
| LAMBADA, answer appears earlier | .396 |
| LAMBADA, answer not earlier | .096 |
| HellaSwag acc_norm | .431 |
| PIQA acc_norm | .679 |
| WinoGrande | .549 |
| ARC (easy+challenge merged) acc_norm | .489 zero-shot, .544 5-shot |
| MQAR (8 / 128 / 256 pairs) | 78.1% / 49.7% / 24.1% |

A same-recipe 18-layer attention transformer trained on the same data at 6.5B
tokens scores CE 2.8097, LAMBADA .285, HellaSwag .334 — a controlled
architecture comparison at that budget. This model's 94.9B-token figures are
not budget-matched against any baseline.

## Intended use and limits

Research artifact: a base model for studying sub-quadratic sequence mixing,
and a small English text generator. Not instruction-tuned, not aligned, and
not filtered for safety — it reproduces whatever FineWeb-Edu contains.

Known weaknesses, measured rather than assumed:

- **Last-word precision.** GPT-2-medium-tier perplexity and commonsense
  accuracy, GPT-2-small-tier LAMBADA accuracy. The deficit concentrates on
  predictions that require precise retrieval.
- **Arbitrary in-context binding** degrades with load (78% at 8 key-value
  pairs, 50% at 128) and is effectively zero beyond the 1024-token attention
  window. Semantic retrieval through the tree is unaffected.
- **Generation recomputes the prefix each step** in this release (~40 tok/s on
  an A6000 at short context); no cache is wired up yet.
- **4096-token context**, fixed by the number of dyadic levels.
