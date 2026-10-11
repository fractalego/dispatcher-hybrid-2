# Dispatcher-Hybrid

A 292M-parameter causal language model that replaces self-attention with a
**gated dyadic shift-and-sum** in 16 of its 18 layers, keeping only 2
sliding-window attention layers. No pairwise token comparison happens in the
tree layers, and the mixing cost is O(N log N) rather than O(N²).

This is the inference-ready release of a revival of the Dispatcher
([arXiv:2105.03994](https://arxiv.org/abs/2105.03994), 2021). The research
code, the full experiment ledger and the negative results live in a separate
repository; what you get here is the architecture, the trained weights, and
the scripts that reproduce the numbers below.

## How the mixer works

Each layer sweeps `log2(context)` levels in parallel. At level *i*, every
position merges the position 2^i tokens behind it, under three learned gates:

```
h_i(t) = relu( W1x·x(t) + W1s·x(t − 2^i) )²      # gate hidden (destination, source)
a_i(t) = σ( Wa_i·h_i(t) + b_a )                  # admit: take this message?
f_i(t) = σ( Wf_i·h_i(t) + b_f )                  # forget: keep what I have?
r_i(t) = σ( Wr_i·h_i(t) + b_r )                  # transmit: is my content worth sending?
V(t)  ← f_i(t)·V(t) + a_i(t)·(r_i·V)(t − 2^i)
```

Twelve levels cover a 4096-token span. Information travels by *jumping* powers
of two rather than by comparing every pair, which is where the complexity
saving comes from — and the two attention layers are there because arbitrary
in-context binding turns out to need real comparison (see Limitations).

## Install

```bash
pip install dispatcher-hybrid-2          # model + weights loading
pip install dispatcher-hybrid-2[fast]    # + Triton kernels (CUDA) for fast decoding
```

## Use

```python
import torch
from transformers import AutoTokenizer
from dispatcher_hybrid import DispatcherHybridForCausalLM

model = DispatcherHybridForCausalLM.from_pretrained("fractalego/dispatcher-hybrid-292m")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

ids = tokenizer("Photosynthesis is the process by which", return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=40, do_sample=True, temperature=0.8, top_k=50)
print(tokenizer.decode(out[0]))
```

It is a standard `PreTrainedModel`, so `AutoModelForCausalLM`, `save_pretrained`,
device maps, `labels=...` for a training loss, and `generate()` all work.
Greedy decoding on a 292M base model loops quickly; sampling is the better
default.

For anything longer than a few tokens, use `fast_generate` instead — same
tokens, far less work (see below):

```python
model = model.to("cuda", torch.bfloat16)
out = model.fast_generate(ids.to("cuda"), max_new_tokens=200, do_sample=True,
                          temperature=0.8, top_k=50)
```

## Decoding: where the architecture pays off

`generate()` re-runs the entire prefix for every new token. The incremental
decoders keep the tree's per-level state instead, so a token costs O(log t)
regardless of how long the context is — a transformer, by contrast, re-reads a
KV cache that grows with t. Measured on an A6000, 292M weights in bf16,
steady-state tokens/second:

| context | `generate()` (recompute) | eager | + CUDA graph | + Triton kernels |
|---|---|---|---|---|
| 256 | 43.8 | 44.0 | 277.6 | **534.8** |
| 1024 | 36.5 | 44.5 | 285.9 | **534.1** |
| 2048 | 17.2 | 44.4 | 282.0 | **534.1** |
| 3900 | 9.1 | 43.8 | 285.9 | **525.1** |

The incremental columns are flat in context, as the mechanism predicts; at
3,900 tokens the fused decoder is 58x the recompute path. The reason the eager
column is slow in absolute terms is that the per-token arithmetic is tiny and
the step becomes kernel-launch-bound — graph capture removes the launch
overhead, and the Triton kernels collapse the per-level serial chain into one
kernel's internal loop.

```python
from dispatcher_hybrid.decoding import FusedGraphedHybridDecoder

dec = FusedGraphedHybridDecoder(model)      # or GraphedHybridDecoder / IncrementalDecoder
logits = dec.prefill(ids[0])                # (T, vocab)
logits = dec.step(next_token)               # (vocab,), O(log t)
```

The per-level state lives in ring buffers (level l is only ever read at lag
2^l, so it keeps a 2^(l+1)-slot circular window): the whole decode state is
~480 MB at 4096 context, against ~1.9 GB for the dense layout the research
code used. Decoders also run **lockstep batches** — same prompt length per
row, one step advances all rows — which covers sampling N continuations of
one prompt (`fast_generate(..., num_return_sequences=N)`) and equal-length
prompt batches; aggregate throughput reaches ~3,000 tok/s at batch 16.
Variable-length batches are not supported: the tree state is
position-anchored, and padding would corrupt it. `fast_generate(decoder=...)` picks `"fused"`,
`"graphed"` or `"eager"`; `"auto"` takes the best available. Graph capture
costs ~0.1s once per prefill, so very short generations see less than the
table suggests.

The same Triton machinery speeds up the parallel forward, which is what
prefill and fine-tuning use:

```python
model.enable_fused_gates()   # ~1.3x forward, -28% peak memory at 4096 context
```

Both fused paths return gate pre-activations in bf16, so they shift numerics
at bf16 rounding level; the eager path stays the correctness reference and the
test suite pins both against it.

## What it scores

Measured with one harness, in fp32. The held-out CE is on 400 FineWeb-Edu
documents from a stream position the training run never reached.

| | Dispatcher-Hybrid 292M | 18-layer transformer twin |
|---|---|---|
| tokens seen | 94.9B | 6.5B |
| held-out CE | **2.5543** | 2.8097 |
| LAMBADA acc | **.347** | .285 |
| HellaSwag acc_norm | **.431** | .334 |
| PIQA acc_norm | **.679** | .636 |
| WinoGrande | **.549** | .518 |

The twin is the same recipe with attention in every layer, trained on the same
data and schedule — a controlled architecture comparison *at its own budget*,
not at this model's. The 94.9B-token result below is not budget-controlled
against anything.

Against public models of similar or larger size:

| model | params | tokens | LAMBADA acc† | HellaSwag‡ | PIQA‡ | WinoGrande‡ |
|---|---|---|---|---|---|---|
| gpt2 | 124M | WebText, multi-epoch | .326 | — | — | — |
| pythia-160m | 160M | 300B | .354 | .302 | .614 | .519 |
| mamba-130m | 130M | 300B | .443 | .353 | .645 | .519 |
| **Dispatcher-Hybrid** | **292M** | **94.9B** | **.347** | **.431** | **.679** | **.549** |
| gpt2-medium | 355M | WebText, multi-epoch | .430 | — | — | — |
| mamba-370m | 370M | 300B | .556 | .465 | .695 | .553 |
| pythia-410m | 410M | 300B | .516 | .406 | .669 | .538 |

† measured here, fp32, `lambada_openai`, greedy last word.
‡ this model's figures measured here; comparison figures from the Mamba paper
(Table 3), same harness protocol. Comparison models saw 3× the tokens on a
different corpus (the Pile, vs FineWeb-Edu here), so corpus and budget are
both confounds in either direction.

Few-shot: 5-shot ARC reaches **.544** (acc_norm), the best of every model
listed above, and the 0→5-shot gain of +5.5 points is the largest in that set.

## Limitations

**Last-word precision is the weak axis.** LAMBADA accuracy (.347) sits in the
GPT-2-small tier while perplexity (15.8) and the commonsense benchmarks sit in
the GPT-2-medium tier. The model is fluent like a larger model and imprecise
like a smaller one, and that gap is the measurable price of removing
attention. Splitting LAMBADA by whether the answer appeared earlier in the
passage: **39.6%** when it did (pure retrieval), **9.6%** when it did not.

**Arbitrary in-context binding is capacity-limited.** On a token-level MQAR
probe (random key→value pairs in context, model never trained on the format):

| key-value pairs | 8 | 32 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| accuracy | 78.1% | 74.0% | 49.7% | 24.1% | 11.1% |

That is well above fixed-state recurrent models of comparable size and below
full attention stacks — binding capacity tracks the two attention layers.
Beyond the 1024-token attention window arbitrary binding is effectively zero;
semantic retrieval still works through the tree.

**Generation is batch-1 on the fast path.** `generate()` handles batches but
recomputes the prefix; the incremental decoders are single-sequence. Batched
incremental decoding is not implemented.

**Context is 4096 tokens**, fixed by the 12 dyadic levels. Longer contexts
need additional levels (and the gates to be trained for them).

## Reproducing

```bash
# convert a research checkpoint to this format
python scripts/convert_checkpoint.py --checkpoint path/to/final.pt --out ./dispatcher-hybrid-292m

# reproduce the held-out CE (2.5543)
python scripts/reproduce_eval.py --model ./dispatcher-hybrid-292m --val-file val_window.jsonl

# decode throughput vs context length
python scripts/benchmark_decode.py --model ./dispatcher-hybrid-292m

# causality, doc-boundary isolation, decoder exactness, and parity against
# the research implementation
pytest tests/
```

The parity test is the reason to trust this extraction: on identical inputs
this implementation and the research implementation produce **bitwise
identical** logits (`max|Δlogit| = 0.0`), and the causality probe shows that
perturbing token *t* leaves every logit before *t* exactly unchanged.

## Citation

```bibtex
@article{cetoli2021dispatcher,
  title   = {Dispatcher: A Message-Passing Approach To Language Modelling},
  author  = {Cetoli, Alberto},
  journal = {arXiv preprint arXiv:2105.03994},
  year    = {2021}
}
```

MIT licensed.
