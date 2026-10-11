"""Decode throughput vs context length: incremental decoders against recompute.

    python scripts/benchmark_decode.py --model ./dispatcher-hybrid-292m

Reports steady-state tokens/second (prefill and CUDA-graph capture excluded,
since both amortize over a long generation). The point of interest is the
SLOPE: tree layers read a fixed number of cached vectors per token, so the
rate should be flat in context, while recomputing the prefix degrades.
"""

import argparse
import time

import torch

from dispatcher_hybrid import DispatcherHybridForCausalLM
from dispatcher_hybrid.decoding import (IncrementalDecoder, GraphedHybridDecoder,
                                        FusedGraphedHybridDecoder)


def rate(fn, n):
    fn()                                      # warm
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return n / (time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./dispatcher-hybrid-292m")
    ap.add_argument("--contexts", type=int, nargs="+", default=[256, 1024, 2048, 3900])
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DispatcherHybridForCausalLM.from_pretrained(args.model).to(
        device, getattr(torch, args.dtype)).eval()

    classes = [("eager", IncrementalDecoder)]
    if device == "cuda":
        classes.append(("graphed", GraphedHybridDecoder))
        try:
            import triton  # noqa: F401
            classes.append(("fused", FusedGraphedHybridDecoder))
        except ImportError:
            pass

    print(f"{'context':>8} | " + " | ".join(f"{n:>9}" for n, _ in classes) + " |  recompute")
    print("-" * (10 + 12 * len(classes) + 13))
    for ctx in args.contexts:
        ids = torch.randint(1000, 30000, (ctx,), device=device)
        row = []
        for _, cls in classes:
            dec = cls(model)
            dec.prefill(ids)
            tok = ids[-1].clone()
            row.append(rate(lambda: dec.step(tok), args.steps))
        with torch.no_grad():
            batch = ids.unsqueeze(0)
            recompute = rate(lambda: model(batch).logits[:, -1], max(4, args.steps // 8))
        print(f"{ctx:>8} | " + " | ".join(f"{r:9.1f}" for r in row) + f" | {recompute:9.1f}")
    print("\ntokens/second, steady state (prefill and graph capture excluded)")


if __name__ == "__main__":
    main()
