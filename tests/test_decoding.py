"""Tests for the incremental decoders.

The contract: every decoder must reproduce the parallel forward. The eager
fp32 decoder is the exactness reference (~1e-5 from reassociation); graph
capture and the Triton kernels must not add error beyond that, and in bf16 all
three must agree with each other to within dtype rounding.
"""

import os

import pytest
import torch

from dispatcher_hybrid import DispatcherHybridForCausalLM
from dispatcher_hybrid.decoding import (IncrementalDecoder, GraphedHybridDecoder,
                                        FusedGraphedHybridDecoder)

MODEL_DIR = os.environ.get("DISPATCHER_MODEL_DIR", "./dispatcher-hybrid-292m")
CUDA = torch.cuda.is_available()
try:
    import triton  # noqa: F401
    TRITON = CUDA
except ImportError:
    TRITON = False


def load(dtype=torch.float32, device="cpu"):
    if not os.path.isdir(MODEL_DIR):
        pytest.skip(f"no converted model at {MODEL_DIR}")
    return DispatcherHybridForCausalLM.from_pretrained(MODEL_DIR).to(device, dtype).eval()


def _compare(dec_cls, model, device, prompt=48, steps=12, tol=1e-4):
    torch.manual_seed(0)
    ids = torch.randint(1000, 30000, (prompt + steps,), device=device)
    with torch.no_grad():
        ref = model(ids.unsqueeze(0)).logits[0, prompt:].float()
    dec = dec_cls(model)
    dec.prefill(ids[:prompt])
    got = torch.stack([dec.step(ids[t]).float() for t in range(prompt, prompt + steps)])
    assert (got.argmax(-1) == ref.argmax(-1)).all(), "argmax disagrees with parallel forward"
    gap = (got - ref).abs().max().item()
    assert gap < tol, f"{dec_cls.__name__}: max|dlogit| {gap:.3e} exceeds {tol:.0e}"
    return gap


def test_incremental_matches_parallel_forward():
    _compare(IncrementalDecoder, load(), "cpu")


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_graphed_matches_parallel_forward():
    _compare(GraphedHybridDecoder, load(device="cuda"), "cuda")


@pytest.mark.skipif(not TRITON, reason="needs CUDA + triton")
def test_fused_matches_parallel_forward():
    _compare(FusedGraphedHybridDecoder, load(device="cuda"), "cuda")


@pytest.mark.skipif(not TRITON, reason="needs CUDA + triton")
def test_decoders_agree_in_bf16():
    """bf16 is the fast path; all three must track each other within rounding."""
    model = load(torch.bfloat16, "cuda")
    torch.manual_seed(0)
    ids = torch.randint(1000, 30000, (60,), device="cuda")
    outs = []
    for cls in (IncrementalDecoder, GraphedHybridDecoder, FusedGraphedHybridDecoder):
        dec = cls(model)
        dec.prefill(ids[:48])
        outs.append(torch.stack([dec.step(ids[t]).float() for t in range(48, 60)]))
    for other in outs[1:]:
        assert (other.argmax(-1) == outs[0].argmax(-1)).all()
        assert (other - outs[0]).abs().max().item() < 1.0


def test_fast_generate_matches_generate():
    """Greedy fast_generate must produce the same tokens as HF generate()."""
    model = load()
    ids = torch.tensor([[464, 11844, 286, 262]])
    a = model.generate(ids, max_new_tokens=12, do_sample=False)
    b = model.fast_generate(ids, max_new_tokens=12, decoder="eager")
    assert torch.equal(a, b), f"\n generate(): {a.tolist()}\n fast_generate(): {b.tolist()}"


def test_document_separator_is_rejected():
    """doc_reset models would diverge, so the decoders must refuse, not guess."""
    model = load()
    if not model.config.doc_reset:
        pytest.skip("doc_reset disabled")
    ids = torch.tensor([100, 200, model.config.eot_token, 300])
    with pytest.raises(ValueError, match="document separator"):
        IncrementalDecoder(model).prefill(ids)


@pytest.mark.skipif(not TRITON, reason="needs CUDA + triton")
def test_fused_gates_prefill_tracks_eager():
    model = load(torch.bfloat16, "cuda")
    ids = torch.randint(1000, 30000, (1, 512), device="cuda")
    with torch.no_grad():
        eager = model(ids).logits
        model.enable_fused_gates()
        fused = model(ids).logits
        model.enable_fused_gates(False)
    agree = (fused.argmax(-1) == eager.argmax(-1)).float().mean().item()
    assert agree > 0.95, f"fused-gate prefill argmax agreement only {agree:.3f}"
