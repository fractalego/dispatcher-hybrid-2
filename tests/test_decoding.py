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
def _close_argmax(got, ref, noise=0.75):
    """argmax must agree wherever the reference's top-2 margin exceeds the
    numeric noise; inside the margin a flip is a tie, not an error."""
    top2 = torch.topk(ref, 2, dim=-1).values
    decisive = (top2[..., 0] - top2[..., 1]) > noise
    agree = got.argmax(-1) == ref.argmax(-1)
    return bool((agree | ~decisive).all())


def test_decoders_agree_in_bf16():
    """bf16 is the fast path; all three must track each other within rounding.
    The fused kernel accumulates in fp32 where the torch path rounds per op,
    so bf16-scale logit differences (and argmax flips on near-ties) are
    expected; disagreement on a DECISIVE argmax is not."""
    model = load(torch.bfloat16, "cuda")
    torch.manual_seed(0)
    ids = torch.randint(1000, 30000, (60,), device="cuda")
    outs = []
    for cls in (IncrementalDecoder, GraphedHybridDecoder, FusedGraphedHybridDecoder):
        dec = cls(model)
        dec.prefill(ids[:48])
        outs.append(torch.stack([dec.step(ids[t]).float() for t in range(48, 60)]))
    for other in outs[1:]:
        assert (other - outs[0]).abs().max().item() < 1.0
        assert _close_argmax(other, outs[0])


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


def test_ring_cache_memory():
    """The per-level cache must be the ring layout, not the dense one."""
    from dispatcher_hybrid.decoding import IncrementalDecoder
    model = load()
    dec = IncrementalDecoder(model)
    dec.prefill(torch.randint(1000, 30000, (32,)))
    tree = next(c for c in dec.caches if hasattr(c, "V_ring"))
    L = tree.L
    assert tree.V_ring.shape[1] == 2 ** (L + 1) - 2          # 8190, not 13*4096
    dense = (L + 1) * model.config.max_len
    assert dense / tree.V_ring.shape[1] > 6


def test_batched_matches_single():
    """A batch of different prompts must reproduce each prompt's solo run."""
    from dispatcher_hybrid.decoding import IncrementalDecoder
    model = load()
    torch.manual_seed(0)
    prompts = torch.randint(1000, 30000, (3, 40))
    nexts = torch.randint(1000, 30000, (3, 8))
    dec = IncrementalDecoder(model)
    batched = [dec.prefill(prompts)[:, -1]]
    for j in range(8):
        batched.append(dec.step(nexts[:, j]))
    batched = torch.stack(batched, dim=1)                    # (3, 9, vocab)
    for b in range(3):
        solo_dec = IncrementalDecoder(model)
        solo = [solo_dec.prefill(prompts[b])[-1]]
        for j in range(8):
            solo.append(solo_dec.step(nexts[b, j]))
        solo = torch.stack(solo)
        gap = (batched[b] - solo).abs().max().item()
        assert gap < 1e-4, f"row {b}: batched vs solo max|d| {gap:.3e}"


@pytest.mark.skipif(not TRITON, reason="needs CUDA + triton")
def test_batched_graphed_and_fused():
    """Batched CUDA-graph decoders must match the batched eager decoder."""
    model = load(torch.bfloat16, "cuda")
    torch.manual_seed(0)
    prompts = torch.randint(1000, 30000, (4, 48), device="cuda")
    nexts = torch.randint(1000, 30000, (4, 6), device="cuda")

    def run(cls):
        dec = cls(model)
        out = [dec.prefill(prompts)[:, -1].float()]
        for j in range(6):
            out.append(dec.step(nexts[:, j]).float())
        return torch.stack(out, dim=1)

    ref = run(IncrementalDecoder)
    got = run(GraphedHybridDecoder)
    # the graphed path is the same torch arithmetic: bitwise identical
    assert torch.equal(got, ref), "graphed batched must match eager batched exactly"
    got = run(FusedGraphedHybridDecoder)
    # the Triton kernel accumulates in fp32: bf16-rounding differences allowed
    assert (got - ref).abs().max().item() < 1.0
    assert _close_argmax(got, ref)


def test_fast_generate_num_return_sequences():
    model = load()
    torch.manual_seed(0)
    ids = torch.tensor([[464, 11844, 286, 262]])
    out = model.fast_generate(ids, max_new_tokens=6, do_sample=True,
                              temperature=1.0, num_return_sequences=3,
                              decoder="eager")
    assert out.shape == (3, 10)
    assert (out[:, :4] == ids).all()


def test_fast_generate_eos_freezes_finished_rows():
    """A row that emits eos must be padded with eos while others continue."""
    model = load()
    torch.manual_seed(0)
    ids = torch.randint(1000, 30000, (2, 8))
    # choose eos = row 0's first greedy token, so row 0 finishes immediately
    eos = int(model.generate(ids[:1], max_new_tokens=1, do_sample=False)[0, -1])
    out = model.fast_generate(ids, max_new_tokens=6, decoder="eager",
                              eos_token_id=eos)
    assert int(out[0, 8]) == eos
    assert (out[0, 8:] == eos).all(), "finished row must be frozen at eos"
    assert out.shape[0] == 2 and out.shape[1] <= 14
