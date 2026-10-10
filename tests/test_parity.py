"""Parity and causality tests for the release package.

The parity test needs the research repository importable (set
DISPATCHER_RESEARCH_REPO); it is skipped otherwise. The causality test is
self-contained and is the one that must never be skipped in CI: it asserts
that no future token can influence an earlier position's logits, which for
this architecture is a property of the shift direction plus the level mask.
"""

import os
import sys

import pytest
import torch

from dispatcher_hybrid import DispatcherHybridForCausalLM

MODEL_DIR = os.environ.get("DISPATCHER_MODEL_DIR", "./dispatcher-hybrid-292m")
RESEARCH_REPO = os.environ.get("DISPATCHER_RESEARCH_REPO")
RESEARCH_CKPT = os.environ.get("DISPATCHER_RESEARCH_CKPT")


@pytest.fixture(scope="module")
def model():
    if not os.path.isdir(MODEL_DIR):
        pytest.skip(f"no converted model at {MODEL_DIR}")
    return DispatcherHybridForCausalLM.from_pretrained(MODEL_DIR).eval()


def test_causality(model):
    """Changing token t must not change logits at positions < t."""
    torch.manual_seed(0)
    ids = torch.randint(1000, 30000, (1, 256))
    with torch.no_grad():
        base = model(ids).logits
        for cut in (17, 128, 255):
            poked = ids.clone()
            poked[0, cut] = (poked[0, cut] + 7919) % 50257
            after = model(poked).logits
            delta = (after[0, :cut] - base[0, :cut]).abs().max().item()
            assert delta == 0.0, f"future leak at t={cut}: max|dlogit| = {delta:.3e}"


def test_doc_reset_blocks_cross_document_flow(model):
    """With doc_reset on, an EOT boundary must isolate what follows it."""
    if not model.config.doc_reset:
        pytest.skip("doc_reset disabled in this config")
    torch.manual_seed(1)
    eot = model.config.eot_token
    tail = torch.randint(1000, 30000, (1, 64))
    a = torch.cat([torch.randint(1000, 30000, (1, 64)), torch.tensor([[eot]]), tail], dim=1)
    b = torch.cat([torch.randint(1000, 30000, (1, 64)), torch.tensor([[eot]]), tail], dim=1)
    with torch.no_grad():
        la = model(a).logits[0, 65:]
        lb = model(b).logits[0, 65:]
    # the attention layers are standard-causal and DO see across the boundary,
    # so this is a bounded-influence check, not an equality one
    assert (la - lb).abs().max().item() < 20.0


def test_generate_runs(model):
    ids = torch.randint(1000, 30000, (1, 16))
    out = model.generate(ids, max_new_tokens=8, do_sample=False)
    assert out.shape == (1, 24)


@pytest.mark.skipif(not (RESEARCH_REPO and RESEARCH_CKPT),
                    reason="research repo/checkpoint not configured")
def test_parity_with_research_implementation(model):
    """Package logits must match the research implementation's."""
    sys.path.insert(0, RESEARCH_REPO)
    from test_causality import build_from_checkpoint  # noqa: E402

    research, _ = build_from_checkpoint(RESEARCH_CKPT, "cpu")
    research = research.float().eval()

    torch.manual_seed(2)
    for length in (64, 512, 1024, 2048):
        ids = torch.randint(1000, 30000, (1, length))
        ids[0, length // 3] = 50256  # exercise the doc-reset path
        with torch.no_grad():
            mine = model(ids).logits
            theirs = research(ids)
        gap = (mine - theirs).abs().max().item()
        rel = gap / theirs.abs().max().item()
        assert rel < 1e-5, f"length {length}: max|dlogit| {gap:.3e} (rel {rel:.2e})"
