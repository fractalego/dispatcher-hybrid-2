"""Fused gate pipeline for the parallel (prefill / training) forward.

Requires triton + CUDA; imported lazily by the mixer, so the package does not
depend on triton unless you enable it:

    model.enable_fused_gates()      # ~1.4x forward, -34% activation memory

The pre-activations come back in bf16, so enabling this changes numerics at
bf16 rounding level; the eager path remains the correctness reference.

Profiling verdict (see ledger): the srcgatemlp champion's cost center is NOT
the level scan (6% compiled) but the gate pipeline —

    hidden = relu2(A + mask*roll(B, 2^l))     # (b, t, L, gh): 400MB at full size
    pre_x  = einsum(hidden, W_x[l]) + b_x     # x in {i, f, r}

whose (b, t, L, gh) intermediate makes every layer a memory-bandwidth event
(~55-65% of a full-size micro-step). These kernels compute hidden TILE-AT-A-
TIME in SRAM and contract immediately — the big tensor never exists, forward
or backward (the backward recomputes each tile's hidden from A and B).

Stage 1: forward, verified vs fp64 ground truth (1.3 bf16 quanta), 3.8x/5.5x
vs compiled eager at d512/full shapes.
Stage 2: doc-mask support (per-batch (b,t,L) mask), backward kernel
(recompute-in-kernel; dA/dB via fp32 atomics, dW/db via fp32 atomic
accumulation), autograd.Function, integration behind --fused-gates.

Semantics covered: gate_mode=add, gate_act=relu2, lstm update, reset gate.
Everything else falls back to eager. GPU-only (Triton); eager remains the
CPU path and the correctness reference.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_pipeline_fwd(A, B, Wi, Wf, Wr, bi, bf, br, Mask,
                       OutI, OutF, OutR,
                       T, GH, H,
                       stride_ab, stride_at,
                       stride_wl, stride_wh,
                       stride_mb, stride_mt,
                       stride_ob, stride_ot, stride_ol,
                       HAS_MASK: tl.constexpr,
                       BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
                       HP: tl.constexpr):
    # program: (batch, t-tile, level). HP = H padded to a tl.dot-legal width.
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    lev = tl.program_id(2)
    shift = 1 << lev

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = tl.arange(0, HP)
    t_ok = offs_t < T
    h_ok = offs_h < H
    src_t = offs_t - shift
    src_ok = (src_t >= 0) & t_ok                # wrap mask
    if HAS_MASK:                                # doc-reset mask (b, t, L) in {0,1}
        m = tl.load(Mask + pid_b * stride_mb + offs_t * stride_mt + lev,
                    mask=t_ok, other=0.0)
        src_ok = src_ok & (m > 0)

    acc_i = tl.zeros((BLOCK_T, HP), dtype=tl.float32)
    acc_f = tl.zeros((BLOCK_T, HP), dtype=tl.float32)
    acc_r = tl.zeros((BLOCK_T, HP), dtype=tl.float32)

    for k0 in range(0, GH, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(A + pid_b * stride_ab + offs_t[:, None] * stride_at + offs_k[None, :],
                    mask=t_ok[:, None], other=0.0)
        b_src = tl.load(B + pid_b * stride_ab + src_t[:, None] * stride_at + offs_k[None, :],
                        mask=src_ok[:, None], other=0.0)
        h = (a + b_src).to(tl.float32)
        h = tl.where(h > 0, h, 0.0)
        h = (h * h).to(tl.bfloat16)             # relu2 (bf16, matching autocast)
        wbase = lev * stride_wl + offs_h[None, :] * stride_wh + offs_k[:, None]
        wmask = h_ok[None, :]
        acc_i += tl.dot(h, tl.load(Wi + wbase, mask=wmask, other=0.0))
        acc_f += tl.dot(h, tl.load(Wf + wbase, mask=wmask, other=0.0))
        acc_r += tl.dot(h, tl.load(Wr + wbase, mask=wmask, other=0.0))

    acc_i += tl.load(bi + lev * H + offs_h, mask=h_ok, other=0.0)[None, :]
    acc_f += tl.load(bf + lev * H + offs_h, mask=h_ok, other=0.0)[None, :]
    acc_r += tl.load(br + lev * H + offs_h, mask=h_ok, other=0.0)[None, :]

    out = pid_b * stride_ob + offs_t[:, None] * stride_ot + lev * stride_ol + offs_h[None, :]
    omask = t_ok[:, None] & h_ok[None, :]
    tl.store(OutI + out, acc_i.to(tl.bfloat16), mask=omask)
    tl.store(OutF + out, acc_f.to(tl.bfloat16), mask=omask)
    tl.store(OutR + out, acc_r.to(tl.bfloat16), mask=omask)


@triton.jit
def _gate_pipeline_bwd(A, B, Wi, Wf, Wr, Mask,
                       dPi, dPf, dPr,
                       dA, dB, dWi, dWf, dWr, dbi, dbf, dbr,
                       T, GH, H,
                       stride_ab, stride_at,
                       stride_wl, stride_wh,
                       stride_mb, stride_mt,
                       stride_pb, stride_pt, stride_pl,
                       HAS_MASK: tl.constexpr,
                       BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
                       HP: tl.constexpr):
    # Same grid as forward. Recomputes h per (tile, k-chunk); emits:
    #   dA(b,t,k)   += dh * 2*relu(x)                 (atomic: levels collide)
    #   dB(b,t-s,k) += mask * dh * 2*relu(x)          (atomic: levels collide)
    #   dW_x[l]     += h^T @ dP_x                     (atomic fp32 across tiles)
    #   db_x[l]     += sum_t dP_x                     (atomic fp32)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    lev = tl.program_id(2)
    shift = 1 << lev

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = tl.arange(0, HP)
    t_ok = offs_t < T
    h_ok = offs_h < H
    src_t = offs_t - shift
    src_ok = (src_t >= 0) & t_ok
    if HAS_MASK:
        m = tl.load(Mask + pid_b * stride_mb + offs_t * stride_mt + lev,
                    mask=t_ok, other=0.0)
        src_ok = src_ok & (m > 0)

    pbase = pid_b * stride_pb + offs_t[:, None] * stride_pt + lev * stride_pl + offs_h[None, :]
    pmask = t_ok[:, None] & h_ok[None, :]
    dpi = tl.load(dPi + pbase, mask=pmask, other=0.0)
    dpf = tl.load(dPf + pbase, mask=pmask, other=0.0)
    dpr = tl.load(dPr + pbase, mask=pmask, other=0.0)

    # db_x[l]: one t-tile's column sum (deterministic per program, atomic across)
    tl.atomic_add(dbi + lev * H + offs_h, tl.sum(dpi.to(tl.float32), axis=0), mask=h_ok)
    tl.atomic_add(dbf + lev * H + offs_h, tl.sum(dpf.to(tl.float32), axis=0), mask=h_ok)
    tl.atomic_add(dbr + lev * H + offs_h, tl.sum(dpr.to(tl.float32), axis=0), mask=h_ok)

    for k0 in range(0, GH, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(A + pid_b * stride_ab + offs_t[:, None] * stride_at + offs_k[None, :],
                    mask=t_ok[:, None], other=0.0)
        b_src = tl.load(B + pid_b * stride_ab + src_t[:, None] * stride_at + offs_k[None, :],
                        mask=src_ok[:, None], other=0.0)
        x = (a + b_src).to(tl.float32)
        rx = tl.where(x > 0, x, 0.0)            # relu(x)
        h = (rx * rx).to(tl.bfloat16)           # forward's hidden chunk, recomputed

        wbase = lev * stride_wl + offs_h[:, None] * stride_wh + offs_k[None, :]
        wmask = h_ok[:, None]
        wi = tl.load(Wi + wbase, mask=wmask, other=0.0)     # (HP, K)
        wf = tl.load(Wf + wbase, mask=wmask, other=0.0)
        wr = tl.load(Wr + wbase, mask=wmask, other=0.0)

        # dh = sum_x dP_x @ W_x   -> (T, K)
        dh = tl.dot(dpi, wi) + tl.dot(dpf, wf) + tl.dot(dpr, wr)
        dx = dh * 2.0 * rx                       # d relu2 = 2*relu(x)

        aptr = pid_b * stride_ab + offs_t[:, None] * stride_at + offs_k[None, :]
        tl.atomic_add(dA + aptr, dx, mask=t_ok[:, None])
        bptr = pid_b * stride_ab + src_t[:, None] * stride_at + offs_k[None, :]
        tl.atomic_add(dB + bptr, dx, mask=src_ok[:, None])

        # dW_x[l] += h^T @ dP_x  -> (K, HP) stored as (HP, K) layout of W
        dwi = tl.dot(tl.trans(h), dpi)           # (K, HP)
        dwf = tl.dot(tl.trans(h), dpf)
        dwr = tl.dot(tl.trans(h), dpr)
        dwptr = lev * stride_wl + offs_h[None, :] * stride_wh + offs_k[:, None]
        dwmask = h_ok[None, :]
        tl.atomic_add(dWi + dwptr, dwi, mask=dwmask)
        tl.atomic_add(dWf + dwptr, dwf, mask=dwmask)
        tl.atomic_add(dWr + dwptr, dwr, mask=dwmask)


# ---- torch.library custom-op registration (2026-08-21, task #30) ----
# The autograd.Function wrapper detonated under torch.compile three ways
# (d512 0.95x, A6000 inductor crash, 5090 >1h hang): dynamo traced INTO it.
# Registered custom ops are opaque atoms to dynamo/inductor — the composition
# fix. Fake (meta) fns give the compiler shapes; autograd is wired via
# register_autograd with a bwd custom op.

from typing import List, Optional


@torch.library.custom_op("dispatcher::gate_preacts_fwd", mutates_args=())
def _gate_preacts_fwd_op(A: torch.Tensor, B: torch.Tensor,
                         Wi: torch.Tensor, Wf: torch.Tensor, Wr: torch.Tensor,
                         bi: torch.Tensor, bf: torch.Tensor, br: torch.Tensor,
                         n_lv: int, mask: Optional[torch.Tensor]) -> List[torch.Tensor]:
    b, T, GH = A.shape
    H = Wi.shape[1]
    A = A.contiguous().to(torch.bfloat16)
    B = B.contiguous().to(torch.bfloat16)
    Wi_, Wf_, Wr_ = (w[:n_lv].contiguous().to(torch.bfloat16) for w in (Wi, Wf, Wr))
    bi_, bf_, br_ = (x[:n_lv].contiguous().float() for x in (bi, bf, br))
    mask_ = mask.contiguous().to(torch.bfloat16) if mask is not None else A
    # three SEPARATE allocations: custom-op outputs may not alias each other
    oi = torch.empty(b, T, n_lv, H, device=A.device, dtype=torch.bfloat16)
    of = torch.empty_like(oi)
    orr = torch.empty_like(oi)
    HP = max(16, triton.next_power_of_2(H))
    grid = (b, triton.cdiv(T, 64), n_lv)
    _gate_pipeline_fwd[grid](
        A, B, Wi_, Wf_, Wr_, bi_, bf_, br_, mask_,
        oi, of, orr,
        T, GH, H,
        A.stride(0), A.stride(1),
        Wi_.stride(0), Wi_.stride(1),
        mask_.stride(0), mask_.stride(1) if mask is not None else 0,
        oi.stride(0), oi.stride(1), oi.stride(2),
        HAS_MASK=mask is not None,
        BLOCK_T=64, BLOCK_K=64, HP=HP)
    return [oi, of, orr]


@_gate_preacts_fwd_op.register_fake
def _(A, B, Wi, Wf, Wr, bi, bf, br, n_lv, mask):
    b, T, _ = A.shape
    H = Wi.shape[1]
    mk = lambda: torch.empty(b, T, n_lv, H, device=A.device, dtype=torch.bfloat16)
    return [mk(), mk(), mk()]


@torch.library.custom_op("dispatcher::gate_preacts_bwd", mutates_args=())
def _gate_preacts_bwd_op(A: torch.Tensor, B: torch.Tensor,
                         Wi: torch.Tensor, Wf: torch.Tensor, Wr: torch.Tensor,
                         dPi: torch.Tensor, dPf: torch.Tensor, dPr: torch.Tensor,
                         n_lv: int, mask: Optional[torch.Tensor]) -> List[torch.Tensor]:
    b, T, GH = A.shape
    H = Wi.shape[1]
    dA = torch.zeros(b, T, GH, device=A.device, dtype=torch.float32)
    dB = torch.zeros_like(dA)
    dWi = torch.zeros(n_lv, H, GH, device=A.device, dtype=torch.float32)
    dWf = torch.zeros_like(dWi)
    dWr = torch.zeros_like(dWi)
    dbi = torch.zeros(n_lv, H, device=A.device, dtype=torch.float32)
    dbf = torch.zeros_like(dbi)
    dbr = torch.zeros_like(dbi)
    dPi = dPi.contiguous().to(torch.bfloat16)
    dPf = dPf.contiguous().to(torch.bfloat16)
    dPr = dPr.contiguous().to(torch.bfloat16)
    HP = max(16, triton.next_power_of_2(H))
    m = mask.contiguous().to(torch.bfloat16) if mask is not None else A
    grid = (b, triton.cdiv(T, 64), n_lv)
    _gate_pipeline_bwd[grid](
        A, B, Wi, Wf, Wr, m,
        dPi, dPf, dPr,
        dA, dB, dWi, dWf, dWr, dbi, dbf, dbr,
        T, GH, H,
        A.stride(0), A.stride(1),
        Wi.stride(0), Wi.stride(1),
        m.stride(0), m.stride(1) if mask is not None else 0,
        dPi.stride(0), dPi.stride(1), dPi.stride(2),
        HAS_MASK=mask is not None,
        BLOCK_T=64, BLOCK_K=64, HP=HP)
    return [dA, dB, dWi, dWf, dWr, dbi, dbf, dbr]


@_gate_preacts_bwd_op.register_fake
def _(A, B, Wi, Wf, Wr, dPi, dPf, dPr, n_lv, mask):
    b, T, GH = A.shape
    H = Wi.shape[1]
    f32 = lambda *sh: torch.empty(*sh, device=A.device, dtype=torch.float32)
    return [f32(b, T, GH), f32(b, T, GH),
            f32(n_lv, H, GH), f32(n_lv, H, GH), f32(n_lv, H, GH),
            f32(n_lv, H), f32(n_lv, H), f32(n_lv, H)]


def _setup_context(ctx, inputs, output):
    A, B, Wi, Wf, Wr, bi, bf, br, n_lv, mask = inputs
    ctx.save_for_backward(A.to(torch.bfloat16), B.to(torch.bfloat16),
                          Wi[:n_lv].contiguous().to(torch.bfloat16),
                          Wf[:n_lv].contiguous().to(torch.bfloat16),
                          Wr[:n_lv].contiguous().to(torch.bfloat16))
    ctx.n_lv = n_lv
    ctx.mask = mask.to(torch.bfloat16) if mask is not None else None
    ctx.w_levels = Wi.shape[0]


def _backward(ctx, grads):
    dPi, dPf, dPr = grads
    A, B, Wi_, Wf_, Wr_ = ctx.saved_tensors
    n_lv, w_levels = ctx.n_lv, ctx.w_levels
    outs = torch.ops.dispatcher.gate_preacts_bwd(
        A, B, Wi_, Wf_, Wr_, dPi, dPf, dPr, n_lv, ctx.mask)
    dA, dB, dWi, dWf, dWr, dbi, dbf, dbr = outs
    H, GH = dWi.shape[1], dWi.shape[2]

    def pad(g, full_l):
        if full_l == n_lv:
            return g
        out = torch.zeros((full_l,) + g.shape[1:], device=g.device, dtype=g.dtype)
        out[:n_lv] = g
        return out

    return (dA.to(torch.bfloat16), dB.to(torch.bfloat16),
            pad(dWi, w_levels), pad(dWf, w_levels), pad(dWr, w_levels),
            pad(dbi, w_levels), pad(dbf, w_levels), pad(dbr, w_levels),
            None, None)


torch.library.register_autograd("dispatcher::gate_preacts_fwd", _backward,
                                setup_context=_setup_context)


def fused_gate_preacts(A, B, Wi, Wf, Wr, bi, bf, br, n_lv, mask=None):
    """(pre_i, pre_f, pre_r), each (b, t, n_lv, h) bf16 — differentiable custom
    op (opaque to torch.compile), matching einsum('btld,lhg->btlh',
    relu2(A.unsqueeze(2) + m*roll(B)), W_x) + b_x. mask: optional (b, t, n_lv)
    doc-reset mask (wrap mask always applied in-kernel)."""
    out = torch.ops.dispatcher.gate_preacts_fwd(A, B, Wi, Wf, Wr, bi, bf, br,
                                                n_lv, mask)
    return out[0], out[1], out[2]
