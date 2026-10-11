"""Triton kernels for the per-token decode step (requires triton + CUDA).

The graphed decoder still ran the level scan as ~13 serial micro-kernels per
layer; at a ~3-5us launch floor each that is ~1ms/token across 17 layers,
which was the entire measured deficit against a transformer at short context.
Two kernels remove it:

    step_gates_kernel   grid (B, L)  — per-level gate hidden + i/f/r readouts,
                                       which are independent across levels
    step_scan_kernel    grid (B, H)  — the whole level recurrence per head,
                                       held in registers

The V state lives in per-level RING buffers: level j's state is only read at
lag 2^j, so it keeps 2^(j+1) slots at offset 2^(j+1)-2, and the slot index is
a bitmask. Both quantities are pure bit arithmetic in the level index, so the
kernels compute them inline.

Imported lazily: the package does not require triton unless you ask for the
fused decoder.
"""

import triton
import triton.language as tl


@triton.jit
def step_gates_kernel(A, B_hist, Wi, Wf, Wr, bi, bf, br, T_buf,
                      IG, FG, RG_hist, T_max,
                      GH: tl.constexpr, H: tl.constexpr, L: tl.constexpr,
                      BK: tl.constexpr):
    b = tl.program_id(0)
    lev = tl.program_id(1)
    t = tl.load(T_buf)
    src = t - (1 << lev)
    ok = src >= 0
    src_c = tl.maximum(src, 0)
    offs_h = tl.arange(0, H)
    acc_i = tl.zeros((H,), dtype=tl.float32)
    acc_f = tl.zeros((H,), dtype=tl.float32)
    acc_r = tl.zeros((H,), dtype=tl.float32)
    for k0 in range(0, GH, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(A + b * GH + offs_k).to(tl.float32)
        s = tl.load(B_hist + (b * T_max + src_c) * GH + offs_k).to(tl.float32) * ok
        h = tl.maximum(a + s, 0.0)
        h = h * h                                        # relu squared
        wbase = lev * H * GH + offs_h[:, None] * GH + offs_k[None, :]
        acc_i += tl.sum(tl.load(Wi + wbase).to(tl.float32) * h[None, :], axis=1)
        acc_f += tl.sum(tl.load(Wf + wbase).to(tl.float32) * h[None, :], axis=1)
        acc_r += tl.sum(tl.load(Wr + wbase).to(tl.float32) * h[None, :], axis=1)
    okf = ok.to(tl.float32)
    ig = tl.sigmoid(acc_i + tl.load(bi + lev * H + offs_h).to(tl.float32)) * okf
    fg = tl.sigmoid(acc_f + tl.load(bf + lev * H + offs_h).to(tl.float32))
    fg = fg * okf + (1.0 - okf)                          # masked levels keep state
    rg = tl.sigmoid(acc_r + tl.load(br + lev * H + offs_h).to(tl.float32))
    tl.store(IG + b * L * H + lev * H + offs_h, ig)
    tl.store(FG + b * L * H + lev * H + offs_h, fg)
    tl.store(RG_hist + ((b * T_max + t) * L + lev) * H + offs_h,   # (B, T, L, H)
             rg.to(RG_hist.dtype.element_ty))


@triton.jit
def step_scan_kernel(V0, IG, FG, RG_hist, V_ring, T_buf, Y,
                     T_max, S,
                     L: tl.constexpr, HD: tl.constexpr, H: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.load(T_buf)
    d: tl.constexpr = H * HD
    offs = tl.arange(0, HD)
    ring_b = V_ring + b * S * d
    V = tl.load(V0 + b * d + h * HD + offs).to(tl.float32)
    # ring j: offset 2^(j+1)-2, mask 2^(j+1)-1. Write level 0 at t.
    tl.store(ring_b + (0 + (t & 1)) * d + h * HD + offs, V.to(V_ring.dtype.element_ty))
    for lev in range(L):
        src = t - (1 << lev)
        ok = src >= 0
        src_c = tl.maximum(src, 0)
        ig = tl.load(IG + b * L * H + lev * H + h)
        fg = tl.load(FG + b * L * H + lev * H + h)
        rg = tl.load(RG_hist + ((b * T_max + src_c) * L + lev) * H + h).to(tl.float32) * ok
        slot_r = ((2 << lev) - 2) + (src_c & ((2 << lev) - 1))
        v_src = tl.load(ring_b + slot_r * d + h * HD + offs).to(tl.float32) * ok
        V = fg * V + ig * rg * v_src
        if lev + 1 < L:
            slot_w = ((2 << (lev + 1)) - 2) + (t & ((2 << (lev + 1)) - 1))
            tl.store(ring_b + slot_w * d + h * HD + offs, V.to(V_ring.dtype.element_ty))
    tl.store(Y + b * d + h * HD + offs, V.to(Y.dtype.element_ty))
