"""Triton kernels for the per-token decode step (requires triton + CUDA).

The graphed decoder still ran the level scan as ~13 serial micro-kernels per
layer; at a ~3-5us launch floor each that is ~1ms/token across 17 layers,
which was the entire measured deficit against a transformer at short context.
Two kernels remove it:

    step_gates_kernel   grid (L,)  — per-level gate hidden + i/f/r readouts,
                                     which are independent across levels
    step_scan_kernel    grid (H,)  — the whole level recurrence per head, held
                                     in registers

Imported lazily: the package does not require triton unless you ask for the
fused decoder.
"""

import triton
import triton.language as tl


@triton.jit
def step_gates_kernel(A, B_hist, Wi, Wf, Wr, bi, bf, br, T_buf,
                      IG, FG, RG_hist,
                      GH: tl.constexpr, H: tl.constexpr, BK: tl.constexpr):
    lev = tl.program_id(0)
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
        a = tl.load(A + offs_k).to(tl.float32)
        b = tl.load(B_hist + src_c * GH + offs_k).to(tl.float32) * ok
        h = tl.maximum(a + b, 0.0)
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
    tl.store(IG + lev * H + offs_h, ig)
    tl.store(FG + lev * H + offs_h, fg)
    tl.store(RG_hist + t * (tl.num_programs(0) * H) + lev * H + offs_h,   # (T, L, H)
             rg.to(RG_hist.dtype.element_ty))


@triton.jit
def step_scan_kernel(V0, IG, FG, RG_hist, V_lev, T_buf, Y,
                     T_max, L: tl.constexpr, HD: tl.constexpr, H: tl.constexpr):
    h = tl.program_id(0)
    t = tl.load(T_buf)
    d: tl.constexpr = H * HD
    offs = tl.arange(0, HD)
    V = tl.load(V0 + h * HD + offs).to(tl.float32)
    tl.store(V_lev + t * d + h * HD + offs, V.to(V_lev.dtype.element_ty))
    for lev in range(L):
        src = t - (1 << lev)
        ok = src >= 0
        src_c = tl.maximum(src, 0)
        ig = tl.load(IG + lev * H + h)
        fg = tl.load(FG + lev * H + h)
        rg = tl.load(RG_hist + src_c * L * H + lev * H + h).to(tl.float32) * ok
        v_src = tl.load(V_lev + lev * T_max * d + src_c * d + h * HD + offs).to(tl.float32) * ok
        V = fg * V + ig * rg * v_src
        tl.store(V_lev + (lev + 1) * T_max * d + t * d + h * HD + offs,
                 V.to(V_lev.dtype.element_ty))
    tl.store(Y + h * HD + offs, V.to(Y.dtype.element_ty))
