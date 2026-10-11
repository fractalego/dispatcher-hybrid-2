"""Exact incremental decoding — the architecture's actual serving advantage.

A transformer layer re-reads its whole KV cache for every generated token:
O(t) per layer, per token, so decode slows as context grows. A tree layer
reads one cached vector per level — 12 of them at 4096 context — which is
CONSTANT in t. This hybrid pays the O(t) toll in 2 layers instead of 18.

`DispatcherHybridForCausalLM.generate()` recomputes the whole prefix each
step and realizes none of that. The decoders here do.

Three implementations, same arithmetic, increasing engineering:

    IncrementalDecoder        exact, eager PyTorch, fp32 by default
    GraphedHybridDecoder      the whole step captured in one CUDA graph
    FusedGraphedHybridDecoder + 2 Triton kernels for the gate/scan chain

The ladder exists because the per-token work is tiny: at ~13 serial
micro-kernels per layer and a ~3-5us launch floor each, the eager decoder is
launch-bound and loses to a transformer despite doing less arithmetic. Graph
capture removes the launch overhead; the fused kernels collapse the serial
chain into one kernel's internal loop. Measured on an A6000 at full-size
shapes: 41 -> 278 -> 546 tok/s at 4096 context, against 362 for the matched
transformer (which decays to 155 at 32k while these stay flat).

Caches, per tree layer, batch 1:
    B_hist  (T, gh)        gate source vectors, read at t-2^l by level l
    rg_hist (T, L, H)      transmit gates, kept at their SOURCE position
    V_lev   (L+1, T, d)    per-level scan states; level l at t reads level l
                           at t-2^l
Attention layers keep a standard windowed KV cache with RoPE at absolute
position.

Known limits (both inherited from the research implementation):
  * batch 1;
  * V_lev is the naive dense layout -- level l only ever needs its last 2^l
    slots, so a ring buffer would cut tree-cache memory ~13x. At 4096 context
    this costs ~1.75GB in bf16 across 16 tree layers.
"""

import math

import torch
import torch.nn.functional as F

from .modeling_dispatcher import AttentionBlock


class _TreeLayerCache:
    def __init__(self, mixer, max_len, device, dtype):
        gh = mixer.gate_hidden.out_features
        self.L = mixer.levels
        self.H = mixer.num_heads
        self.hd = mixer.head_dim
        d = self.H * self.hd
        self.B_hist = torch.zeros(max_len, gh, device=device, dtype=dtype)
        self.rg_hist = torch.zeros(max_len, self.L, self.H, device=device, dtype=dtype)
        self.V_lev = torch.zeros(self.L + 1, max_len, d, device=device, dtype=dtype)


class _AttnCache:
    def __init__(self, attn, max_len, device, dtype):
        self.k = torch.zeros(max_len, attn.num_heads, attn.head_dim,
                             device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        # materialized RoPE tables: the model computes these lazily, but graph
        # capture needs stable device tensors to index
        cos, sin = attn.rope_tables(max_len, device, dtype)
        self.rope_cos, self.rope_sin = cos, sin


def _rope_at(cache, x, pos):
    # x: (H, hd) at absolute position pos
    cos, sin = cache.rope_cos[pos], cache.rope_sin[pos]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


def _check_single_document(model, idx):
    """The decoders implement the wrap mask only, not the doc-reset mask, so a
    prompt spanning documents would diverge from the parallel forward."""
    if getattr(model.config, "doc_reset", False):
        if bool((idx == model.config.eot_token).any()):
            raise ValueError(
                f"prompt contains the document separator (token "
                f"{model.config.eot_token}) and this model was trained with "
                f"doc_reset=True; the incremental decoders assume a single "
                f"document. Remove the separator, or use model.generate().")


class IncrementalDecoder:
    """Exact per-token decoder.

        dec = IncrementalDecoder(model)
        logits = dec.prefill(ids)      # (T, vocab) — parallel, fills the caches
        logits = dec.step(token)       # (vocab,)   — one token, O(log t) work
    """

    def __init__(self, model, max_len=None, dtype=None):
        self.m = model
        self.max_len = max_len or model.config.max_len
        param_dtype = next(model.parameters()).dtype
        # the caches and the running activation must share the weights' dtype:
        # a mixed-dtype matmul errors, and autocasting inside a CUDA graph
        # would re-cast every weight on every token. Cast the MODEL to choose
        # precision (bf16 roughly doubles decode throughput).
        if dtype is not None and dtype != param_dtype:
            raise ValueError(
                f"decoder dtype {dtype} != model dtype {param_dtype}; cast the "
                f"model first, e.g. model.to(torch.bfloat16)")
        self.dtype = param_dtype
        dev = next(model.parameters()).device
        self.caches = []
        for blk in model.blocks:
            if isinstance(blk, AttentionBlock):
                self.caches.append(_AttnCache(blk.attn, self.max_len, dev, self.dtype))
            else:
                self.caches.append(_TreeLayerCache(blk.mixer, self.max_len, dev, self.dtype))
        self.t = 0

    # ---------- tree layer ----------
    def _tree_prefill(self, blk, cache, x):
        # replicate the parallel mixer while stashing B / rg / per-level V
        mx = blk.mixer
        T = x.shape[0]
        h1 = blk.norm1(x)
        A = mx.gate_hidden(h1)                                    # (T, gh)
        B = mx.gate_hidden_src(h1)
        cache.B_hist[:T] = B
        L, H, hd = cache.L, cache.H, cache.hd
        n_lv = min(L, int(math.log2(T)) + 1) if T > 1 else 1
        V = mx.linear_in(h1).view(T, H, hd)
        cache.V_lev[0, :T] = V.reshape(T, H * hd)
        rolled = torch.stack([torch.roll(B, shifts=2 ** l, dims=0)
                              for l in range(n_lv)], dim=1)       # (T, n_lv, gh)
        wrap = torch.zeros(T, n_lv, 1, device=x.device, dtype=x.dtype)
        for l in range(n_lv):
            wrap[2 ** l:, l] = 1.0
        hidden = torch.relu(A.unsqueeze(1) + rolled * wrap) ** 2
        pre_i = torch.einsum("tlg,lhg->tlh", hidden, mx.gate_out[:n_lv]) + mx.gate_out_bias[:n_lv]
        pre_f = torch.einsum("tlg,lhg->tlh", hidden, mx.forget_out[:n_lv]) + mx.forget_out_bias[:n_lv]
        pre_r = torch.einsum("tlg,lhg->tlh", hidden, mx.reset_out[:n_lv]) + mx.reset_out_bias[:n_lv]
        ig = torch.sigmoid(pre_i) * wrap
        fg = torch.sigmoid(pre_f) * wrap + (1.0 - wrap)
        rg = torch.sigmoid(pre_r)
        cache.rg_hist[:T, :n_lv] = rg
        if n_lv < L:
            # levels past the prefill horizon contribute no message yet, but
            # later steps will read these positions' transmit gates
            hid0 = torch.relu(A) ** 2
            for l in range(n_lv, L):
                cache.rg_hist[:T, l] = torch.sigmoid(
                    hid0 @ mx.reset_out[l].T + mx.reset_out_bias[l])
        for l in range(n_lv):
            src = rg[:, l].unsqueeze(-1) * V                      # transmit at source
            msg = torch.roll(src, shifts=2 ** l, dims=0)
            V = fg[:, l].unsqueeze(-1) * V + ig[:, l].unsqueeze(-1) * msg
            cache.V_lev[l + 1, :T] = V.reshape(T, H * hd)
        for l in range(n_lv, L):
            cache.V_lev[l + 1, :T] = cache.V_lev[l, :T]
        x = x + mx.linear_out(V.reshape(T, H * hd))
        return x + blk.ffn(blk.norm2(x))

    def _tree_step(self, blk, cache, x, t):
        # x: (d,) — one position. Every level's gate math is batched into a
        # handful of ops; only the L-step V recurrence stays sequential.
        mx = blk.mixer
        L, H, hd = cache.L, cache.H, cache.hd
        h1 = blk.norm1(x)
        A = mx.gate_hidden(h1)
        cache.B_hist[t] = mx.gate_hidden_src(h1)
        V = mx.linear_in(h1).view(H, hd)
        cache.V_lev[0, t] = V.reshape(-1)
        arangeL = torch.arange(L, device=x.device)
        src_t = t - torch.pow(2, arangeL)
        ok = src_t >= 0
        src_c = src_t.clamp(min=0)
        Bg = cache.B_hist[src_c] * ok.unsqueeze(-1)               # (L, gh)
        hid = torch.relu(A.unsqueeze(0) + Bg) ** 2
        okf = ok.unsqueeze(-1)
        i_ = torch.sigmoid(torch.einsum("lg,lhg->lh", hid, mx.gate_out)
                           + mx.gate_out_bias) * okf
        f_ = torch.sigmoid(torch.einsum("lg,lhg->lh", hid, mx.forget_out)
                           + mx.forget_out_bias)
        f_ = f_ * okf + (~okf).to(f_.dtype)                       # masked: f = 1
        r_ = torch.sigmoid(torch.einsum("lg,lhg->lh", hid, mx.reset_out)
                           + mx.reset_out_bias)
        cache.rg_hist[t] = r_
        rg_src = cache.rg_hist[src_c, arangeL]                    # (L, H)
        V_src = cache.V_lev[arangeL, src_c].view(L, H, hd)
        msg = rg_src.unsqueeze(-1) * V_src
        for l in range(L):
            V = f_[l].unsqueeze(-1) * V + i_[l].unsqueeze(-1) * msg[l]
            cache.V_lev[l + 1, t] = V.reshape(-1)
        x = x + mx.linear_out(V.reshape(-1))
        return x + blk.ffn(blk.norm2(x))

    # ---------- attention layer (windowed, RoPE) ----------
    def _attn_prefill(self, blk, cache, x):
        T = x.shape[0]
        at = blk.attn
        h1 = blk.norm1(x)
        q = at.linear_q(h1).view(T, at.num_heads, at.head_dim)
        k = at.linear_k(h1).view(T, at.num_heads, at.head_dim)
        v = at.linear_v(h1).view(T, at.num_heads, at.head_dim)
        qb = at.apply_rope(q.transpose(0, 1).unsqueeze(0), T)[0].contiguous()
        kb = at.apply_rope(k.transpose(0, 1).unsqueeze(0), T)[0].contiguous()
        cache.k[:T] = kb.transpose(0, 1)
        cache.v[:T] = v
        vb = v.transpose(0, 1).contiguous()
        if 0 < at.window < T:
            tt = torch.arange(T, device=x.device)
            band = (tt[:, None] >= tt[None, :]) & (tt[:, None] - tt[None, :] < at.window)
            y = F.scaled_dot_product_attention(qb.unsqueeze(0), kb.unsqueeze(0),
                                               vb.unsqueeze(0), attn_mask=band)[0]
        else:
            # 4-D input keeps SDPA on the flash path; 3-D falls back to the
            # math backend at long T and materializes the score matrix
            y = F.scaled_dot_product_attention(qb.unsqueeze(0), kb.unsqueeze(0),
                                               vb.unsqueeze(0), is_causal=True)[0]
        x = x + at.linear_out(y.transpose(0, 1).reshape(T, at.embed_dim))
        return x + blk.ffn(blk.norm2(x))

    def _attn_step(self, blk, cache, x, t):
        at = blk.attn
        h1 = blk.norm1(x)
        q = _rope_at(cache, at.linear_q(h1).view(at.num_heads, at.head_dim), t)
        cache.k[t] = _rope_at(cache, at.linear_k(h1).view(at.num_heads, at.head_dim), t)
        cache.v[t] = at.linear_v(h1).view(at.num_heads, at.head_dim)
        lo = max(0, t + 1 - at.window) if at.window > 0 else 0
        k = cache.k[lo:t + 1].transpose(0, 1)                     # (H, w, hd)
        v = cache.v[lo:t + 1].transpose(0, 1)
        att = torch.softmax((q.unsqueeze(1) * k).sum(-1) / math.sqrt(at.head_dim), dim=-1)
        y = (att.unsqueeze(-1) * v).sum(1).reshape(-1)
        x = x + at.linear_out(y)
        return x + blk.ffn(blk.norm2(x))

    # ---------- public ----------
    @torch.no_grad()
    def prefill(self, idx):
        """idx: (T,) token ids. Returns (T, vocab) logits."""
        idx = idx.view(-1)
        _check_single_document(self.m, idx)
        if idx.shape[0] > self.max_len:
            raise ValueError(f"prompt of {idx.shape[0]} exceeds max_len {self.max_len}")
        x = self.m.embedding(idx).to(self.dtype)
        for blk, cache in zip(self.m.blocks, self.caches):
            x = (self._attn_prefill(blk, cache, x) if isinstance(blk, AttentionBlock)
                 else self._tree_prefill(blk, cache, x))
        self.t = idx.shape[0]
        return self.m.lm_head(self.m.norm_out(x))

    @torch.no_grad()
    def step(self, token):
        """token: scalar/1-element tensor. Returns (vocab,) logits."""
        if self.t >= self.max_len:
            raise ValueError(f"context exhausted at max_len {self.max_len}")
        t = self.t
        x = self.m.embedding(token.view(1))[0].to(self.dtype)
        for blk, cache in zip(self.m.blocks, self.caches):
            x = (self._attn_step(blk, cache, x, t) if isinstance(blk, AttentionBlock)
                 else self._tree_step(blk, cache, x, t))
        self.t = t + 1
        return self.m.lm_head(self.m.norm_out(x))


# ======================================================================
# CUDA-graph decoder. The per-token work is so small that kernel-launch
# overhead dominates it; capturing the whole step as one replayable graph is
# what makes the O(log t) arithmetic visible in wall-clock. Attention reads a
# padded fixed-size cache with a validity mask, which is mathematically
# identical to the sliced version but has a capture-stable shape.
# ======================================================================


class _GraphedBase:
    def _capture(self, step_fn):
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):                      # warmup on a side stream
                step_fn()
                self.t_buf.fill_(self.t)            # undo warmup advances
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            step_fn()
        self.graph = g

    @torch.no_grad()
    def step(self, token):
        self.token_buf.copy_(token.view(1))
        self.graph.replay()
        self.t += 1
        # the graph writes into a fixed output buffer that the next replay
        # overwrites; return a copy so callers can keep the result
        return self.logits_out.clone()


class GraphedHybridDecoder(_GraphedBase):
    """Eager prefill, then one CUDA-graph replay per generated token."""

    def __init__(self, model, max_len=None, dtype=None):
        if not torch.cuda.is_available():
            raise RuntimeError("GraphedHybridDecoder requires CUDA")
        self.inner = IncrementalDecoder(model, max_len, dtype=dtype)
        self.m = model
        self.max_len = self.inner.max_len
        self.dtype = self.inner.dtype
        dev = next(model.parameters()).device
        self.token_buf = torch.zeros(1, dtype=torch.long, device=dev)
        self.t_buf = torch.zeros(1, dtype=torch.long, device=dev)
        self.logits_out = torch.zeros(model.lm_head.out_features, device=dev, dtype=dtype)
        self.arangeT = torch.arange(self.max_len, device=dev)
        self.t = 0

    @torch.no_grad()
    def prefill(self, idx):
        out = self.inner.prefill(idx)
        self.t = self.inner.t
        self.t_buf.fill_(self.t)
        self._capture(self._step_all)
        return out

    def _tree_step_g(self, blk, cache, x):
        mx = blk.mixer
        L, H, hd = cache.L, cache.H, cache.hd
        dev = x.device
        h1 = blk.norm1(x)
        A = mx.gate_hidden(h1)
        cache.B_hist.index_copy_(0, self.t_buf, mx.gate_hidden_src(h1).unsqueeze(0))
        V = mx.linear_in(h1).view(H, hd)
        arangeL = torch.arange(L, device=dev)
        src = self.t_buf - torch.pow(2, arangeL)
        ok = src >= 0
        src_c = src.clamp(min=0)
        Bg = cache.B_hist.index_select(0, src_c) * ok.unsqueeze(-1)
        hid = torch.relu(A.unsqueeze(0) + Bg) ** 2
        okf = ok.unsqueeze(-1)
        i_ = torch.sigmoid(torch.einsum("lg,lhg->lh", hid, mx.gate_out)
                           + mx.gate_out_bias) * okf
        f_ = torch.sigmoid(torch.einsum("lg,lhg->lh", hid, mx.forget_out)
                           + mx.forget_out_bias)
        f_ = f_ * okf + (~okf).to(f_.dtype)
        r_ = torch.sigmoid(torch.einsum("lg,lhg->lh", hid, mx.reset_out)
                           + mx.reset_out_bias)
        cache.rg_hist.index_copy_(0, self.t_buf, r_.unsqueeze(0))
        rg_src = cache.rg_hist.index_select(0, src_c)[arangeL, arangeL]
        V_src = cache.V_lev[arangeL, src_c].view(L, H, hd)
        msg = rg_src.unsqueeze(-1) * V_src
        levs = [V.reshape(-1)]
        for l in range(L):                          # unrolled at capture time
            V = f_[l].unsqueeze(-1) * V + i_[l].unsqueeze(-1) * msg[l]
            levs.append(V.reshape(-1))
        cache.V_lev.index_copy_(1, self.t_buf, torch.stack(levs).unsqueeze(1))
        x = x + mx.linear_out(V.reshape(-1))
        return x + blk.ffn(blk.norm2(x))

    def _attn_step_g(self, blk, cache, x):
        at = blk.attn
        h1 = blk.norm1(x)
        cos = cache.rope_cos.index_select(0, self.t_buf)[0]
        sin = cache.rope_sin.index_select(0, self.t_buf)[0]

        def rope(v):
            v1, v2 = v[..., 0::2], v[..., 1::2]
            return torch.stack([v1 * cos - v2 * sin, v1 * sin + v2 * cos], dim=-1).flatten(-2)

        q = rope(at.linear_q(h1).view(at.num_heads, at.head_dim))
        cache.k.index_copy_(0, self.t_buf,
                            rope(at.linear_k(h1).view(at.num_heads, at.head_dim)).unsqueeze(0))
        cache.v.index_copy_(0, self.t_buf,
                            at.linear_v(h1).view(at.num_heads, at.head_dim).unsqueeze(0))
        valid = self.arangeT <= self.t_buf
        if at.window > 0:
            valid = valid & (self.arangeT > self.t_buf - at.window)
        scores = torch.einsum("hd,thd->ht", q, cache.k) / math.sqrt(at.head_dim)
        scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
        att = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        y = torch.einsum("ht,thd->hd", att, cache.v).reshape(-1)
        x = x + at.linear_out(y)
        return x + blk.ffn(blk.norm2(x))

    def _step_all(self):
        x = self.m.embedding(self.token_buf)[0].to(self.dtype)
        for blk, cache in zip(self.m.blocks, self.inner.caches):
            x = (self._attn_step_g(blk, cache, x) if isinstance(blk, AttentionBlock)
                 else self._tree_step_g(blk, cache, x))
        self.logits_out.copy_(self.m.lm_head(self.m.norm_out(x)))
        self.t_buf.add_(1)


class FusedGraphedHybridDecoder(GraphedHybridDecoder):
    """Graphed decoder whose tree step runs as 2 Triton kernels instead of
    ~13 serial torch ops — the gate readouts in parallel over levels, and the
    level recurrence as one kernel's in-register loop. Same caches, same graph.
    """

    def __init__(self, model, max_len=None, dtype=None):
        from .fused_step import step_gates_kernel, step_scan_kernel   # needs triton
        self._k_gates, self._k_scan = step_gates_kernel, step_scan_kernel
        super().__init__(model, max_len, dtype)
        dev = next(model.parameters()).device
        mix0 = next(b.mixer for b in model.blocks if hasattr(b, "mixer"))
        L, H = mix0.levels, mix0.num_heads
        # scratch shared across layers (contents are per-layer transient)
        self.ig_buf = torch.zeros(L * H, device=dev, dtype=torch.float32)
        self.fg_buf = torch.zeros(L * H, device=dev, dtype=torch.float32)
        self.y_buf = torch.zeros(model.config.d_model, device=dev, dtype=self.dtype)
        self.v0_buf = torch.zeros(model.config.d_model, device=dev, dtype=self.dtype)

    def _tree_step_g(self, blk, cache, x):
        mx = blk.mixer
        L, H, hd = cache.L, cache.H, cache.hd
        GH = mx.gate_hidden.out_features
        h1 = blk.norm1(x)
        A = mx.gate_hidden(h1)
        cache.B_hist.index_copy_(0, self.t_buf, mx.gate_hidden_src(h1).unsqueeze(0))
        self.v0_buf.copy_(mx.linear_in(h1))
        self._k_gates[(L,)](A, cache.B_hist,
                            mx.gate_out, mx.forget_out, mx.reset_out,
                            mx.gate_out_bias, mx.forget_out_bias, mx.reset_out_bias,
                            self.t_buf, self.ig_buf, self.fg_buf, cache.rg_hist,
                            GH=GH, H=H, BK=min(GH, 128))
        self._k_scan[(H,)](self.v0_buf, self.ig_buf, self.fg_buf,
                           cache.rg_hist, cache.V_lev, self.t_buf, self.y_buf,
                           cache.V_lev.shape[1], L=L, HD=hd, H=H)
        x = x + mx.linear_out(self.y_buf)
        return x + blk.ffn(blk.norm2(x))
