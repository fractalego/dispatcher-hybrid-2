"""Exact incremental decoding — the architecture's actual serving advantage.

A transformer layer re-reads its whole KV cache for every generated token:
O(t) per layer, per token, so decode slows as context grows. A tree layer
reads one cached vector per level — 12 of them at 4096 context — which is
CONSTANT in t. This hybrid pays the O(t) toll in 2 layers instead of 18.

`DispatcherHybridForCausalLM.generate()` recomputes the whole prefix each
step and realizes none of that. The decoders here do.

Three implementations, same arithmetic, increasing engineering:

    IncrementalDecoder        exact, eager PyTorch
    GraphedHybridDecoder      the whole step captured in one CUDA graph
    FusedGraphedHybridDecoder + 2 Triton kernels for the gate/scan chain

The ladder exists because the per-token work is tiny: at ~13 serial
micro-kernels per layer and a ~3-5us launch floor each, the eager decoder is
launch-bound and loses to a transformer despite doing less arithmetic. Graph
capture removes the launch overhead; the fused kernels collapse the serial
chain into one kernel's internal loop.

Batching: all decoders run B sequences in lockstep — same prompt length, one
step advances every row. That covers sampling N continuations of one prompt
and batches of equal-length prompts; variable-length batches are not
supported (the tree state is position-anchored, so padding would corrupt it).

Caches, per tree layer:
    B_hist  (B, T, gh)      gate source vectors, read at t-2^l by level l
    rg_hist (B, T, L, H)    transmit gates, kept at their SOURCE position
    V_ring  (B, S, d)       per-level scan states in RING buffers: the level-l
                            state is only ever read at lag 2^l, so level l
                            keeps a circular window of 2^(l+1) slots (doubled
                            so the write at t cannot collide with the read at
                            t-2^l, and so the slot index is a bitmask:
                            offsets[l] + (t & masks[l])). S = 2^(L+1)-2 slots
                            against the dense layout's (L+1)*T — 6.5x less.
Attention layers keep a standard windowed KV cache with RoPE at absolute
position.
"""

import math

import torch
import torch.nn.functional as F

from .modeling_dispatcher import AttentionBlock


class _TreeLayerCache:
    def __init__(self, mixer, batch, max_len, device, dtype):
        gh = mixer.gate_hidden.out_features
        self.L = L = mixer.levels
        self.H = mixer.num_heads
        self.hd = mixer.head_dim
        d = self.H * self.hd
        self.B_hist = torch.zeros(batch, max_len, gh, device=device, dtype=dtype)
        self.rg_hist = torch.zeros(batch, max_len, L, self.H, device=device, dtype=dtype)
        # ring j holds level-j states: 2^(j+1) slots at offset 2^(j+1)-2.
        # level L's output is consumed immediately and never cached.
        self.S = 2 ** (L + 1) - 2
        self.V_ring = torch.zeros(batch, self.S, d, device=device, dtype=dtype)
        self.offsets = torch.tensor([2 ** (j + 1) - 2 for j in range(L)],
                                    device=device)
        self.masks = torch.tensor([2 ** (j + 1) - 1 for j in range(L)],
                                  device=device)

    def write_slab(self, j, slab):
        """Fill ring j from a (B, T, d) slab of level-j states (prefill).
        Only the last 2^(j+1) positions can survive; consecutive positions map
        to distinct slots, so later positions simply overwrite earlier ones."""
        T = slab.shape[1]
        size = 2 ** (j + 1)
        p0 = max(0, T - size)
        pos = torch.arange(p0, T, device=slab.device)
        self.V_ring[:, self.offsets[j] + (pos & self.masks[j])] = slab[:, p0:]


class _AttnCache:
    def __init__(self, attn, batch, max_len, device, dtype):
        self.k = torch.zeros(batch, max_len, attn.num_heads, attn.head_dim,
                             device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        # materialized RoPE tables: the model computes these lazily, but graph
        # capture needs stable device tensors to index
        self.rope_cos, self.rope_sin = attn.rope_tables(max_len, device, dtype)


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
        logits = dec.prefill(ids)       # ids (T,) or (B, T); logits match
        logits = dec.step(tokens)       # (B,) or scalar; one lockstep token

    Caches are allocated at prefill for the prompt's batch size.
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
        self.device = next(model.parameters()).device
        self.caches = None
        self.batch = None
        self.t = 0

    def _alloc(self, batch):
        if self.caches is not None and self.batch == batch:
            return
        self.batch = batch
        self.caches = []
        for blk in self.m.blocks:
            if isinstance(blk, AttentionBlock):
                self.caches.append(_AttnCache(blk.attn, batch, self.max_len,
                                              self.device, self.dtype))
            else:
                self.caches.append(_TreeLayerCache(blk.mixer, batch, self.max_len,
                                                   self.device, self.dtype))

    # ---------- tree layer ----------
    def _tree_prefill(self, blk, cache, x):
        # replicate the parallel mixer while stashing B / rg / per-level rings
        mx = blk.mixer
        Bb, T = x.shape[0], x.shape[1]
        h1 = blk.norm1(x)
        A = mx.gate_hidden(h1)                                    # (B, T, gh)
        Bs = mx.gate_hidden_src(h1)
        cache.B_hist[:, :T] = Bs
        L, H, hd = cache.L, cache.H, cache.hd
        n_lv = min(L, int(math.log2(T)) + 1) if T > 1 else 1
        V = mx.linear_in(h1).view(Bb, T, H, hd)
        cache.write_slab(0, V.reshape(Bb, T, H * hd))
        rolled = torch.stack([torch.roll(Bs, shifts=2 ** l, dims=1)
                              for l in range(n_lv)], dim=2)       # (B, T, n_lv, gh)
        wrap = torch.zeros(T, n_lv, 1, device=x.device, dtype=x.dtype)
        for l in range(n_lv):
            wrap[2 ** l:, l] = 1.0
        hidden = torch.relu(A.unsqueeze(2) + rolled * wrap) ** 2
        pre_i = torch.einsum("btlg,lhg->btlh", hidden, mx.gate_out[:n_lv]) + mx.gate_out_bias[:n_lv]
        pre_f = torch.einsum("btlg,lhg->btlh", hidden, mx.forget_out[:n_lv]) + mx.forget_out_bias[:n_lv]
        pre_r = torch.einsum("btlg,lhg->btlh", hidden, mx.reset_out[:n_lv]) + mx.reset_out_bias[:n_lv]
        ig = torch.sigmoid(pre_i) * wrap
        fg = torch.sigmoid(pre_f) * wrap + (1.0 - wrap)
        rg = torch.sigmoid(pre_r)
        cache.rg_hist[:, :T, :n_lv] = rg
        if n_lv < L:
            # levels past the prefill horizon contribute no message yet, but
            # later steps will read these positions' transmit gates
            hid0 = torch.relu(A) ** 2
            for l in range(n_lv, L):
                cache.rg_hist[:, :T, l] = torch.sigmoid(
                    hid0 @ mx.reset_out[l].T + mx.reset_out_bias[l])
        for l in range(n_lv):
            src = rg[:, :, l].unsqueeze(-1) * V                   # transmit at source
            msg = torch.roll(src, shifts=2 ** l, dims=1)
            V = fg[:, :, l].unsqueeze(-1) * V + ig[:, :, l].unsqueeze(-1) * msg
            if l + 1 < L:
                cache.write_slab(l + 1, V.reshape(Bb, T, H * hd))
        # levels beyond the horizon pass the state through unchanged
        for j in range(n_lv + 1, L):
            cache.write_slab(j, V.reshape(Bb, T, H * hd))
        x = x + mx.linear_out(V.reshape(Bb, T, H * hd))
        return x + blk.ffn(blk.norm2(x))

    def _tree_step(self, blk, cache, x, t):
        # x: (B, d) — one lockstep position. Every level's gate math is
        # batched; only the L-step V recurrence stays sequential.
        mx = blk.mixer
        Bb = x.shape[0]
        L, H, hd = cache.L, cache.H, cache.hd
        h1 = blk.norm1(x)
        A = mx.gate_hidden(h1)                                    # (B, gh)
        cache.B_hist[:, t] = mx.gate_hidden_src(h1)
        V = mx.linear_in(h1).view(Bb, H, hd)
        cache.V_ring[:, cache.offsets[0] + (t & int(cache.masks[0]))] = V.reshape(Bb, -1)
        arangeL = torch.arange(L, device=x.device)
        src_t = t - torch.pow(2, arangeL)
        ok = src_t >= 0
        src_c = src_t.clamp(min=0)
        Bg = cache.B_hist[:, src_c] * ok.view(1, L, 1)            # (B, L, gh)
        hid = torch.relu(A.unsqueeze(1) + Bg) ** 2
        okf = ok.view(1, L, 1)
        i_ = torch.sigmoid(torch.einsum("blg,lhg->blh", hid, mx.gate_out)
                           + mx.gate_out_bias) * okf
        f_ = torch.sigmoid(torch.einsum("blg,lhg->blh", hid, mx.forget_out)
                           + mx.forget_out_bias)
        f_ = f_ * okf + (~okf).to(f_.dtype)                       # masked: f = 1
        r_ = torch.sigmoid(torch.einsum("blg,lhg->blh", hid, mx.reset_out)
                           + mx.reset_out_bias)
        cache.rg_hist[:, t] = r_
        rg_src = cache.rg_hist[:, src_c, arangeL]                 # (B, L, H)
        slots = cache.offsets + (src_c & cache.masks)
        V_src = cache.V_ring[:, slots].view(Bb, L, H, hd)
        msg = rg_src.unsqueeze(-1) * V_src
        for l in range(L):
            V = f_[:, l].unsqueeze(-1) * V + i_[:, l].unsqueeze(-1) * msg[:, l]
            if l + 1 < L:
                cache.V_ring[:, cache.offsets[l + 1] + (t & int(cache.masks[l + 1]))] = V.reshape(Bb, -1)
        x = x + mx.linear_out(V.reshape(Bb, -1))
        return x + blk.ffn(blk.norm2(x))

    # ---------- attention layer (windowed, RoPE) ----------
    def _attn_prefill(self, blk, cache, x):
        Bb, T = x.shape[0], x.shape[1]
        at = blk.attn
        h1 = blk.norm1(x)
        shape = (Bb, T, at.num_heads, at.head_dim)
        q = at.linear_q(h1).view(shape).transpose(1, 2)
        k = at.linear_k(h1).view(shape).transpose(1, 2)
        v = at.linear_v(h1).view(shape)
        qb = at.apply_rope(q, T)
        kb = at.apply_rope(k, T)
        cache.k[:, :T] = kb.transpose(1, 2)
        cache.v[:, :T] = v
        vb = v.transpose(1, 2)
        if 0 < at.window < T:
            tt = torch.arange(T, device=x.device)
            band = (tt[:, None] >= tt[None, :]) & (tt[:, None] - tt[None, :] < at.window)
            y = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=band)
        else:
            y = F.scaled_dot_product_attention(qb, kb, vb, is_causal=True)
        x = x + at.linear_out(y.transpose(1, 2).reshape(Bb, T, at.embed_dim))
        return x + blk.ffn(blk.norm2(x))

    def _attn_step(self, blk, cache, x, t):
        at = blk.attn
        Bb = x.shape[0]
        h1 = blk.norm1(x)
        cos, sin = cache.rope_cos[t], cache.rope_sin[t]

        def rope(z):
            z1, z2 = z[..., 0::2], z[..., 1::2]
            return torch.stack([z1 * cos - z2 * sin, z1 * sin + z2 * cos], dim=-1).flatten(-2)

        q = rope(at.linear_q(h1).view(Bb, at.num_heads, at.head_dim))
        cache.k[:, t] = rope(at.linear_k(h1).view(Bb, at.num_heads, at.head_dim))
        cache.v[:, t] = at.linear_v(h1).view(Bb, at.num_heads, at.head_dim)
        lo = max(0, t + 1 - at.window) if at.window > 0 else 0
        k = cache.k[:, lo:t + 1]                                  # (B, w, H, hd)
        v = cache.v[:, lo:t + 1]
        att = torch.softmax(torch.einsum("bhd,bthd->bht", q, k) / math.sqrt(at.head_dim), dim=-1)
        y = torch.einsum("bht,bthd->bhd", att, v).reshape(Bb, -1)
        x = x + at.linear_out(y)
        return x + blk.ffn(blk.norm2(x))

    # ---------- public ----------
    @torch.no_grad()
    def prefill(self, idx):
        """idx: (T,) or (B, T) token ids. Returns (T, vocab) / (B, T, vocab)."""
        self._squeeze = idx.dim() == 1
        ids = idx.view(1, -1) if self._squeeze else idx
        _check_single_document(self.m, ids)
        if ids.shape[1] > self.max_len:
            raise ValueError(f"prompt of {ids.shape[1]} exceeds max_len {self.max_len}")
        self._alloc(ids.shape[0])
        x = self.m.embedding(ids).to(self.dtype)
        for blk, cache in zip(self.m.blocks, self.caches):
            x = (self._attn_prefill(blk, cache, x) if isinstance(blk, AttentionBlock)
                 else self._tree_prefill(blk, cache, x))
        self.t = ids.shape[1]
        out = self.m.lm_head(self.m.norm_out(x))
        return out[0] if self._squeeze else out

    @torch.no_grad()
    def step(self, tokens):
        """tokens: scalar / (1,) / (B,). Returns (vocab,) or (B, vocab)."""
        if self.t >= self.max_len:
            raise ValueError(f"context exhausted at max_len {self.max_len}")
        t = self.t
        toks = tokens.view(-1)
        if toks.shape[0] != self.batch:
            if self.batch == 1 and toks.shape[0] == 1:
                pass
            else:
                raise ValueError(f"got {toks.shape[0]} tokens for batch {self.batch}")
        x = self.m.embedding(toks).to(self.dtype)
        for blk, cache in zip(self.m.blocks, self.caches):
            x = (self._attn_step(blk, cache, x, t) if isinstance(blk, AttentionBlock)
                 else self._tree_step(blk, cache, x, t))
        self.t = t + 1
        out = self.m.lm_head(self.m.norm_out(x))
        return out[0] if self._squeeze else out


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
    def step(self, tokens):
        self.token_buf.copy_(tokens.view(-1))
        self.graph.replay()
        self.t += 1
        # the graph writes into a fixed output buffer that the next replay
        # overwrites; return a copy so callers can keep the result
        out = self.logits_out.clone()
        return out[0] if self._squeeze else out


class GraphedHybridDecoder(_GraphedBase):
    """Eager prefill, then one CUDA-graph replay per generated token."""

    def __init__(self, model, max_len=None, dtype=None):
        if not torch.cuda.is_available():
            raise RuntimeError("GraphedHybridDecoder requires CUDA")
        self.inner = IncrementalDecoder(model, max_len, dtype=dtype)
        self.m = model
        self.max_len = self.inner.max_len
        self.dtype = self.inner.dtype
        self.t = 0

    @torch.no_grad()
    def prefill(self, idx):
        out = self.inner.prefill(idx)
        self._squeeze = self.inner._squeeze
        self.t = self.inner.t
        dev = next(self.m.parameters()).device
        Bb = self.inner.batch
        self.token_buf = torch.zeros(Bb, dtype=torch.long, device=dev)
        self.t_buf = torch.zeros(1, dtype=torch.long, device=dev)
        self.t_buf.fill_(self.t)
        self.logits_out = torch.zeros(Bb, self.m.lm_head.out_features,
                                      device=dev, dtype=self.dtype)
        self.arangeT = torch.arange(self.max_len, device=dev)
        self._prepare()
        self._capture(self._step_all)
        return out

    def _prepare(self):
        pass

    def _tree_step_g(self, blk, cache, x):
        mx = blk.mixer
        Bb = x.shape[0]
        L, H, hd = cache.L, cache.H, cache.hd
        dev = x.device
        h1 = blk.norm1(x)
        A = mx.gate_hidden(h1)
        cache.B_hist.index_copy_(1, self.t_buf, mx.gate_hidden_src(h1).unsqueeze(1))
        V = mx.linear_in(h1).view(Bb, H, hd)
        arangeL = torch.arange(L, device=dev)
        src = self.t_buf - torch.pow(2, arangeL)
        ok = src >= 0
        src_c = src.clamp(min=0)
        Bg = cache.B_hist.index_select(1, src_c) * ok.view(1, L, 1)
        hid = torch.relu(A.unsqueeze(1) + Bg) ** 2
        okf = ok.view(1, L, 1)
        i_ = torch.sigmoid(torch.einsum("blg,lhg->blh", hid, mx.gate_out)
                           + mx.gate_out_bias) * okf
        f_ = torch.sigmoid(torch.einsum("blg,lhg->blh", hid, mx.forget_out)
                           + mx.forget_out_bias)
        f_ = f_ * okf + (~okf).to(f_.dtype)
        r_ = torch.sigmoid(torch.einsum("blg,lhg->blh", hid, mx.reset_out)
                           + mx.reset_out_bias)
        cache.rg_hist.index_copy_(1, self.t_buf, r_.unsqueeze(1))
        rg_src = cache.rg_hist.index_select(1, src_c)[:, arangeL, arangeL]
        slots_r = cache.offsets + (src_c & cache.masks)
        V_src = cache.V_ring.index_select(1, slots_r).view(Bb, L, H, hd)
        msg = rg_src.unsqueeze(-1) * V_src
        levs = [V.reshape(Bb, -1)]
        for l in range(L):                          # unrolled at capture time
            V = f_[:, l].unsqueeze(-1) * V + i_[:, l].unsqueeze(-1) * msg[:, l]
            if l + 1 < L:
                levs.append(V.reshape(Bb, -1))
        slots_w = cache.offsets + (self.t_buf & cache.masks)
        cache.V_ring.index_copy_(1, slots_w, torch.stack(levs, dim=1))
        x = x + mx.linear_out(V.reshape(Bb, -1))
        return x + blk.ffn(blk.norm2(x))

    def _attn_step_g(self, blk, cache, x):
        at = blk.attn
        Bb = x.shape[0]
        h1 = blk.norm1(x)
        cos = cache.rope_cos.index_select(0, self.t_buf)[0]
        sin = cache.rope_sin.index_select(0, self.t_buf)[0]

        def rope(z):
            z1, z2 = z[..., 0::2], z[..., 1::2]
            return torch.stack([z1 * cos - z2 * sin, z1 * sin + z2 * cos], dim=-1).flatten(-2)

        q = rope(at.linear_q(h1).view(Bb, at.num_heads, at.head_dim))
        cache.k.index_copy_(1, self.t_buf,
                            rope(at.linear_k(h1).view(Bb, at.num_heads, at.head_dim)).unsqueeze(1))
        cache.v.index_copy_(1, self.t_buf,
                            at.linear_v(h1).view(Bb, at.num_heads, at.head_dim).unsqueeze(1))
        valid = self.arangeT <= self.t_buf
        if at.window > 0:
            valid = valid & (self.arangeT > self.t_buf - at.window)
        scores = torch.einsum("bhd,bthd->bht", q, cache.k) / math.sqrt(at.head_dim)
        scores = scores.masked_fill(~valid.view(1, 1, -1), float("-inf"))
        att = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        y = torch.einsum("bht,bthd->bhd", att, cache.v).reshape(Bb, -1)
        x = x + at.linear_out(y)
        return x + blk.ffn(blk.norm2(x))

    def _step_all(self):
        x = self.m.embedding(self.token_buf).to(self.dtype)
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

    def _prepare(self):
        dev = next(self.m.parameters()).device
        mix0 = next(b.mixer for b in self.m.blocks if hasattr(b, "mixer"))
        L, H = mix0.levels, mix0.num_heads
        Bb = self.inner.batch
        # scratch shared across layers (contents are per-layer transient)
        self.ig_buf = torch.zeros(Bb, L * H, device=dev, dtype=torch.float32)
        self.fg_buf = torch.zeros(Bb, L * H, device=dev, dtype=torch.float32)
        self.y_buf = torch.zeros(Bb, self.m.config.d_model, device=dev, dtype=self.dtype)
        self.v0_buf = torch.zeros(Bb, self.m.config.d_model, device=dev, dtype=self.dtype)

    def _tree_step_g(self, blk, cache, x):
        mx = blk.mixer
        Bb = x.shape[0]
        L, H, hd = cache.L, cache.H, cache.hd
        GH = mx.gate_hidden.out_features
        h1 = blk.norm1(x)
        A = mx.gate_hidden(h1)
        cache.B_hist.index_copy_(1, self.t_buf, mx.gate_hidden_src(h1).unsqueeze(1))
        self.v0_buf.copy_(mx.linear_in(h1))
        self._k_gates[(Bb, L)](A, cache.B_hist,
                               mx.gate_out, mx.forget_out, mx.reset_out,
                               mx.gate_out_bias, mx.forget_out_bias, mx.reset_out_bias,
                               self.t_buf, self.ig_buf, self.fg_buf, cache.rg_hist,
                               self.max_len,
                               GH=GH, H=H, L=L, BK=min(GH, 128))
        self._k_scan[(Bb, H)](self.v0_buf, self.ig_buf, self.fg_buf,
                              cache.rg_hist, cache.V_ring, self.t_buf, self.y_buf,
                              self.max_len, cache.S,
                              L=L, HD=hd, H=H)
        x = x + mx.linear_out(self.y_buf)
        return x + blk.ffn(blk.norm2(x))
