"""Dispatcher-Hybrid: a dyadic shift-and-sum language model with a few
sliding-window attention layers.

This is the inference-ready extraction of the research implementation: the
same algebra, same parameter names, with the ~20 experimental gate variants
of the research code removed. Weights converted from a research checkpoint
load here 1:1 and produce bitwise-comparable logits (see tests/test_parity.py).

The mixer, per layer, per level i = 0 .. log2(max_len)-1:

    h_i(t)  = relu( W1x.x(t) + W1s.x(t - 2**i) )**2        # gate hidden
    a_i(t)  = sigmoid( Wa_i . h_i(t) + b_a )               # admit
    f_i(t)  = sigmoid( Wf_i . h_i(t) + b_f )               # forget
    r_i(t)  = sigmoid( Wr_i . h_i(t) + b_r )               # transmit (at source)
    V(t)   <- f_i(t) * V(t) + a_i(t) * (r_i * V)(t - 2**i)

All of it is comparison-free (no q.k anywhere) and runs as `levels` batched
steps, so cost is O(N log N) with no N x N matrix. Causality rests on two
things: the shift direction, and a mask that zeroes level i for t < 2**i so
no wrapped (future) token is ever read. Both are covered by the causality
test in tests/.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

from .configuration_dispatcher import DispatcherConfig


class DispatcherMixer(nn.Module):
    """Gated dyadic shift-and-sum — the drop-in replacement for attention."""

    def __init__(self, config: DispatcherConfig):
        super().__init__()
        d_model, num_heads = config.d_model, config.num_heads
        self.embed_dim = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.levels = config.levels
        self.shifts = [2 ** i for i in range(self.levels)]

        self.use_fused_gates = False   # set via model.enable_fused_gates()
        gh = d_model  # gate hidden width (gate_hidden_dim=None in the release config)
        self.linear_in = nn.Linear(d_model, d_model, bias=False)
        self.gate_hidden = nn.Linear(d_model, gh, bias=False)       # W1x, destination
        self.gate_hidden_src = nn.Linear(d_model, gh, bias=False)   # W1s, source
        self.gate_out = nn.Parameter(torch.empty(self.levels, num_heads, gh))
        self.gate_out_bias = nn.Parameter(
            torch.full((self.levels, num_heads), float(config.gate_bias_flat)))
        self.forget_out = nn.Parameter(torch.empty(self.levels, num_heads, gh))
        self.forget_out_bias = nn.Parameter(
            torch.full((self.levels, num_heads), float(config.gate_fbias_flat)))
        self.reset_out = nn.Parameter(torch.empty(self.levels, num_heads, gh))
        self.reset_out_bias = nn.Parameter(torch.full((self.levels, num_heads), 2.0))
        self.linear_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask):
        # x: (B, T, d_model); mask: (T, levels) or (B, T, levels) when doc_reset
        batch, length = x.shape[0], x.shape[1]
        heads, hd = self.num_heads, self.head_dim

        # fold heads into the batch axis — the transpose here is load-bearing:
        # a plain reshape would make heads into sequence blocks (the 2021 bug).
        V = self.linear_in(x).view(batch, length, heads, hd).transpose(1, 2)
        V = V.reshape(batch * heads, length, hd)

        A = self.gate_hidden(x)        # (B, T, gh)
        B = self.gate_hidden_src(x)    # (B, T, gh)

        n_lv = min(self.levels, int(math.log2(length)) + 1)
        if mask.dim() == 3:
            m4 = mask[..., :n_lv].unsqueeze(-1)                      # (B, T, n_lv, 1)
            mbh = (mask[..., :n_lv].unsqueeze(1)
                   .expand(batch, heads, length, n_lv)
                   .reshape(batch * heads, length, n_lv))
        else:
            m4 = mask[:, :n_lv].view(1, length, n_lv, 1)
            mbh = mask[:, :n_lv]

        def heads_first(pre):
            return torch.sigmoid(pre).permute(0, 3, 1, 2).reshape(
                batch * heads, length, n_lv).to(V.dtype)

        if self.use_fused_gates and x.is_cuda:
            # Triton path: the (B, T, levels, gh) hidden tensor is the mixer's
            # cost center (~400MB/layer at 4096 context); these kernels build it
            # a tile at a time in SRAM and contract immediately, so it never
            # exists. Masks are applied in-kernel.
            from .fused_gates import fused_gate_preacts
            pre_i, pre_f, pre_r = fused_gate_preacts(
                A, B, self.gate_out, self.forget_out, self.reset_out,
                self.gate_out_bias, self.forget_out_bias, self.reset_out_bias,
                n_lv, mask[..., :n_lv] if mask.dim() == 3 else None)
        else:
            # source features per level, with wrapped (future) rows zeroed
            rolled = torch.stack([torch.roll(B, shifts=s, dims=1)
                                  for s in self.shifts[:n_lv]], dim=2) * m4
            hidden = F.relu(A.unsqueeze(2) + rolled) ** 2            # (B, T, n_lv, gh)
            pre_i, pre_f, pre_r = (
                torch.einsum("btld,lhd->btlh", hidden, p[:n_lv]) + b[:n_lv]
                for p, b in ((self.gate_out, self.gate_out_bias),
                             (self.forget_out, self.forget_out_bias),
                             (self.reset_out, self.reset_out_bias)))

        admit = heads_first(pre_i) * mbh
        # masked levels receive no message, so they must not forget: f -> 1
        forget = heads_first(pre_f) * mbh + (1.0 - mbh)
        transmit = heads_first(pre_r)

        for li, shift in enumerate(self.shifts[:n_lv]):
            if shift > length:
                break
            msg = torch.roll(transmit[..., li:li + 1] * V, shifts=shift, dims=1)
            V = forget[..., li:li + 1] * V + admit[..., li:li + 1] * msg

        V = V.view(batch, heads, length, hd).transpose(1, 2).reshape(batch, length, self.embed_dim)
        return self.linear_out(V)


class WindowAttention(nn.Module):
    """Causal sliding-window multi-head attention with RoPE."""

    def __init__(self, config: DispatcherConfig):
        super().__init__()
        d_model, num_heads = config.d_model, config.num_heads
        self.embed_dim = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window = config.attn_window

        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)

        self._rope_key = None

    def rope_tables(self, length, device, dtype):
        """RoPE cos/sin, computed on demand and cached.

        Deliberately NOT a registered buffer: a non-persistent buffer is absent
        from the checkpoint and `from_pretrained` would leave it unmaterialized
        (silently garbage), while a persistent one would force every checkpoint
        to carry a derived table.
        """
        key = (length, device, dtype)
        if self._rope_key != key:
            theta = 10000.0 ** (
                -torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
                / self.head_dim)
            angles = (torch.arange(length, device=device, dtype=torch.float32)[:, None]
                      * theta[None, :])
            self._rope = (angles.cos().to(dtype), angles.sin().to(dtype))
            self._rope_key = key
        return self._rope

    def apply_rope(self, x, length):
        cos, sin = self.rope_tables(length, x.device, x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

    def forward(self, x):
        batch, length = x.shape[0], x.shape[1]
        shape = (batch, length, self.num_heads, self.head_dim)
        q = self.linear_q(x).view(shape).transpose(1, 2)
        k = self.linear_k(x).view(shape).transpose(1, 2)
        v = self.linear_v(x).view(shape).transpose(1, 2)
        q = self.apply_rope(q, length)
        k = self.apply_rope(k, length)

        if 0 < self.window < length:
            t = torch.arange(length, device=x.device)
            band = (t[:, None] >= t[None, :]) & (t[:, None] - t[None, :] < self.window)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=band)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).reshape(batch, length, self.embed_dim)
        return self.linear_out(y)


class SwiGLU(nn.Module):
    def __init__(self, config: DispatcherConfig):
        super().__init__()
        d, ff = config.d_model, config.dim_feedforward
        self.linear_gate = nn.Linear(d, ff, bias=False)
        self.linear_up = nn.Linear(d, ff, bias=False)
        self.linear_down = nn.Linear(ff, d, bias=False)

    def forward(self, h):
        return self.linear_down(F.silu(self.linear_gate(h)) * self.linear_up(h))


class DispatcherBlock(nn.Module):
    """Pre-norm block with the tree mixer."""

    def __init__(self, config: DispatcherConfig):
        super().__init__()
        self.mixer = DispatcherMixer(config)
        self.ffn = SwiGLU(config)
        self.norm1 = nn.RMSNorm(config.d_model)
        self.norm2 = nn.RMSNorm(config.d_model)

    def forward(self, x, mask):
        x = x + self.mixer(self.norm1(x), mask)
        return x + self.ffn(self.norm2(x))


class AttentionBlock(nn.Module):
    """Pre-norm block with sliding-window attention; ignores the tree mask."""

    def __init__(self, config: DispatcherConfig):
        super().__init__()
        self.attn = WindowAttention(config)
        self.ffn = SwiGLU(config)
        self.norm1 = nn.RMSNorm(config.d_model)
        self.norm2 = nn.RMSNorm(config.d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class DispatcherPreTrainedModel(PreTrainedModel):
    config_class = DispatcherConfig
    base_model_prefix = "dispatcher"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DispatcherBlock", "AttentionBlock"]

    def _init_weights(self, module):
        # W1s (gate_hidden_src) is zero-init by design: at step 0 the gates see
        # only the destination token, and |W1s| growth measures the model's
        # demand for source matching. Irrelevant when loading trained weights.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, DispatcherMixer):
            for p in (module.gate_out, module.forget_out, module.reset_out):
                nn.init.normal_(p, mean=0.0, std=0.02)
            module.gate_hidden_src.weight.data.zero_()


class DispatcherHybridForCausalLM(DispatcherPreTrainedModel, GenerationMixin):
    """Dispatcher/attention hybrid causal language model.

    >>> from transformers import AutoTokenizer
    >>> from dispatcher_hybrid import DispatcherHybridForCausalLM
    >>> model = DispatcherHybridForCausalLM.from_pretrained("fractalego/dispatcher-hybrid-292m")
    >>> tok = AutoTokenizer.from_pretrained("gpt2")
    >>> ids = tok("The dispatcher merges", return_tensors="pt").input_ids
    >>> tok.decode(model.generate(ids, max_new_tokens=20, do_sample=False)[0])

    Note on generation: the tree state is recomputed for the whole prefix at
    every step (`use_cache=False`), so `generate` is correct but O(T) forwards.
    An exact incremental decoder is the natural next addition; see README.
    """

    _tied_weights_keys = {"lm_head.weight": "embedding.weight"}
    main_input_name = "input_ids"

    def __init__(self, config: DispatcherConfig):
        super().__init__(config)
        attn_slots = set(config.attn_layers) if config.attn_window > 0 else set()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            AttentionBlock(config) if (i + 1) in attn_slots else DispatcherBlock(config)
            for i in range(config.num_layers)
        )
        self.norm_out = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._mask_key = None
        self.post_init()

    def level_mask(self, length, device=None, dtype=torch.float32):
        """mask[t, i] = 1 iff token t has a predecessor 2**i positions back.

        Zeroing level i for t < 2**i is what stops a wrapped (future) token
        from being read; see tests/test_parity.py::test_causality. Computed on
        demand for the same reason as the RoPE tables.
        """
        key = (length, device, dtype)
        if self._mask_key != key:
            mask = torch.zeros(length, self.config.levels, device=device, dtype=dtype)
            for i in range(self.config.levels):
                if 2 ** i < length:
                    mask[2 ** i:, i] = 1.0
            self._mask_cache = mask
            self._mask_key = key
        return self._mask_cache

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def _tree_mask(self, input_ids, length):
        mask = self.level_mask(length, input_ids.device, self.embedding.weight.dtype)
        if not self.config.doc_reset:
            return mask
        # doc_id[t] = exclusive prefix count of EOTs; a level-i edge is allowed
        # only when source and destination sit in the same document.
        is_eot = (input_ids == self.config.eot_token).to(mask.dtype)
        doc_id = torch.cumsum(is_eot, dim=1) - is_eot
        allowed = torch.stack(
            [(doc_id == torch.roll(doc_id, 2 ** i, dims=1)) for i in range(self.config.levels)],
            dim=-1,
        ).to(mask.dtype)
        return mask.unsqueeze(0) * allowed[..., :mask.shape[-1]]

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                inputs_embeds=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, **kwargs):
        if input_ids is None:
            raise ValueError("DispatcherHybridForCausalLM requires input_ids "
                             "(the tree mask is built from token identities)")
        return_dict = True if return_dict is None else return_dict
        length = input_ids.size(1)
        if length > self.config.max_len:
            raise ValueError(f"sequence length {length} exceeds max_len "
                             f"{self.config.max_len}")

        mask = self._tree_mask(input_ids, length)
        hidden_states = () if output_hidden_states else None

        x = self.embedding(input_ids)
        for block in self.blocks:
            if output_hidden_states:
                hidden_states = hidden_states + (x,)
            if getattr(self, "gradient_checkpointing", False) and self.training:
                x = self._gradient_checkpointing_func(block.__call__, x, mask)
            else:
                x = block(x, mask)
        x = self.norm_out(x)
        if output_hidden_states:
            hidden_states = hidden_states + (x,)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                   labels[:, 1:].reshape(-1))
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=hidden_states)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # no cache: always feed the (cropped) prefix
        return {"input_ids": input_ids[:, -self.config.max_len:]}

    def enable_fused_gates(self, enabled=True):
        """Use the Triton gate kernels in the parallel forward (CUDA only).

        ~1.4x faster with ~34% less activation memory at full shapes, at
        bf16 rounding in the gate pre-activations. Needs `triton` installed
        (`pip install dispatcher-hybrid-2[fast]`).
        """
        if enabled:
            from . import fused_gates  # noqa: F401 — fail now, not mid-forward
        for blk in self.blocks:
            if hasattr(blk, "mixer"):
                blk.mixer.use_fused_gates = enabled
        return self

    @torch.no_grad()
    def fast_generate(self, input_ids, max_new_tokens=50, do_sample=False,
                      temperature=1.0, top_k=None, eos_token_id=None,
                      num_return_sequences=1, decoder="auto"):
        """Generate with an exact incremental decoder instead of recomputing.

        `generate()` re-runs the whole prefix for every token; this keeps the
        per-level caches, so per-token cost is O(log t) and flat in context.

        input_ids: (T,) or (B, T) — rows run in lockstep, so a 2-D batch must
        share one prompt length (no padding). num_return_sequences tiles a
        single prompt into a batch of samples. `decoder` is "auto" (best
        available), "eager", "graphed" (CUDA graph) or "fused" (graph +
        Triton kernels). Rows that hit eos are frozen (padded with eos) while
        the rest continue.
        """
        from .decoding import (IncrementalDecoder, GraphedHybridDecoder,
                               FusedGraphedHybridDecoder)
        ids = input_ids.view(1, -1) if input_ids.dim() == 1 else input_ids
        if num_return_sequences > 1:
            if ids.shape[0] != 1:
                raise ValueError("num_return_sequences needs a single prompt")
            if not do_sample:
                raise ValueError("num_return_sequences without sampling would "
                                 "return identical rows")
            ids = ids.expand(num_return_sequences, -1)

        choices = {"eager": IncrementalDecoder, "graphed": GraphedHybridDecoder,
                   "fused": FusedGraphedHybridDecoder}
        if decoder == "auto":
            order = ["fused", "graphed", "eager"] if ids.is_cuda else ["eager"]
            dec = None
            for name in order:
                try:
                    dec = choices[name](self)
                    break
                except Exception:
                    continue
            if dec is None:
                raise RuntimeError("no usable decoder")
        else:
            dec = choices[decoder](self)

        Bb = ids.shape[0]
        logits = dec.prefill(ids)[:, -1]                       # (B, vocab)
        finished = torch.zeros(Bb, dtype=torch.bool, device=ids.device)
        out = [ids]
        for _ in range(max_new_tokens):
            logits = logits.float() / max(temperature, 1e-6)
            if top_k is not None:
                kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = (torch.multinomial(F.softmax(logits, dim=-1), 1)[:, 0] if do_sample
                   else logits.argmax(-1))
            if eos_token_id is not None:
                nxt = torch.where(finished, torch.full_like(nxt, eos_token_id), nxt)
                finished = finished | (nxt == eos_token_id)
            out.append(nxt.unsqueeze(1))
            if eos_token_id is not None and bool(finished.all()):
                break
            if dec.t >= dec.max_len:
                break
            logits = dec.step(nxt)
        return torch.cat(out, dim=1)
