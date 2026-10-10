"""Configuration for the Dispatcher-Hybrid language model."""

import math

from transformers import PretrainedConfig


class DispatcherConfig(PretrainedConfig):
    """Config for a Dispatcher/attention hybrid causal LM.

    The Dispatcher (arXiv:2105.03994) replaces self-attention with a gated
    dyadic shift-and-sum: at level i every token merges the token 2**i
    positions back, so a span of 2**levels is covered in `levels` parallel
    steps at O(N log N) cost and with no pairwise comparisons. This hybrid
    keeps that tree in most layers and places a few sliding-window attention
    layers among them, which is what buys in-context binding.

    Args:
        vocab_size: GPT-2 BPE by default (50257).
        d_model, num_heads, dim_feedforward, num_layers: trunk shape.
        max_len: context length; must be a power of two (levels = log2(max_len)).
        attn_layers: 1-indexed positions of the sliding-window attention layers.
            Empty tuple = pure tree.
        attn_window: attention window width; 0 disables the attention slots.
        gate_bias_flat / gate_fbias_flat: init biases of the admit/forget gates
            (stored because they shape the module, not just the init).
        doc_reset: block tree edges that cross an `eot_token` boundary. Kept on
            at inference so behaviour matches training.
        tie_word_embeddings: input embedding is the output head (True as trained).
    """

    model_type = "dispatcher-hybrid"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50257,
        d_model=1024,
        num_heads=16,
        dim_feedforward=2816,
        num_layers=18,
        max_len=4096,
        attn_layers=(8, 17),
        attn_window=1024,
        gate_bias_flat=-2.0,
        gate_fbias_flat=2.0,
        doc_reset=True,
        eot_token=50256,
        tie_word_embeddings=True,
        use_cache=False,
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs,
    ):
        if max_len & (max_len - 1):
            raise ValueError(f"max_len must be a power of two, got {max_len}")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.max_len = max_len
        self.attn_layers = tuple(attn_layers)
        self.attn_window = attn_window
        self.gate_bias_flat = gate_bias_flat
        self.gate_fbias_flat = gate_fbias_flat
        self.doc_reset = doc_reset
        self.eot_token = eot_token
        # the tree is recomputed per step rather than cached; see modeling docs
        self.use_cache = use_cache
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @property
    def levels(self):
        return int(math.log2(self.max_len))

    # transformers looks for these generic names in a few places
    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.num_heads

    @property
    def num_hidden_layers(self):
        return self.num_layers

    @property
    def max_position_embeddings(self):
        return self.max_len
