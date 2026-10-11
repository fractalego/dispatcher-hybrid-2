"""Dispatcher-Hybrid: an attention-light language model.

A revival of the Dispatcher (arXiv:2105.03994), where self-attention is
replaced by a gated dyadic shift-and-sum, kept in 16 of 18 layers and paired
with 2 sliding-window attention layers.
"""

from .configuration_dispatcher import DispatcherConfig
from .modeling_dispatcher import DispatcherHybridForCausalLM

__version__ = "0.1.2"
__all__ = ["DispatcherConfig", "DispatcherHybridForCausalLM"]

try:  # make AutoConfig/AutoModelForCausalLM aware of the architecture
    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register("dispatcher-hybrid", DispatcherConfig)
    AutoModelForCausalLM.register(DispatcherConfig, DispatcherHybridForCausalLM)
except Exception:  # pragma: no cover - already registered, or old transformers
    pass
