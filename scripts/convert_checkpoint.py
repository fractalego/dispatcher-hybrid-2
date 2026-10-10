"""Convert a research training checkpoint into a HuggingFace-format model.

    python scripts/convert_checkpoint.py \
        --checkpoint /path/to/checkpoints_full_95b/final.pt \
        --out ./dispatcher-hybrid-292m

Three things happen here beyond a rename:
  * optimizer/RNG/data-stream state is dropped (3.3GB -> ~1.2GB);
  * the tied embedding, stored twice in the training checkpoint, is stored
    once (safetensors refuses shared storage, and it saves ~200MB);
  * the FFN weights move under a `ffn.` submodule prefix, the only structural
    difference from the research module tree.
"""

import argparse
import json

import torch

from dispatcher_hybrid import DispatcherConfig, DispatcherHybridForCausalLM

FFN_KEYS = ("linear_gate", "linear_up", "linear_down")


def remap(state_dict):
    """Research checkpoint key -> package key. Returns (mapped, dropped)."""
    mapped, dropped = {}, []
    for k, v in state_dict.items():
        if k == "mask":  # regenerated deterministically from max_len
            dropped.append(k)
            continue
        parts = k.split(".")
        # blocks.<i>.<ffn weight>.weight  ->  blocks.<i>.ffn.<ffn weight>.weight
        if len(parts) >= 3 and parts[0] == "blocks" and parts[2] in FFN_KEYS:
            parts = parts[:2] + ["ffn"] + parts[2:]
        mapped[".".join(parts)] = v
    return mapped, dropped


def config_from_args(args_dict, vocab_size):
    attn_layers = args_dict.get("attn_layers") or ""
    if isinstance(attn_layers, str):
        attn_layers = tuple(int(z) for z in attn_layers.split(",") if z)
    return DispatcherConfig(
        vocab_size=vocab_size,
        d_model=args_dict["d_model"],
        num_heads=args_dict["num_heads"],
        dim_feedforward=args_dict["dim_feedforward"],
        num_layers=args_dict["num_layers"],
        max_len=args_dict["bptt"],
        attn_layers=attn_layers,
        attn_window=args_dict.get("attn_window") or 0,
        gate_bias_flat=args_dict.get("gate_bias_flat", 0.0),
        gate_fbias_flat=args_dict.get("gate_fbias_flat", 0.0),
        doc_reset=bool(args_dict.get("doc_reset")),
        eot_token=args_dict.get("eot_token") or 50256,
        tie_word_embeddings=not args_dict.get("untie", False),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--no-remote-code", action="store_true",
                    help="skip bundling the modeling code (and auto_map) into the output")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = ckpt["args"]
    unsupported = {k: train_args.get(k) for k in
                   ("recency", "retrieval", "delta", "gate_headmod", "gate_moe",
                    "gate_vector", "gate_norm", "gate_exp", "gate_msgtx",
                    "gate_conv", "gate_rglru", "gate_dparam", "gate_srclora",
                    "gate_topk", "gate_headmod_rank", "recency_peek")
                   if train_args.get(k)}
    if unsupported:
        raise SystemExit(f"this checkpoint uses research-only features that the "
                         f"release package does not implement: {unsupported}")
    for key, expected in (("gate_update", "lstm"), ("gate_act", "relu2")):
        if train_args.get(key) != expected:
            raise SystemExit(f"release package implements {key}={expected!r}, "
                             f"checkpoint has {train_args.get(key)!r}")
    if not train_args.get("gate_reset"):
        raise SystemExit("release package implements the transmit gate "
                         "(gate_reset=True); checkpoint has it off")

    sd = ckpt["model_state_dict"]
    vocab_size = sd["embedding.weight"].shape[0]
    config = config_from_args(train_args, vocab_size)
    mapped, dropped = remap(sd)

    model = DispatcherHybridForCausalLM(config)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]  # tied
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch\n  missing: {missing}\n"
                         f"  unexpected: {unexpected}")
    model = model.to(getattr(torch, args.dtype)).eval()

    if not args.no_remote_code:
        # bundle the defining modules into the model directory and set auto_map,
        # so the Hub copy loads with trust_remote_code=True and no pip install
        DispatcherConfig.register_for_auto_class()
        DispatcherHybridForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    model.save_pretrained(args.out, safe_serialization=True)
    meta = {
        "source_checkpoint": args.checkpoint,
        "source_step": ckpt.get("step"),
        "tokens_seen": ckpt.get("tokens_seen"),
        "best_val_loss": ckpt.get("best_val"),
        "dropped_keys": dropped,
        "dtype": args.dtype,
    }
    with open(f"{args.out}/conversion_info.json", "w") as f:
        json.dump(meta, f, indent=2)

    unique = sum(p.numel() for p in model.parameters())
    print(f"wrote {args.out}")
    print(f"  step {meta['source_step']} | {meta['tokens_seen'] / 1e9:.2f}B tokens "
          f"| val {meta['best_val_loss']:.4f}")
    print(f"  {unique / 1e6:.1f}M unique params, dtype {args.dtype}, dropped {dropped}")


if __name__ == "__main__":
    main()
