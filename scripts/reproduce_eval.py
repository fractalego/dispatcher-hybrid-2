"""Reproduce the model card's held-out cross-entropy from the released weights.

    python scripts/reproduce_eval.py --model ./dispatcher-hybrid-292m \
        --val-file val_window_100bt_95500000.jsonl

The validation window is 400 FineWeb-Edu documents from a stream position the
training run never reached. Tokenization is GPT-2 BPE, documents are packed
into 4096-token blocks with an EOT between them, and the model is evaluated in
bf16 autocast over fp32 weights -- the same protocol as the research harness.
"""

import argparse
import json
import math

import tiktoken
import torch
import torch.nn.functional as F

from dispatcher_hybrid import DispatcherHybridForCausalLM


def blocks(docs, tok, block_size, batch_size, device):
    eot = tok.eot_token
    buffer, span = [], batch_size * block_size + 1
    for doc in docs:
        buffer.extend(tok.encode_ordinary(doc["text"]))
        buffer.append(eot)
        while len(buffer) >= span:
            chunk = torch.tensor(buffer[:span], dtype=torch.long)
            buffer = buffer[span - 1:]          # overlap 1 so no target is lost
            yield (chunk[:-1].view(batch_size, block_size).to(device),
                   chunk[1:].view(batch_size, block_size).to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./dispatcher-hybrid-292m")
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DispatcherHybridForCausalLM.from_pretrained(args.model).to(device).eval()
    tok = tiktoken.get_encoding("gpt2")
    docs = [json.loads(line) for line in open(args.val_file)]

    autocast = torch.autocast(device_type=device, dtype=torch.bfloat16,
                              enabled=(device == "cuda"))
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(blocks(docs, tok, model.config.max_len,
                                          args.batch_size, device)):
            if i >= args.max_batches:
                break
            with autocast:
                logits = model(x).logits
            losses.append(F.cross_entropy(logits.view(-1, logits.size(-1)),
                                          y.view(-1)).item())
    ce = sum(losses) / len(losses)
    print(f"{args.model}: held-out CE {ce:.4f} | ppl {math.exp(ce):.2f} "
          f"| {len(losses)} batches of {args.batch_size}x{model.config.max_len}")


if __name__ == "__main__":
    main()
