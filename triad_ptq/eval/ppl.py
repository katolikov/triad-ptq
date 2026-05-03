"""WikiText-2 / C4 perplexity evaluation for HF causal language models.

Standard implementation: concatenate corpus, slice into non-overlapping
windows of `seq_len`, compute mean NLL per token, return exp(NLL).
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


@torch.no_grad()
def perplexity(
    model,
    tokenizer,
    text: str,
    *,
    device: torch.device,
    seq_len: int = 1024,
    stride: int | None = None,
    progress: bool = True,
    max_tokens: int | None = None,
) -> dict:
    """Sliding-window perplexity. Returns {'ppl', 'nll', 'n_tokens', 'sec'}."""
    model.eval()
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    if max_tokens is not None:
        enc = enc[: max_tokens]
    n = enc.size(0)
    if stride is None:
        stride = seq_len
    nll_sum = 0.0
    tok_count = 0
    t0 = time.perf_counter()
    iterator = range(0, n - 1, stride)
    if progress:
        iterator = tqdm(list(iterator), desc="ppl", leave=False)
    for i in iterator:
        end = min(i + seq_len, n)
        if end - i < 2:
            break
        ids = enc[i:end].unsqueeze(0).to(device)
        out = model(input_ids=ids)
        logits = out.logits if hasattr(out, "logits") else out
        # next-token loss
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        nll_sum += loss.item()
        tok_count += shift_labels.numel()
    elapsed = time.perf_counter() - t0
    nll_mean = nll_sum / max(tok_count, 1)
    return {
        "ppl": math.exp(nll_mean),
        "nll": nll_mean,
        "n_tokens": tok_count,
        "sec": elapsed,
    }


def load_wikitext2(split: str = "test") -> str:
    """Load WikiText-2-raw and concatenate the chosen split."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(ds["text"])


def load_c4_subset(n_docs: int = 256) -> str:
    """Load a small C4 subset for cheap eval."""
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    docs = []
    for i, ex in enumerate(ds):
        if i >= n_docs:
            break
        docs.append(ex["text"])
    return "\n\n".join(docs)
