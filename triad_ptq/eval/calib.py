"""Calibration data builders for HF causal LMs."""
from __future__ import annotations

from typing import Iterable

import torch


def build_text_calib(
    tokenizer,
    text_chunks: list[str],
    *,
    seq_len: int = 1024,
    n_samples: int = 128,
    seed: int = 0,
    device: torch.device | None = None,
) -> list[dict]:
    """Sample non-overlapping windows of length seq_len from concatenated text."""
    if isinstance(text_chunks, str):
        text_chunks = [text_chunks]
    big = "\n\n".join(text_chunks)
    enc = tokenizer(big, return_tensors="pt").input_ids[0]
    n = enc.size(0)
    if n < seq_len * n_samples:
        n_samples = max(n // seq_len, 1)
    g = torch.Generator().manual_seed(seed)
    starts = torch.randperm(max(n - seq_len - 1, 1), generator=g)[:n_samples]
    batches = []
    for s in starts.tolist():
        ids = enc[s : s + seq_len].unsqueeze(0)
        if device is not None:
            ids = ids.to(device)
        batches.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    return batches


def build_wikitext_calib(tokenizer, *, n_samples=128, seq_len=1024, seed=0, device=None):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if len(t.strip()) > 0)
    return build_text_calib(
        tokenizer, [text],
        seq_len=seq_len, n_samples=n_samples, seed=seed, device=device,
    )
