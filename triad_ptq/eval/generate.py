"""Generate sample completions for qualitative comparison."""
from __future__ import annotations

import time

import torch


@torch.no_grad()
def sample_completions(model, tokenizer, prompts: list[str], *,
                        device: torch.device, max_new_tokens: int = 60,
                        do_sample: bool = False) -> list[dict]:
    out = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
        t0 = time.perf_counter()
        gen = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id or 0,
        )
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t0
        text = tokenizer.decode(gen[0, ids.size(1):], skip_special_tokens=True)
        out.append({
            "prompt": p,
            "completion": text,
            "n_new_tokens": int(gen.size(1) - ids.size(1)),
            "sec": elapsed,
            "tok_per_sec": (gen.size(1) - ids.size(1)) / max(elapsed, 1e-6),
        })
    return out


@torch.no_grad()
def measure_decode_throughput(model, tokenizer, *, device: torch.device,
                              n_tokens: int = 100, n_warmup: int = 5,
                              prompt: str = "The capital of France is") -> dict:
    """Measure tokens/sec at batch=1, decode-only, with mps.synchronize()."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    # warmup
    _ = model.generate(ids, max_new_tokens=n_warmup, do_sample=False, pad_token_id=tokenizer.eos_token_id or 0)
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=n_tokens, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id or 0)
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    new_tokens = int(out.size(1) - ids.size(1))
    return {
        "tokens": new_tokens,
        "sec": elapsed,
        "tok_per_sec": new_tokens / max(elapsed, 1e-6),
        "prompt": prompt,
    }
