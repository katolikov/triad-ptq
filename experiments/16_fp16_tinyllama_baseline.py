"""TinyLlama FP16 baseline PPL on M1 — anchor for the +1.0 PPL acceptance.

Output: results/fp16_tinyllama_m1.json with WikiText-2 PPL on the same
window used by experiments/13_tinyllama_phase3.py (seq=512,
max_tokens=4096) so the two numbers are comparable.

Runs on CPU because the same hardware that calibrated the TRIAD model
is also the one we read the FP16 number from, and CPU is the only
backend safe under the post-calib swap pressure (see STATUS.md). On M1
this takes ~5-10 min.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from triad_ptq.eval.ppl import load_wikitext2, perplexity  # noqa: E402

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def main():
    dev = torch.device("cpu")
    print(f"FP16 baseline (on {dev}) for {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    text_eval = load_wikitext2("test")

    # FP16 weights, fp16 forward; on CPU, fp16 is slow but PPL is what we need.
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)

    t0 = time.perf_counter()
    res = perplexity(
        m, tok, text_eval, device=dev,
        seq_len=512, max_tokens=4096, progress=False,
    )
    print(
        f"FP16 PPL={res['ppl']:.3f} on {res['n_tokens']} tokens "
        f"in {res['sec']:.1f}s",
        flush=True,
    )

    out = ROOT / "results" / "fp16_tinyllama_m1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": MODEL,
        "method": "FP16",
        "bits": 16,
        "device": "cpu",
        "ppl_wikitext2": float(res["ppl"]),
        "ppl_n_tokens": int(res["n_tokens"]),
        "ppl_sec": float(res["sec"]),
        "ppl_seq_len": 512,
        "ppl_max_tokens": 4096,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
