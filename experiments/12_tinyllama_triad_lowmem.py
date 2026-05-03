"""TinyLlama TRIAD INT4 with low-memory settings (Gram on CPU).

Standalone retry after the default sweep OOM'd. Writes its result back
into results/tables/llm_sweep.json (replaces any prior TRIAD row).
"""
from __future__ import annotations

import gc
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

from triad_ptq.compile import compile_model  # noqa: E402
from triad_ptq.eval.calib import build_wikitext_calib  # noqa: E402
from triad_ptq.eval.generate import sample_completions, measure_decode_throughput  # noqa: E402
from triad_ptq.eval.ppl import load_wikitext2, perplexity  # noqa: E402

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

PROMPTS = [
    "The capital of France is",
    "Photosynthesis is the process by which",
    "In a few sentences, the difference between machine learning and traditional programming is",
    "Once upon a time in a small village,",
    "Quantization in deep learning refers to",
]


def main():
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device={dev}, model={MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    text_eval = load_wikitext2("test")

    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)

    calib = build_wikitext_calib(tok, n_samples=8, seq_len=512, device=dev)
    print(f"calib: {len(calib)} batches of seq_len=512")

    t0 = time.perf_counter()
    compile_model(
        m,
        bits=4,
        calibration=calib,
        super_weight_frac=5e-4,
        bit_allocator="trace",
        cov_grid="analytic",
        n_calib=8,
        rho_probe_n=1,
        group_size=64,
        progress=True,
        device="mps",
    )
    calib_sec = time.perf_counter() - t0
    print(f"TRIAD compile: {calib_sec:.0f}s")

    if dev.type == "mps":
        torch.mps.empty_cache()
    gc.collect()

    res = perplexity(m, tok, text_eval, device=dev,
                     seq_len=1024, max_tokens=16384, progress=False)
    print(f"PPL={res['ppl']:.3f}  on {res['n_tokens']} tokens in {res['sec']:.1f}s")

    try:
        thr = measure_decode_throughput(m, tok, device=dev, n_tokens=64)
        tps = thr["tok_per_sec"]
    except Exception as e:
        print(f"throughput failed: {e}")
        tps = None

    try:
        samples = sample_completions(m, tok, PROMPTS, device=dev, max_new_tokens=40)
    except Exception as e:
        samples = [{"error": str(e)}]

    row = {
        "model": MODEL, "method": "TRIAD", "bits": 4,
        "ppl": res["ppl"], "nll": res["nll"], "n_eval_tokens": res["n_tokens"],
        "calib_sec": calib_sec, "eval_sec": res["sec"],
        "tok_per_sec": tps, "samples": samples,
    }
    out = ROOT / "results" / "tables" / "llm_sweep.json"
    blob = json.loads(out.read_text())
    blob["runs"] = [r for r in blob["runs"]
                    if not (r.get("model") == MODEL and r.get("method") == "TRIAD")]
    blob["runs"].append(row)
    out.write_text(json.dumps(blob, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
