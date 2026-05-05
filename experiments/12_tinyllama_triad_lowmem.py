"""TinyLlama TRIAD INT4 with low-memory settings (Gram on CPU).

Streaming compile_model from feat/exynos-cholesky-fix keeps only one
layer's heavy state (A, U, W_prime, H_prime) on the compute device at a
time. Per-layer Gram matrices live on CPU via `a_device='cpu'`; the
active layer's A is moved to MPS only for its eigh+GPTQ window then
freed. Together with the dict-of-everything cleanup in compile.py this
brings TinyLlama-1.1B INT4 calibration under M1 Pro 8 GB.

Writes its result back into results/tables/llm_sweep.json (replaces any
prior TRIAD row) and into results/triad_tinyllama_int4_m1.json (the
Phase-3 acceptance file referenced by docs/decisions/...).
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
    pre_alloc = (
        torch.mps.current_allocated_memory() / 1e9 if dev.type == "mps" else 0.0
    )
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
        a_device="cpu",       # Phase 2: stream Gram matrices via host RAM.
    )
    calib_sec = time.perf_counter() - t0
    post_alloc = (
        torch.mps.current_allocated_memory() / 1e9 if dev.type == "mps" else 0.0
    )
    peak_alloc = (
        torch.mps.driver_allocated_memory() / 1e9
        if dev.type == "mps" and hasattr(torch.mps, "driver_allocated_memory") else 0.0
    )
    print(
        f"TRIAD compile: {calib_sec:.0f}s | "
        f"MPS pre={pre_alloc:.2f} GB post={post_alloc:.2f} GB peak~={peak_alloc:.2f} GB"
    )

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
    if out.exists():
        blob = json.loads(out.read_text())
        blob.setdefault("runs", [])
        blob["runs"] = [r for r in blob["runs"]
                        if not (r.get("model") == MODEL and r.get("method") == "TRIAD")]
        blob["runs"].append(row)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(blob, indent=2))
        print(f"wrote {out}")
    else:
        print(f"(skipped llm_sweep.json: not present)")

    # Phase-3 acceptance file referenced by ADRs / STATUS.md
    acc = ROOT / "results" / "triad_tinyllama_int4_m1.json"
    acc.parent.mkdir(parents=True, exist_ok=True)
    acc.write_text(json.dumps({
        "model": MODEL,
        "method": "TRIAD",
        "bits": 4,
        "group_size": 64,
        "n_calib": 8,
        "seq_len_calib": 512,
        "calib_sec": calib_sec,
        "ppl_wikitext2": res["ppl"],
        "n_eval_tokens": res["n_tokens"],
        "tok_per_sec_decode_m1": tps,
        "mps_peak_gb_during_calib": peak_alloc,
        "samples": samples,
        "notes": (
            "Streaming compile_model + a_device='cpu' so Gram matrices live on "
            "host RAM. PPL acceptance for Phase 3 is <= 9.45 (FP16 baseline 8.45 "
            "+ 1.0 budget) -- this script's 8-sample seq=512 calib is the Phase-2 "
            "OOM-fix smoke; the full Phase-3 (n_calib=128, seq=2048) is in "
            "experiments/13_tinyllama_phase3.py if reached."
        ),
    }, indent=2))
    print(f"wrote {acc}")


if __name__ == "__main__":
    main()
