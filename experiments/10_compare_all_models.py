"""Tier-1 LLM sweep: FP32, RTN-INT4, AWQ-like-INT4, TRIAD-INT4.

For each model x method, we record:
  - WikiText-2 perplexity (subset)
  - decode tokens/sec (batch=1, 100 generated tokens, simulated INT4)
  - quantization wall time
  - sample completions for 5 prompts

Results are written incrementally to results/tables/llm_sweep.json so a
crash mid-sweep does not lose prior data.

Usage:
  uv run python experiments/10_compare_all_models.py [--models smollm-135 smollm-360 tinyllama]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from triad_ptq import optimize  # noqa: E402
from triad_ptq.baselines.awq import awq_like_quantize  # noqa: E402
from triad_ptq.baselines.rtn import quantize_rtn  # noqa: E402
from triad_ptq.eval.calib import build_wikitext_calib  # noqa: E402
from triad_ptq.eval.generate import measure_decode_throughput, sample_completions  # noqa: E402
from triad_ptq.eval.ppl import load_wikitext2, perplexity  # noqa: E402

console = Console()

MODEL_REGISTRY = {
    "smollm-135":  ("HuggingFaceTB/SmolLM-135M",      32, 1024, 32_768),
    "smollm-360":  ("HuggingFaceTB/SmolLM-360M",      32, 1024, 32_768),
    "tinyllama":   ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 16, 1024, 16_384),
}

PROMPTS = [
    "The capital of France is",
    "Photosynthesis is the process by which",
    "In a few sentences, the difference between machine learning and traditional programming is",
    "Once upon a time in a small village,",
    "Quantization in deep learning refers to",
]

OUT = ROOT / "results" / "tables" / "llm_sweep.json"
SAMPLES_DIR = ROOT / "results" / "samples"


def _free():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _load_results():
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {"runs": []}


def _save_results(d):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2))


def _key(model: str, method: str, bits: int) -> str:
    return f"{model}::{method}::b{bits}"


def _fresh(model_id, dev):
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    m.to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def run_one(model_id, method, bits, dev, *, n_calib, seq_len, eval_tokens,
            tok, calib, text_eval, samples_dir):
    t0 = time.perf_counter()
    m = _fresh(model_id, dev)
    if method == "FP32":
        calib_sec = 0.0
    elif method == "RTN":
        quantize_rtn(m, bits=bits, group_size=64, device=dev)
        calib_sec = time.perf_counter() - t0
    elif method == "AWQ-like":
        awq_like_quantize(m, calib, bits=bits, group_size=64,
                          n_calib=n_calib, n_grid=10, device=dev)
        calib_sec = time.perf_counter() - t0
    elif method == "TRIAD":
        optimize(m, bits=bits, calibration=calib,
                 super_weight_frac=5e-4, bit_allocator="trace",
                 cov_grid="analytic", n_calib=n_calib,
                 rho_probe_n=2, group_size=64, progress=False)
        calib_sec = time.perf_counter() - t0
    else:
        raise ValueError(method)

    # PPL
    ppl_res = perplexity(m, tok, text_eval, device=dev,
                         seq_len=seq_len, max_tokens=eval_tokens, progress=False)

    # Throughput (best effort -- generation may fail in some quant configs)
    try:
        thr = measure_decode_throughput(m, tok, device=dev, n_tokens=64, n_warmup=4)
    except Exception as e:
        thr = {"error": str(e), "tok_per_sec": None}

    # Samples
    try:
        samples = sample_completions(m, tok, PROMPTS, device=dev, max_new_tokens=40)
    except Exception as e:
        samples = [{"error": str(e)}]

    return {
        "model": model_id,
        "method": method,
        "bits": bits,
        "ppl": ppl_res["ppl"],
        "nll": ppl_res["nll"],
        "n_eval_tokens": ppl_res["n_tokens"],
        "calib_sec": calib_sec,
        "eval_sec": ppl_res["sec"],
        "tok_per_sec": thr.get("tok_per_sec"),
        "samples": samples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["smollm-135", "smollm-360"])
    ap.add_argument("--methods", nargs="*", default=["FP32", "RTN", "AWQ-like", "TRIAD"])
    ap.add_argument("--bits", type=int, default=4)
    args = ap.parse_args()

    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    console.log(f"device={dev}")
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    results = _load_results()
    done_keys = {(r["model"], r["method"], r["bits"]) for r in results["runs"]}

    for tag in args.models:
        if tag not in MODEL_REGISTRY:
            console.log(f"[skip] unknown model {tag}")
            continue
        model_id, n_calib, seq_len, eval_tokens = MODEL_REGISTRY[tag]
        console.log(f"\n=== {model_id}  (n_calib={n_calib}, seq_len={seq_len}) ===")
        tok = AutoTokenizer.from_pretrained(model_id)
        text_eval = load_wikitext2("test")

        # Build calibration once per model (shared across methods)
        m_for_cal = _fresh(model_id, dev)
        calib = build_wikitext_calib(tok, n_samples=n_calib, seq_len=seq_len, device=dev)
        del m_for_cal
        _free()

        for method in args.methods:
            bits = 32 if method == "FP32" else args.bits
            key = (model_id, method, bits)
            if key in done_keys:
                console.log(f"  [skip] {method} bits={bits} already done")
                continue
            console.log(f"  --- {method} bits={bits} ---")
            try:
                row = run_one(
                    model_id, method, bits, dev,
                    n_calib=n_calib, seq_len=seq_len, eval_tokens=eval_tokens,
                    tok=tok, calib=calib, text_eval=text_eval,
                    samples_dir=SAMPLES_DIR,
                )
                console.log(
                    f"    PPL={row['ppl']:.3f}  tok/s={row['tok_per_sec']}  "
                    f"calib={row['calib_sec']:.0f}s eval={row['eval_sec']:.0f}s"
                )
                results["runs"].append(row)
                _save_results(results)
            except Exception:
                tb = traceback.format_exc()
                console.log(f"    [FAIL] {tb}")
                results["runs"].append({
                    "model": model_id, "method": method, "bits": bits,
                    "error": tb[-1000:],
                })
                _save_results(results)
            _free()

        # Save consolidated samples per-model
        runs_for_model = [r for r in results["runs"] if r.get("model") == model_id]
        if runs_for_model:
            sample_blob = {
                "model": model_id,
                "completions": [
                    {
                        "method": r["method"], "bits": r["bits"],
                        "ppl": r.get("ppl"),
                        "samples": r.get("samples", []),
                    }
                    for r in runs_for_model
                ],
            }
            (SAMPLES_DIR / f"{tag}.json").write_text(json.dumps(sample_blob, indent=2))

    console.log(f"\nWrote {OUT} ({len(results['runs'])} runs)")


if __name__ == "__main__":
    main()
