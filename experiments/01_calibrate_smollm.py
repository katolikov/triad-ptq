"""Phase-4 integration: SmolLM-135M -> TRIAD-INT4 -> WikiText-2 PPL.

Run:  uv run python experiments/01_calibrate_smollm.py

This is the smoke-test experiment. We compare FP16, RTN-INT4 and TRIAD-INT4
on a small calibration / eval window so the whole loop fits in <5 min on M1.
The full sweep is in experiments/10_compare_all_models.py.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Repo-local HF cache so we don't pollute $HOME
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from triad_ptq import optimize  # noqa: E402
from triad_ptq.baselines.rtn import quantize_rtn  # noqa: E402
from triad_ptq.eval.calib import build_wikitext_calib  # noqa: E402
from triad_ptq.eval.ppl import load_wikitext2, perplexity  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM-135M"
SEQ = 1024
N_CALIB = 32
EVAL_TOKENS = 32_768  # subset; full WT2 test ~280k tokens

console = Console()


def main():
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    console.log(f"device={dev}, model={MODEL}")

    tok = AutoTokenizer.from_pretrained(MODEL)

    def fresh_model():
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
        m.to(dev).eval()
        return m

    text_eval = load_wikitext2("test")

    results = []

    # ---- FP32 baseline (we use FP32 since MPS doesn't fully accelerate FP16 GEMM
    # for some HF blocks; the PPL number is what matters) ----
    console.log("[FP32] eval")
    m = fresh_model()
    t0 = time.perf_counter()
    res = perplexity(m, tok, text_eval, device=dev, seq_len=SEQ, max_tokens=EVAL_TOKENS, progress=False)
    res["method"] = "FP32"
    res["bits"] = 32
    res["calib_sec"] = 0.0
    res["eval_sec"] = res["sec"]
    console.log(f"  PPL={res['ppl']:.3f} on {res['n_tokens']} tokens in {res['sec']:.1f}s")
    results.append(res)
    del m

    # ---- RTN-INT4 ----
    console.log("[RTN-INT4]")
    m = fresh_model()
    t0 = time.perf_counter()
    quantize_rtn(m, bits=4, group_size=64, device=dev)
    calib_sec = time.perf_counter() - t0
    res = perplexity(m, tok, text_eval, device=dev, seq_len=SEQ, max_tokens=EVAL_TOKENS, progress=False)
    res["method"] = "RTN"
    res["bits"] = 4
    res["calib_sec"] = calib_sec
    res["eval_sec"] = res["sec"]
    console.log(f"  PPL={res['ppl']:.3f} (calib {calib_sec:.1f}s, eval {res['sec']:.1f}s)")
    results.append(res)
    del m

    # ---- TRIAD-INT4 ----
    console.log("[TRIAD-INT4]")
    m = fresh_model()
    calib = build_wikitext_calib(tok, n_samples=N_CALIB, seq_len=SEQ, device=dev)
    t0 = time.perf_counter()
    optimize(
        m, bits=4, calibration=calib,
        super_weight_frac=5e-4,
        bit_allocator="trace",
        cov_grid="analytic",
        n_calib=N_CALIB,
        rho_probe_n=2,
        group_size=64,
        progress=True,
    )
    calib_sec = time.perf_counter() - t0
    res = perplexity(m, tok, text_eval, device=dev, seq_len=SEQ, max_tokens=EVAL_TOKENS, progress=False)
    res["method"] = "TRIAD"
    res["bits"] = 4
    res["calib_sec"] = calib_sec
    res["eval_sec"] = res["sec"]
    console.log(f"  PPL={res['ppl']:.3f} (calib {calib_sec:.1f}s, eval {res['sec']:.1f}s)")
    results.append(res)
    del m

    out_path = ROOT / "results" / "tables" / "smollm135_smoke.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": MODEL, "results": results, "eval_tokens": EVAL_TOKENS}, f, indent=2)
    console.log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
