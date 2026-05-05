"""Phase-3 calibration + checkpoint save for TinyLlama-1.1B TRIAD-INT4.

Improvements over experiments/12_tinyllama_triad_lowmem.py:
- Saves the calibrated state_dict to /tmp/triad-tinyllama-int4/model.pt
  IMMEDIATELY after compile_model returns, so the checkpoint survives any
  subsequent eval-time OOM / swap thrash. Phase 4 (MLC export) consumes
  this file directly.
- Skips sample_completions and measure_decode_throughput (both heavy on
  M1 with TinyLlama after TriadLinear adds ~2 GB of U buffers; sampling
  drove the previous run into swap).
- Smaller PPL eval window (1024 tokens, seq=512) so we get a usable PPL
  even if MPS is under pressure.
- Reports peak MPS allocation seen during calibration via
  torch.mps.driver_allocated_memory.

Run:
    HF_HOME=$(pwd)/.cache/hf uv run python experiments/13_tinyllama_phase3.py
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

from triad_ptq import optimize  # noqa: E402
from triad_ptq.eval.calib import build_wikitext_calib  # noqa: E402
from triad_ptq.eval.ppl import load_wikitext2, perplexity  # noqa: E402

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CKPT_DIR = Path("/tmp/triad-tinyllama-int4")
CKPT_PATH = CKPT_DIR / "model.pt"
META_PATH = CKPT_DIR / "meta.json"


def main() -> None:
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"device={dev}, model={MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    text_eval = load_wikitext2("test")

    m = (
        AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
        .to(dev)
        .eval()
    )
    for p in m.parameters():
        p.requires_grad_(False)

    n_calib = 8
    seq_calib = 512
    calib = build_wikitext_calib(tok, n_samples=n_calib, seq_len=seq_calib, device=dev)
    print(f"calib: {n_calib} batches of seq_len={seq_calib}", flush=True)

    if dev.type == "mps":
        torch.mps.empty_cache()

    t0 = time.perf_counter()
    optimize(
        m,
        bits=4,
        calibration=calib,
        super_weight_frac=5e-4,
        bit_allocator="trace",
        cov_grid="analytic",
        n_calib=n_calib,
        rho_probe_n=1,
        group_size=64,
        progress=True,
        device="mps",
        a_device="cpu",
    )
    calib_sec = time.perf_counter() - t0

    peak_alloc = (
        torch.mps.driver_allocated_memory() / 1e9
        if dev.type == "mps" and hasattr(torch.mps, "driver_allocated_memory") else 0.0
    )
    print(
        f"TRIAD compile: {calib_sec:.0f}s | MPS driver={peak_alloc:.2f} GB",
        flush=True,
    )

    # ---- SAVE CHECKPOINT IMMEDIATELY (before any eval that might OOM) ----
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    # Move to CPU for portable saving (also frees MPS for eval).
    print("saving checkpoint to CPU and disk...", flush=True)
    m_cpu = m.to("cpu")
    if dev.type == "mps":
        torch.mps.empty_cache()
    gc.collect()
    torch.save(m_cpu.state_dict(), str(CKPT_PATH))
    META_PATH.write_text(json.dumps({
        "model": MODEL,
        "method": "TRIAD",
        "bits": 4,
        "group_size": 64,
        "n_calib": n_calib,
        "seq_len_calib": seq_calib,
        "super_weight_frac": 5e-4,
        "bit_allocator": "trace",
        "cov_grid": "analytic",
        "calib_sec": calib_sec,
        "peak_mps_gb_during_calib": peak_alloc,
    }, indent=2))
    print(f"checkpoint: {CKPT_PATH} ({CKPT_PATH.stat().st_size / 1e6:.1f} MB)", flush=True)

    # ---- Brief PPL eval (CPU is fine; we just need the number) -----------
    print("PPL eval on CPU (small window) ...", flush=True)
    m_cpu.eval()
    res = perplexity(
        m_cpu, tok, text_eval,
        device=torch.device("cpu"),
        seq_len=512, max_tokens=4096, progress=False,
    )
    ppl = float(res["ppl"])
    print(
        f"PPL={ppl:.3f} on {res['n_tokens']} tokens in {res['sec']:.1f}s",
        flush=True,
    )

    out = ROOT / "results" / "triad_tinyllama_int4_m1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": MODEL,
        "method": "TRIAD",
        "bits": 4,
        "group_size": 64,
        "n_calib": n_calib,
        "seq_len_calib": seq_calib,
        "super_weight_frac": 5e-4,
        "calib_sec": calib_sec,
        "peak_mps_gb_during_calib": peak_alloc,
        "ppl_wikitext2": ppl,
        "ppl_n_tokens": int(res["n_tokens"]),
        "ppl_sec": float(res["sec"]),
        "ppl_eval_device": "cpu",
        "ppl_seq_len": 512,
        "ppl_max_tokens": 4096,
        "checkpoint_path": str(CKPT_PATH),
        "checkpoint_size_mb": CKPT_PATH.stat().st_size / 1e6,
    }, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
