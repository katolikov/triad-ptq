"""Phase-3 v2 calibration: TRIAD-INT4 TinyLlama-1.1B with B.2/B.4 stack.

Differences from experiments/13_tinyllama_phase3.py:
  B.2  n_calib 8 -> 64 (+8x; literature shows GPTQ stability past n=32)
  B.3  GPTAQ deferred -- full asymmetric multi-pass calibration does not fit
       this session's time budget; documented in docs/decisions/007.
  B.4  per-group activation-aware clip search before GPTQ Cholesky
  B.5  asymmetric quantization confirmed ON (no code change required)

The new bundle ships to /tmp/triad-tinyllama-int4-v2/model.pt and is the
input for the v2 MLC export.

Run:
    HF_HOME=$(pwd)/.cache/hf uv run python experiments/16_tinyllama_phase3_v2.py
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
CKPT_DIR = Path("/tmp/triad-tinyllama-int4-v2")
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

    # B.2: n_calib bumped 8 -> 64
    n_calib = 64
    seq_calib = 512
    calib = build_wikitext_calib(tok, n_samples=n_calib, seq_len=seq_calib, device=dev)
    print(f"calib v2: {n_calib} batches of seq_len={seq_calib}", flush=True)

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
        # B.4: new knob introduced this session
        clip_search=True,
    )
    calib_sec = time.perf_counter() - t0

    peak_alloc = (
        torch.mps.driver_allocated_memory() / 1e9
        if dev.type == "mps" and hasattr(torch.mps, "driver_allocated_memory") else 0.0
    )
    print(
        f"TRIAD v2 compile: {calib_sec:.0f}s | MPS driver={peak_alloc:.2f} GB",
        flush=True,
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print("saving v2 checkpoint to CPU and disk...", flush=True)
    m_cpu = m.to("cpu")
    if dev.type == "mps":
        torch.mps.empty_cache()
    gc.collect()
    torch.save(m_cpu.state_dict(), str(CKPT_PATH))
    META_PATH.write_text(json.dumps({
        "model": MODEL,
        "method": "TRIAD-v2",
        "bits": 4,
        "group_size": 64,
        "n_calib": n_calib,
        "seq_len_calib": seq_calib,
        "super_weight_frac": 5e-4,
        "bit_allocator": "trace",
        "cov_grid": "analytic",
        "gptq_variant": "standard (GPTAQ deferred per ADR-007)",
        "clip_search": True,
        "asymmetric_quant": True,
        "calib_sec": calib_sec,
        "peak_mps_gb_during_calib": peak_alloc,
    }, indent=2))
    print(f"checkpoint: {CKPT_PATH} ({CKPT_PATH.stat().st_size / 1e6:.1f} MB)", flush=True)

    # PPL eval matched window with v1 for delta comparability
    print("PPL eval on CPU (matched 4096-token window) ...", flush=True)
    m_cpu.eval()
    res = perplexity(
        m_cpu, tok, text_eval,
        device=torch.device("cpu"),
        seq_len=512, max_tokens=4096, progress=False,
    )
    ppl = float(res["ppl"])
    print(
        f"v2 PPL={ppl:.3f} on {res['n_tokens']} tokens in {res['sec']:.1f}s",
        flush=True,
    )

    out = ROOT / "results" / "triad_tinyllama_int4_v2_m1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": MODEL,
        "method": "TRIAD-v2",
        "bits": 4,
        "group_size": 64,
        "n_calib": n_calib,
        "seq_len_calib": seq_calib,
        "super_weight_frac": 5e-4,
        "gptq_variant": "standard (GPTAQ deferred per ADR-007)",
        "clip_search": True,
        "asymmetric_quant": True,
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
