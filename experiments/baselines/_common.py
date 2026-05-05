"""Shared helpers for the v2 baseline runners.

Each baseline script is a thin wrapper that:
  1. Reads a list of HF model IDs from CLI args (default: the v2 sweep set
     llama-3.2-1B, TinyLlama-1.1B, SmolLM-360M).
  2. Runs the baseline's quantize step (autoawq, gptq, gptaq, ...).
  3. Optionally exports to MLC q4f16_1 via `mlc_llm convert_weight + compile`.
  4. Evaluates WikiText-2 PPL on the M1/CPU/CUDA host.
  5. Writes a JSON to `results/baselines/<method>__<model>.json` with
     the schema below.

Result schema
-------------
{
    "method":          "autoawq" | "gptq" | "gptaq" | "quarot_offline" | ...,
    "model_id":        str,                   # HF repo id or local path
    "bits":            4,
    "group_size":      32,
    "wt2_ppl":         float | None,          # None on OOM / failure
    "calib_seconds":   float | None,
    "host":            {"device": str, "torch": str, ...},
    "exit_status":     "ok" | "oom" | "skipped:reason" | "error:msg",
    "git_commit":      str,                    # this repo's HEAD
    "timestamp_utc":   str,
}

The runners are *executable but skip-by-default* — invoking them on the M1
calibration host without the CUDA-only deps (autoawq, the official GPTQ
repo, the GPTAQ repo) returns `exit_status: "skipped:no_cuda"` and writes
the JSON. This is intentional: the v2 plan gates Phase A only on the
runners' EXISTENCE, not on numbers, since real baseline numbers must be
collected on the 4090 calibration host (which is not the M1 dev host).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODEL_IDS = (
    "meta-llama/Llama-3.2-1B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "HuggingFaceTB/SmolLM-360M",
)
RESULT_DIR = Path("results/baselines")


def host_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python":   sys.version.split()[0],
        "platform": sys.platform,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        info["mps"]  = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        info["torch"] = None
    return info


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def write_result(
    *,
    method: str,
    model_id: str,
    exit_status: str,
    bits: int = 4,
    group_size: int = 32,
    wt2_ppl: float | None = None,
    calib_seconds: float | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = model_id.replace("/", "_")
    out = RESULT_DIR / f"{method}__{safe_model}.json"
    payload = {
        "method":         method,
        "model_id":       model_id,
        "bits":           bits,
        "group_size":     group_size,
        "wt2_ppl":        wt2_ppl,
        "calib_seconds":  calib_seconds,
        "host":           host_info(),
        "exit_status":    exit_status,
        "git_commit":     git_commit(),
        "timestamp_utc":  _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }
    if extra:
        payload["extra"] = extra
    out.write_text(json.dumps(payload, indent=2))
    return out


def parse_models_argv(argv: list[str] | None = None) -> tuple[str, ...]:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        return DEFAULT_MODEL_IDS
    return tuple(argv)


def cuda_or_skip(method: str) -> bool:
    """Return True iff CUDA is available; otherwise emit a skipped result for
    each model in the default set and return False.

    The CUDA-only baselines (autoawq inference engine, GPTAQ official repo,
    QuaRot R1+R2) cannot run on the M1 dev host — only the 4090
    calibration host. We honour that gate explicitly here.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    for mid in DEFAULT_MODEL_IDS:
        write_result(method=method, model_id=mid, exit_status="skipped:no_cuda")
    print(f"[{method}] CUDA not available — wrote skipped results for {len(DEFAULT_MODEL_IDS)} models", file=sys.stderr)
    return False
