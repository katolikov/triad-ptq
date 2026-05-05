"""Profile peak memory of compile_model on a single TinyLlama-shaped block.

This is the Phase-2.1 profile artefact referenced by ADR / STATUS.md. The
profile JSON is written to experiments/profiles/tinyllama_cholesky.json.

By design the script targets a *single TinyLlama transformer block*
(not the full model) so it can run on M1 8 GB even before any of the
Phase-2 fixes -- the profile that motivates the fix and the profile that
verifies the fix can be compared 1:1.

Run:
    HF_HOME=$(pwd)/.cache/hf uv run python experiments/profile_cholesky.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Use the repo-local HF cache so this profile can run without network.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))


from triad_ptq import optimize  # noqa: E402


class TinyLlamaBlockShape(nn.Module):
    """One transformer-block-shaped MLP at TinyLlama widths.

    TinyLlama-1.1B: hidden=2048, intermediate=5632, GQA (kv heads=4 -> 256).
    For profile purposes we collapse KV down-projection into a single
    `attn_proj` of shape (2048, 2048) -- the heaviest A is on `down_proj`
    (n=5632), which dominates Cholesky cost and is what we want to profile.
    """

    def __init__(self, d_model: int = 2048, d_ffn: int = 5632):
        super().__init__()
        self.attn_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_ffn, bias=False)
        self.up = nn.Linear(d_model, d_ffn, bias=False)
        self.down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn_proj(x)
        return self.down(torch.nn.functional.silu(self.gate(h)) * self.up(h))


def main() -> None:
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    torch.manual_seed(0)
    model = TinyLlamaBlockShape().to(dev).eval()
    calib = [torch.randn(1, 512, 2048, device=dev) for _ in range(8)]

    out_dir = ROOT / "experiments" / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if dev.type == "mps":
        # MPS profiler activity if present in this torch build
        if hasattr(ProfilerActivity, "MPS"):
            activities.append(ProfilerActivity.MPS)

    with profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        with record_function("triad_compile_one_block"):
            optimize(
                model,
                bits=4,
                calibration=calib,
                super_weight_frac=5e-4,
                bit_allocator="trace",
                cov_grid="analytic",
                n_calib=8,
                rho_probe_n=2,
                group_size=64,
                progress=False,
                a_device="cpu",
            )

    # Sort by self-CPU memory (mps_memory_usage is unavailable in some
    # torch builds; CPU-self is the cross-platform proxy).
    sort_key = "self_cpu_memory_usage"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=30)
    print(table)

    # Trace export for chrome://tracing.
    trace_path = out_dir / "tinyllama_cholesky.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"wrote {trace_path}")

    # Also dump a small summary JSON for easy reading.
    summary_path = out_dir / "tinyllama_cholesky_summary.json"
    rows = []
    for ev in prof.key_averages():
        rows.append({
            "name": ev.key,
            "cpu_time_us": ev.cpu_time_total,
            "self_cpu_memory_bytes": getattr(ev, "self_cpu_memory_usage", 0),
            "self_mps_memory_bytes": getattr(ev, "self_mps_memory_usage", 0),
            "count": ev.count,
        })
    rows.sort(key=lambda r: -r["self_cpu_memory_bytes"])
    summary_path.write_text(json.dumps(rows[:30], indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
