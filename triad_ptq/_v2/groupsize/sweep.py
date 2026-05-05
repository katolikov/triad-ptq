"""Phase G — Hardware-aware group-size sweep.

Runs the full v2 calibration pipeline at G ∈ {32, 64, 128} on a model and
emits one packed-weight result per G with metadata for the recommended-
default decision.

Why G=64 is the *candidate* default for Xclipse 950
---------------------------------------------------
- The Xclipse 950 native subgroup size is 64 (wave64), confirmed by the
  Phase-0 Vulkan probe in `docs/probe/SUMMARY.md`.
- G=64 halves the per-group FP16 scale traffic vs G=32 (one fp16 per
  64 weights instead of per 32).
- Quantization quality: the 32→64 group bump nudges per-group MSE up
  modestly; v2's selective LWC + sign+perm rotation absorbs most of it.

The recommendation is **gated on measured Mali decode tok/s being ≥ G=32**.
ADR-015 documents the gating logic and the deferred measurement.

Pipeline-agnostic API
---------------------
This module does NOT itself run TRIAD calibration; it exposes a small
harness that drives a caller-supplied `calibrate_at_g(g) -> SweepRow`
callable and aggregates the results into a single
`results/v2/group_size_sweep__<model>.json`. Phase H's runbook supplies
the calibrate_at_g closure.

Disk-MB estimator
-----------------
:func:`estimate_disk_mb` computes the on-disk size of a v2-packed
ChannelInt8Bundle without actually serializing it: 0.5 byte/weight for
INT4 + 1 byte/weight for INT8 + 2 byte/group for fp16 scales + 1 bit/
output for the bit indicator. This lets us make the disk-MB falsification
gate (H4 #5: ``disk_MB_at_G64 ≤ 0.92 × q4f16_1_G32``) before any Mali
measurement runs.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

IMPLEMENTED = True
GROUP_SIZES_TO_SWEEP: tuple[int, ...] = (32, 64, 128)
RECOMMENDED_DEFAULT_G = 64


# --------------------------------------------------------------------- estimators

def estimate_disk_mb(
    *,
    n_weights_total: int,
    super_channel_rate: float,
    group_size: int,
    out_features: int,
    bits_main: int = 4,
    bits_super: int = 8,
) -> float:
    """Bytes-on-disk estimator for a v2 packed bundle.

    The v2 bundle stores:
      * INT4 main weights:      0.5 byte / weight
      * INT8 super weights:     1.0 byte / weight   (super_channel_rate fraction)
      * fp16 group scales:      2 bytes per (out_channel, group)
      * fp16 super scales:      2 bytes per (super_channel, group)
      * 1-bit indicator:        ceil(out / 8) bytes

    This MATCHES the on-disk layout that Phase E's
    :class:`ChannelInt8Bundle` would serialize via safetensors.
    """
    if not (0.0 <= super_channel_rate < 1.0):
        raise ValueError(f"super_channel_rate must be in [0, 1), got {super_channel_rate}")
    if group_size <= 0 or out_features <= 0:
        raise ValueError("group_size, out_features must be positive")

    n_super = max(1, int(round(super_channel_rate * out_features)))
    n_main = out_features - n_super
    if n_main <= 0:
        raise ValueError("at least one main (INT4) channel must remain")

    # in_features per output channel:
    in_features = n_weights_total // out_features
    if in_features * out_features != n_weights_total:
        raise ValueError("n_weights_total must equal out_features * in_features")
    if in_features % group_size != 0:
        raise ValueError(f"in_features {in_features} must be divisible by group_size {group_size}")
    n_groups = in_features // group_size

    bytes_main_w = (n_main * in_features) * (bits_main / 8.0)
    bytes_super_w = (n_super * in_features) * (bits_super / 8.0)
    bytes_main_s = n_main * n_groups * 2
    bytes_super_s = n_super * n_groups * 2
    bytes_indicator = -(-out_features // 8)  # ceil(out/8)

    total = bytes_main_w + bytes_super_w + bytes_main_s + bytes_super_s + bytes_indicator
    return float(total) / (1024 * 1024)


# --------------------------------------------------------------------- rows

@dataclass
class SweepRow:
    """One row of a group-size sweep result.

    Caller must populate measured fields; estimator fields are computed.
    """

    group_size: int
    wt2_ppl: float | None = None       # measured by caller (FP16 forward through quantized weights)
    decode_tps: float | None = None    # measured ON DEVICE via tools/bench_android.sh
    decode_tps_stdev: float | None = None
    n_iter: int | None = None
    disk_mb: float | None = None       # estimated or measured
    extra: dict = field(default_factory=dict)


@dataclass
class SweepResult:
    model_id: str
    rows: list[SweepRow]
    notes: str = ""

    def best_by_decode(self) -> int | None:
        valid = [r for r in self.rows if r.decode_tps is not None]
        if not valid:
            return None
        return max(valid, key=lambda r: r.decode_tps).group_size

    def best_by_ppl(self) -> int | None:
        valid = [r for r in self.rows if r.wt2_ppl is not None]
        if not valid:
            return None
        return min(valid, key=lambda r: r.wt2_ppl).group_size


# --------------------------------------------------------------------- harness

def run_group_size_sweep(
    model_id: str,
    calibrate_at_g: Callable[[int], SweepRow],
    *,
    group_sizes: Iterable[int] = GROUP_SIZES_TO_SWEEP,
    output_path: str | Path | None = None,
) -> SweepResult:
    """Drive a caller-supplied calibration callable across multiple group
    sizes and aggregate the rows.

    `calibrate_at_g(g)` MUST return a SweepRow; the harness does not
    catch exceptions — the caller decides whether a partial sweep is
    fatal.
    """
    rows: list[SweepRow] = []
    for g in group_sizes:
        row = calibrate_at_g(g)
        if row.group_size != g:
            raise ValueError(
                f"calibrate_at_g returned group_size={row.group_size} != requested {g}"
            )
        rows.append(row)
    res = SweepResult(model_id=model_id, rows=rows)
    if output_path is not None:
        write_sweep_result(res, output_path)
    return res


def write_sweep_result(res: SweepResult, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "v2_group_size_sweep/1",
        "model_id": res.model_id,
        "notes": res.notes,
        "rows": [asdict(r) for r in res.rows],
        "best_by_decode": res.best_by_decode(),
        "best_by_ppl": res.best_by_ppl(),
    }
    p.write_text(json.dumps(payload, indent=2))
    return p


# --------------------------------------------------------------------- decision

def decide_default_group_size(res: SweepResult) -> int:
    """Apply the v2 plan's gating rule:

    * If decode tok/s for G=64 is *at least* the decode tok/s for G=32,
      recommend G=64 (the v2 plan's preferred default).
    * Otherwise revert to G=32 and emit a note in the result for ADR-015.

    If decode measurements are unavailable for either G, raise — the
    decision MUST be measurement-driven, not a guess.
    """
    by_g = {r.group_size: r for r in res.rows}
    if 32 not in by_g or 64 not in by_g:
        raise ValueError("decide_default_group_size: requires both G=32 and G=64 rows")

    r32 = by_g[32]
    r64 = by_g[64]
    if r32.decode_tps is None or r64.decode_tps is None:
        raise ValueError(
            "decide_default_group_size: decode_tps missing — measurement on the "
            "target device required before the default can be set."
        )
    return 64 if r64.decode_tps >= r32.decode_tps else 32


__all__ = [
    "GROUP_SIZES_TO_SWEEP",
    "RECOMMENDED_DEFAULT_G",
    "IMPLEMENTED",
    "SweepRow",
    "SweepResult",
    "estimate_disk_mb",
    "run_group_size_sweep",
    "write_sweep_result",
    "decide_default_group_size",
]
