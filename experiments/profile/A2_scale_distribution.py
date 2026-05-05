"""A.2 alternative: compare per-group fp16 scale + zero distributions
between the TRIAD bundle and the community reference.

Both bundles compile to bit-identical kernel code (verified by md5 of
llama_q4f16_1_devc.o). Therefore any device-side timing gap can only
come from:
  - Memory access patterns (cache hit rates from weight values).
    Layout is identical -> cache footprint identical.
  - Numerical fast/slow paths (denormals, NaN/Inf, very small scales
    that cause subnormal arithmetic in the dequantize).

This script extracts per-group fp16 scales from each bundle and emits
descriptive stats: distribution shape, fraction near subnormal threshold,
fraction in [-2^-14, 2^-14] (fp16 subnormal boundary), and the
per-record histogram of magnitudes.

Output: experiments/profile/A2_scale_distribution_summary.json
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRIAD_DIR = Path("/tmp/triad-tinyllama-int4-mlc")
REF_DIR = Path("/tmp/tinyllama-mlc-ref")
OUT = ROOT / "experiments" / "profile" / "A2_scale_distribution_summary.json"


def _load_records(bundle_dir: Path) -> list[dict]:
    """Read tensor-cache.json (or ndarray-cache.json) and return the
    flat list of records with their byte offsets in shard files."""
    cache_path = bundle_dir / "tensor-cache.json"
    if not cache_path.exists():
        cache_path = bundle_dir / "ndarray-cache.json"
    cache = json.loads(cache_path.read_text())
    out = []
    for shard in cache["records"]:
        path = bundle_dir / shard["dataPath"]
        for rec in shard["records"]:
            rec["shard_path"] = str(path)
            out.append(rec)
    return out


def _dtype_to_np(dtype: str) -> np.dtype:
    return {
        "uint32": np.uint32,
        "float16": np.float16,
        "float32": np.float32,
        "int32": np.int32,
        "int8": np.int8,
        "int16": np.int16,
    }[dtype]


def _read_tensor(rec: dict) -> np.ndarray:
    with open(rec["shard_path"], "rb") as f:
        f.seek(int(rec["byteOffset"]))
        raw = f.read(int(rec["nbytes"]))
    arr = np.frombuffer(raw, dtype=_dtype_to_np(rec["dtype"]))
    return arr.reshape(rec["shape"])


def _scale_stats(scales: np.ndarray, name: str) -> dict:
    s = scales.astype(np.float32)
    abs_s = np.abs(s)
    # fp16 normal range: smallest normal ~ 6.10e-5; smallest subnormal ~ 5.96e-8
    n_total = s.size
    n_subnormal = int(np.sum(abs_s < 6.10e-5))
    n_tiny = int(np.sum(abs_s < 1e-3))
    n_denormalish = int(np.sum(abs_s < 1e-4))
    return {
        "name": name,
        "n": int(n_total),
        "min": float(s.min()),
        "max": float(s.max()),
        "abs_min": float(abs_s.min()),
        "abs_max": float(abs_s.max()),
        "abs_mean": float(abs_s.mean()),
        "abs_p50": float(np.median(abs_s)),
        "abs_p99": float(np.percentile(abs_s, 99)),
        "abs_p99_9": float(np.percentile(abs_s, 99.9)),
        "frac_subnormal_fp16": n_subnormal / max(n_total, 1),
        "frac_below_1e-4": n_denormalish / max(n_total, 1),
        "frac_below_1e-3": n_tiny / max(n_total, 1),
    }


def main():
    triad_recs = _load_records(TRIAD_DIR)
    ref_recs = _load_records(REF_DIR)
    print(f"TRIAD records: {len(triad_recs)}; REF records: {len(ref_recs)}")

    # Aggregate q_scale records (per-group fp16 scales)
    triad_scales = []
    ref_scales = []
    triad_per_layer = {}
    ref_per_layer = {}
    for rec in triad_recs:
        if rec["name"].endswith(".q_scale"):
            arr = _read_tensor(rec).astype(np.float32).flatten()
            triad_scales.append(arr)
            triad_per_layer[rec["name"]] = _scale_stats(arr, rec["name"])
    for rec in ref_recs:
        if rec["name"].endswith(".q_scale"):
            arr = _read_tensor(rec).astype(np.float32).flatten()
            ref_scales.append(arr)
            ref_per_layer[rec["name"]] = _scale_stats(arr, rec["name"])

    triad_all = np.concatenate(triad_scales)
    ref_all = np.concatenate(ref_scales)
    print(f"TRIAD: {len(triad_all)} scales total; REF: {len(ref_all)} scales total")

    summary = {
        "global": {
            "triad": _scale_stats(triad_all, "ALL_TRIAD"),
            "reference": _scale_stats(ref_all, "ALL_REF"),
        },
        "per_layer_count": {
            "triad": len(triad_per_layer),
            "reference": len(ref_per_layer),
        },
        "delta_summary": {},
    }

    # Compute key deltas
    g = summary["global"]
    summary["delta_summary"] = {
        "abs_min_ratio_T_over_R": g["triad"]["abs_min"] / max(g["reference"]["abs_min"], 1e-30),
        "abs_max_ratio_T_over_R": g["triad"]["abs_max"] / max(g["reference"]["abs_max"], 1e-30),
        "frac_subnormal_T_minus_R": g["triad"]["frac_subnormal_fp16"] - g["reference"]["frac_subnormal_fp16"],
        "frac_below_1e-4_T_minus_R": g["triad"]["frac_below_1e-4"] - g["reference"]["frac_below_1e-4"],
        "frac_below_1e-3_T_minus_R": g["triad"]["frac_below_1e-3"] - g["reference"]["frac_below_1e-3"],
        "p99_9_ratio_T_over_R": g["triad"]["abs_p99_9"] / max(g["reference"]["abs_p99_9"], 1e-30),
    }

    # Per-layer worst offenders: layers where TRIAD's fraction near
    # subnormal is most elevated over the reference.
    common = set(triad_per_layer) & set(ref_per_layer)
    elev = []
    for k in common:
        t = triad_per_layer[k]
        r = ref_per_layer[k]
        elev.append({
            "layer": k,
            "T_frac_below_1e-4": t["frac_below_1e-4"],
            "R_frac_below_1e-4": r["frac_below_1e-4"],
            "delta_pct_pts": (t["frac_below_1e-4"] - r["frac_below_1e-4"]) * 100,
            "T_abs_min": t["abs_min"],
            "R_abs_min": r["abs_min"],
        })
    elev.sort(key=lambda x: -x["delta_pct_pts"])
    summary["top_5_layers_more_subnormalish_in_TRIAD"] = elev[:5]
    summary["bottom_5_layers_more_subnormalish_in_REF"] = elev[-5:]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}\n")
    print(json.dumps(summary["global"], indent=2))
    print(json.dumps(summary["delta_summary"], indent=2))


if __name__ == "__main__":
    main()
