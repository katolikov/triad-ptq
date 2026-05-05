"""Phase F — GPTAQ ρ-weighted α scheduling.

Modifies (without rewriting) v1's GPTAQ asymmetric calibration in
`triad_ptq/core/gptaq_asym.py` + the alpha-mix step in
`triad_ptq/compile.py`. v1 hard-coded ``α = 0.5`` (ADR-010); v2 schedules
α per block based on the Squisher ρ:

    α^(ℓ) = min(α_max, sigmoid(c · log ρ^(ℓ)))                 (F1)

with default ``c = 1.0`` and ``α_max = 0.8``. The plan requires:

* α stays in [0, α_max] (sigmoid range bounded by α_max).
* The scope-limit from ADR-010 (exclude `o_proj` and `down_proj` from
  the asymmetric correction) is preserved.
* Fixed-α=0.5 ablation reproduces v1 GPTAQ numbers within 0.05 PPL.

This module exposes the α function and a monitoring writer; it does NOT
itself patch `compile.py` — Phase H wires it in by routing the result
through the existing `asym_alpha` argument.

Per-block α is logged to `results/tables/v2_gptaq_alpha.json` for any
calibration run that calls `write_alpha_log`.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

IMPLEMENTED = True
DEFAULT_C = 1.0
DEFAULT_ALPHA_MAX = 0.8
DEFAULT_EXCLUDE_SUFFIXES = ("o_proj", "down_proj")


def alpha_from_rho(rho: float, *, c: float = DEFAULT_C, alpha_max: float = DEFAULT_ALPHA_MAX) -> float:
    """Map a single block's ρ to its α via Eq. (F1).

    α(ρ) = min(α_max, sigmoid(c · log ρ))

    Properties (verified by unit tests):
      * α(1.0) = sigmoid(0) = 0.5  → matches v1's hard-coded default.
      * α(ρ → 0) → 0,  α(ρ → ∞) → α_max  (saturating).
      * Strictly increasing in ρ.
    """
    if not math.isfinite(rho):
        raise ValueError(f"alpha_from_rho: ρ must be finite, got {rho}")
    if rho <= 0.0:
        return 0.0
    if not (0.0 < alpha_max <= 1.0):
        raise ValueError(f"alpha_max must be in (0, 1], got {alpha_max}")
    log_rho = math.log(rho)
    sig = 1.0 / (1.0 + math.exp(-c * log_rho))
    return min(alpha_max, sig)


def alpha_schedule(
    rho_per_block: dict[str, float],
    *,
    c: float = DEFAULT_C,
    alpha_max: float = DEFAULT_ALPHA_MAX,
    exclude_suffixes: tuple[str, ...] = DEFAULT_EXCLUDE_SUFFIXES,
) -> dict[str, float]:
    """Compute α per (block, layer) name. Layers whose name ends with any
    suffix in `exclude_suffixes` get α = 0 (i.e. fall back to v1
    symmetric calibration on those, per ADR-010).
    """
    out: dict[str, float] = {}
    for name, rho in rho_per_block.items():
        if any(name.endswith(suf) for suf in exclude_suffixes):
            out[name] = 0.0
        else:
            out[name] = alpha_from_rho(rho, c=c, alpha_max=alpha_max)
    return out


@dataclass
class AlphaLogEntry:
    block: str
    rho: float
    alpha: float
    excluded: bool


def write_alpha_log(
    rho_per_block: dict[str, float],
    alpha_per_block: dict[str, float],
    output_path: str | Path,
    *,
    c: float = DEFAULT_C,
    alpha_max: float = DEFAULT_ALPHA_MAX,
    exclude_suffixes: tuple[str, ...] = DEFAULT_EXCLUDE_SUFFIXES,
) -> Path:
    """Write per-block α/ρ to a JSON for Phase H result archives.

    Format::
        {
          "schema":          "v2_gptaq_alpha/1",
          "config":          {"c": float, "alpha_max": float,
                              "exclude_suffixes": [...]},
          "n_blocks":        int,
          "n_excluded":      int,
          "alpha_min":       float,
          "alpha_max_obs":   float,
          "alpha_mean":      float,
          "entries":         [{"block": str, "rho": float, "alpha": float, "excluded": bool}, ...]
        }
    """
    entries: list[AlphaLogEntry] = []
    for name in sorted(rho_per_block):
        if name not in alpha_per_block:
            continue
        excluded = any(name.endswith(suf) for suf in exclude_suffixes)
        entries.append(
            AlphaLogEntry(
                block=name,
                rho=float(rho_per_block[name]),
                alpha=float(alpha_per_block[name]),
                excluded=excluded,
            )
        )

    alphas = [e.alpha for e in entries if not e.excluded]
    payload: dict = {
        "schema": "v2_gptaq_alpha/1",
        "config": {
            "c": c,
            "alpha_max": alpha_max,
            "exclude_suffixes": list(exclude_suffixes),
        },
        "n_blocks": len(entries),
        "n_excluded": sum(1 for e in entries if e.excluded),
        "alpha_min": (min(alphas) if alphas else None),
        "alpha_max_obs": (max(alphas) if alphas else None),
        "alpha_mean": (sum(alphas) / len(alphas) if alphas else None),
        "entries": [asdict(e) for e in entries],
    }
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    return p


__all__ = [
    "DEFAULT_C",
    "DEFAULT_ALPHA_MAX",
    "DEFAULT_EXCLUDE_SUFFIXES",
    "IMPLEMENTED",
    "AlphaLogEntry",
    "alpha_from_rho",
    "alpha_schedule",
    "write_alpha_log",
]
