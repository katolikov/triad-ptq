"""Phase D — Selective OmniQuant-style Learnable Weight Clipping.

Activated only on blocks with ρ^(ℓ) above the 75th percentile (top ~25 %,
typically 5–6 blocks in a 22-layer 1B model). Per-group α_g ∈ [α_min,
α_max] replaces the per-group max::

    s_g = α_g · max|W_g| / (2^{b−1} − 1)

Trained jointly with learnable β in 100 Adam steps (Phase D). Frozen
α_g = 1.0 below the percentile threshold.

The plumbing here is intentionally *configuration-shaped*; the actual
training loop lives in
:mod:`triad_ptq._v2.transform.learnable_beta::train_learnable_beta` so
β and α_g share the same Adam optimiser.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

IMPLEMENTED = True
LWC_THRESHOLD_PERCENTILE_DEFAULT = 75.0
ALPHA_MIN_DEFAULT = 0.5
ALPHA_MAX_DEFAULT = 1.0


# --------------------------------------------------------------------- config

@dataclass
class LWCConfig:
    """Caller-supplied LWC configuration for a single block.

    `enabled` is set per-block by the selector based on the block's ρ
    rank (Phase D2). `alpha_min` / `alpha_max` clip the learnable α_g to
    a meaningful range — OmniQuant uses [0.5, 1.0]; values below 0.5
    drive too much clipping at INT4 g64.
    """

    enabled: bool = True
    alpha_min: float = ALPHA_MIN_DEFAULT
    alpha_max: float = ALPHA_MAX_DEFAULT


# --------------------------------------------------------------------- params

@dataclass
class LWCParameters:
    """Holder for the per-group α tensor and its clamp bounds.

    Lives alongside the learnable β trainer in
    `learnable_beta.train_learnable_beta`.
    """

    alpha: nn.Parameter
    alpha_min: float = ALPHA_MIN_DEFAULT
    alpha_max: float = ALPHA_MAX_DEFAULT


# --------------------------------------------------------------------- selector

def select_lwc_blocks(
    rho_per_block: dict[str, float],
    *,
    threshold_percentile: float = LWC_THRESHOLD_PERCENTILE_DEFAULT,
) -> dict[str, bool]:
    """Decide which blocks get LWC enabled, given per-block ρ values.

    Returns a dict ``{block_name: enabled}``. The decision rule:
    LWC enabled iff ρ^(ℓ) ≥ percentile(rho_values, threshold_percentile).
    """
    if not rho_per_block:
        return {}
    if not (0.0 <= threshold_percentile <= 100.0):
        raise ValueError(f"threshold_percentile must be in [0, 100], got {threshold_percentile}")
    values = torch.tensor(list(rho_per_block.values()), dtype=torch.float64)
    cutoff = float(torch.quantile(values, threshold_percentile / 100.0).item())
    return {name: float(v) >= cutoff for name, v in rho_per_block.items()}


def apply_selective_lwc(
    rho_per_block: dict[str, float],
    *,
    threshold_percentile: float = LWC_THRESHOLD_PERCENTILE_DEFAULT,
    alpha_min: float = ALPHA_MIN_DEFAULT,
    alpha_max: float = ALPHA_MAX_DEFAULT,
) -> dict[str, LWCConfig]:
    """Build per-block LWCConfig given a ρ map and a percentile threshold.

    Returns ``{block_name: LWCConfig}`` ready to feed into
    :func:`triad_ptq._v2.transform.learnable_beta.train_learnable_beta`.
    """
    sel = select_lwc_blocks(rho_per_block, threshold_percentile=threshold_percentile)
    return {
        name: LWCConfig(enabled=enabled, alpha_min=alpha_min, alpha_max=alpha_max)
        for name, enabled in sel.items()
    }


__all__ = [
    "IMPLEMENTED",
    "LWC_THRESHOLD_PERCENTILE_DEFAULT",
    "ALPHA_MIN_DEFAULT",
    "ALPHA_MAX_DEFAULT",
    "LWCConfig",
    "LWCParameters",
    "select_lwc_blocks",
    "apply_selective_lwc",
]
