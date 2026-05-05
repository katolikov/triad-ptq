"""Public API for TRIAD-PTQ. Stub during Phase 2; populated in Phase 3-4."""
from __future__ import annotations

from typing import Iterable

import torch.nn as nn


def optimize(
    model: nn.Module,
    *,
    bits: int = 4,
    calibration: Iterable | None = None,
    super_weight_frac: float | None = 1e-3,
    bit_allocator: str = "trace",
    cov_grid: str = "analytic",
    target: str = "auto",
    time_budget_min: float = 5.0,
    group_size: int = 64,
    n_calib: int = 128,
    device: str | None = None,
    a_device: str | None = None,
    forward_fn=None,
    output_fn=None,
    rho_probe_n: int = 4,
    progress: bool = True,
    clip_search: bool = False,
    asymmetric_calib: bool = False,
    asym_alpha: float = 0.5,
    asym_exclude_suffixes: tuple = ("o_proj", "down_proj"),
    return_meta: bool = False,
):
    """One-line PTQ entry point.

    See README "Quickstart" for usage. The function returns the same `model`
    instance with quantizable layers replaced in place by TriadLinear /
    TriadConv2d wrappers.
    """
    from .compile import compile_model

    return compile_model(
        model,
        bits=bits,
        calibration=calibration,
        super_weight_frac=super_weight_frac,
        bit_allocator=bit_allocator,
        cov_grid=cov_grid,
        target=target,
        time_budget_min=time_budget_min,
        group_size=group_size,
        n_calib=n_calib,
        device=device,
        a_device=a_device,
        forward_fn=forward_fn,
        output_fn=output_fn,
        rho_probe_n=rho_probe_n,
        progress=progress,
        clip_search=clip_search,
        asymmetric_calib=asymmetric_calib,
        asym_alpha=asym_alpha,
        asym_exclude_suffixes=asym_exclude_suffixes,
        return_meta=return_meta,
    )
