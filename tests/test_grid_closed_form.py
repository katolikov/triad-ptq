"""Numerical verification of equation (5): closed-form beta*.

The test:
  1. Construct a synthetic linear layer with controlled activation Gram A.
  2. Compute beta* via the closed-form formula (eq 5).
  3. Grid-search beta over [0, 0.5] in steps of 0.01 to minimize the
     same quadratic surrogate objective.
  4. Assert the closed-form beta* is within 0.02 of the grid optimum.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from triad_ptq.core.grid import (
    closed_form_beta,
    closed_form_objective,
    compute_grid,
)


def _rand_psd(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(d, d, generator=g)
    A = R @ R.T / d + 0.05 * torch.eye(d)
    return A


def _grid_search_beta(eig: torch.Tensor, s_spec: torch.Tensor) -> float:
    betas = np.arange(0.0, 0.5 + 1e-9, 0.01)
    vals = [closed_form_objective(eig, s_spec, float(b)) for b in betas]
    return float(betas[int(np.argmin(vals))])


@pytest.mark.parametrize("d,seed", [(64, 0), (64, 7), (128, 3), (32, 11)])
def test_closed_form_matches_grid(d: int, seed: int) -> None:
    torch.manual_seed(seed)
    A = _rand_psd(d, seed)
    W = torch.randn(2 * d, d, generator=torch.Generator().manual_seed(seed + 1))

    res = compute_grid(W, A)
    beta_cf = res.beta_star
    beta_grid = _grid_search_beta(res.eig, res.s_spec)

    assert 0.0 <= beta_cf <= 0.5, f"closed-form beta out of range: {beta_cf}"
    assert abs(beta_cf - beta_grid) < 0.02, (
        f"d={d} seed={seed}: closed-form beta={beta_cf:.4f} vs grid={beta_grid:.4f}"
    )


def test_closed_form_function_directly() -> None:
    """Direct call of closed_form_beta agrees with compute_grid."""
    torch.manual_seed(0)
    d = 96
    A = _rand_psd(d, 99)
    W = torch.randn(150, d)
    res = compute_grid(W, A)
    beta2 = closed_form_beta(res.eig, res.s_spec)
    assert abs(beta2 - res.beta_star) < 1e-6


def test_endpoints_recovered() -> None:
    """When all log-eigenvalues are zero, the formula returns 0/0 -> 0."""
    d = 32
    eig = torch.ones(d)
    s = torch.ones(d)
    # log(1) = 0 -> num=0, den=0 (clamped) -> beta = 0
    b = closed_form_beta(eig, s)
    assert b == 0.0
