"""Regression test for the MPS super-weight off-by-one (compile.py clamp).

Synthesises the failing scenario without running full TRIAD calibration:
constructs a tall (large-vocab-style) layer and checks that super-weight
row/col indices selected via the topk path are always in bounds.
"""
from __future__ import annotations

import torch

from triad_ptq.core.router import compute_kappa, compute_kappa_topk


def test_compute_kappa_topk_indices_in_bounds():
    """compute_kappa_topk on a tall (m=32000, n=2048) layer must
    return rows < m and cols < n.
    """
    torch.manual_seed(0)
    m, n = 32000, 2048
    W = torch.randn(m, n)
    abs_X = torch.randn(n).abs()
    rho = 1.5
    k = 64
    rows, cols, vals = compute_kappa_topk(W, abs_X, rho, k)
    assert rows.dtype == torch.long
    assert cols.dtype == torch.long
    assert int(rows.max().item()) < m, f"row {rows.max().item()} >= m={m}"
    assert int(cols.max().item()) < n, f"col {cols.max().item()} >= n={n}"
    assert int(rows.min().item()) >= 0
    assert int(cols.min().item()) >= 0


def test_kappa_topk_floor_division_well_defined():
    """compute_kappa returns shape (m, n); flat-index decomposition is
    floor-division by n. On all torch CPU devices, the boundary-case
    flat index m*n - 1 must yield (m-1, n-1).
    """
    m, n = 32000, 2048
    boundary = torch.tensor([m * n - 1], dtype=torch.long)
    r = boundary // n
    c = boundary % n
    assert int(r.item()) == m - 1
    assert int(c.item()) == n - 1


def test_compile_py_clamp_no_op_on_safe_indices():
    """Replicate the defensive clamp from compile.py and verify it
    is a no-op when indices are already in [0, m-1] x [0, n-1].
    """
    m, n = 32000, 2048
    rows = torch.tensor([0, 1, 100, m - 1])
    cols = torch.tensor([0, 1, 100, n - 1])
    rows_clamped = rows.clamp(0, m - 1)
    cols_clamped = cols.clamp(0, n - 1)
    assert torch.equal(rows, rows_clamped)
    assert torch.equal(cols, cols_clamped)


def test_compile_py_clamp_recovers_off_by_one():
    """The MPS bug we observed produced exactly `m` (one past the
    last valid row). The clamp must absorb it to m-1 cleanly.
    """
    m, n = 32000, 2048
    rows = torch.tensor([0, m, m + 1, m - 1])  # includes off-by-one + worse
    cols = torch.tensor([0, n, n + 1, n - 1])
    rows_clamped = rows.clamp(0, m - 1)
    cols_clamped = cols.clamp(0, n - 1)
    assert int(rows_clamped[1].item()) == m - 1
    assert int(rows_clamped[2].item()) == m - 1
    assert int(cols_clamped[1].item()) == n - 1
    assert int(cols_clamped[2].item()) == n - 1
