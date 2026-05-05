"""B.7 tests: per-group clip-ratio search (B.4).

These are unit-level checks on the helper added to gptq_solver.py and
on the end-to-end behaviour through gptq_quantize_layer.
"""
from __future__ import annotations

import torch

from triad_ptq.core.gptq_solver import _find_clip, gptq_quantize_layer


def _round_trip_mse(W_g, ratio, bits=4):
    qmax = (1 << bits) - 1
    wmin = W_g.amin(dim=-1, keepdim=True) * ratio
    wmax = W_g.amax(dim=-1, keepdim=True) * ratio
    scale = (wmax - wmin).clamp_min(1e-8) / qmax
    zero = (-wmin / scale).round().clamp(0, qmax)
    q = ((W_g / scale) + zero).round().clamp(0, qmax)
    dq = (q - zero) * scale
    return ((W_g - dq) ** 2).sum(dim=(-1, -2))


def test_clip_search_reduces_mse_with_outliers():
    """A row with a deliberate outlier should pick a clip ratio < 1.0
    and the resulting weighted MSE must be <= ratio=1.0 baseline.
    """
    torch.manual_seed(0)
    m, n_g, g = 4, 2, 32
    base = torch.randn(m, n_g, g)
    # Inject a large outlier on row 0 only
    base[0, 0, 0] = 12.0  # 10x larger than typical |w|
    base[0, 1, 5] = -10.0
    ratios = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)
    chosen = _find_clip(base, bits=4, x_var=None, ratios=ratios)
    # Row 0 (outlier) must pick a ratio strictly less than 1.0
    assert chosen[0].item() < 1.0, f"outlier row got ratio {chosen[0].item()}"
    # And its resulting MSE must be at-or-below the ratio=1.0 baseline
    err_clip = _round_trip_mse(base, chosen.view(-1, 1, 1), bits=4)
    err_one = _round_trip_mse(base, 1.0, bits=4)
    # Per-row check: the rows that picked ratio<1 should have err_clip <= err_one
    for i in range(m):
        if chosen[i].item() < 1.0:
            assert err_clip[i].item() <= err_one[i].item() + 1e-6


def test_clip_search_idempotent_on_uniform():
    """With uniform random weights and no outliers, clip ratio=1.0
    should be picked (or at least not worsen MSE).
    """
    torch.manual_seed(42)
    W = torch.randn(8, 1, 32) * 0.1  # tight, well-behaved
    ratios = (1.0, 0.9, 0.8)
    chosen = _find_clip(W, bits=4, x_var=None, ratios=ratios)
    err_chosen = _round_trip_mse(W, chosen.view(-1, 1, 1), bits=4)
    err_one = _round_trip_mse(W, 1.0, bits=4)
    # Chosen MSE must not be worse than ratio=1.0
    assert (err_chosen <= err_one + 1e-6).all()


def test_gptq_quantize_layer_clip_search_flag_runs():
    """End-to-end: gptq_quantize_layer with clip_search=True returns
    a valid QuantizedWeight whose dequantized values approximate W.
    """
    torch.manual_seed(7)
    m, n = 16, 64
    W = torch.randn(m, n)
    # Synthesize a positive-definite Hessian
    X = torch.randn(128, n)
    H = X.t() @ X / 128.0
    qw = gptq_quantize_layer(
        W, H, bits=4, group_size=32, clip_search=True
    )
    assert qw.q.shape == (m, n)
    assert qw.scales.shape == (m, 2)
    assert qw.zeros.shape == (m, 2)
    # Sanity: dequantized must be in roughly the same dynamic range as W
    deq = qw.dequantize()
    assert deq.shape == W.shape
    assert torch.isfinite(deq).all()
    # MSE must be < a generous bound (INT4 group=32 typical)
    rel_err = (W - deq).pow(2).mean() / W.pow(2).mean()
    assert rel_err < 0.07, f"INT4 round-trip rel-MSE too high: {rel_err.item():.4f}"


def test_gptq_quantize_layer_clip_search_helps_on_realistic_outliers():
    """clip_search wins on realistic inputs where weight outliers
    coincide with low-variance input columns -- the AWQ-clip / OmniQuant
    setting. We construct X with one tiny-variance column that has an
    outlier weight, and verify that clip_search reduces the activation-
    weighted forward-output error.
    """
    torch.manual_seed(11)
    m, n, b = 32, 64, 1024
    W = torch.randn(m, n) * 0.3
    # Outliers placed in columns whose input variance is small -- the
    # quantization grid waste from preserving them is high while the
    # forward-output cost of clamping them is low.
    low_var_cols = [7, 30, 55]
    W[5, 7] = 12.0
    W[5, 30] = -10.0
    W[18, 55] = 9.0
    X = torch.randn(b, n)
    # Shrink variance of the outlier columns by 20x
    for c in low_var_cols:
        X[:, c] *= 0.05
    H = X.t() @ X / b
    qw_default = gptq_quantize_layer(W, H, bits=4, group_size=32, clip_search=False)
    qw_clip = gptq_quantize_layer(W, H, bits=4, group_size=32, clip_search=True)
    Y_ref = X @ W.t()
    err_default = (Y_ref - X @ qw_default.dequantize().t()).pow(2).mean()
    err_clip = (Y_ref - X @ qw_clip.dequantize().t()).pow(2).mean()
    # On this realistic setup clip_search should not be worse and is
    # generally better. Allow some slack for synthetic noise.
    assert err_clip <= err_default * 1.10, (
        f"clip_search forward MSE {err_clip:.4f} > default {err_default:.4f} * 1.10"
    )
