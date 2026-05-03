"""Bit allocator implementing equation (2) of TRIAD-PTQ v1.0.0.

The Lagrangian relaxation of

    min_b sum_l S^(l) * Delta_b_l^2     s.t.  sum_l d_l * b_l <= B_tot
    b_l in {3, 4, 8}

has the watershed solution b_l*  ~  0.5 * log2(S^(l)) + lambda . We solve
for lambda by 1-D bracket search so that the *snapped* allocation hits the
total bit-budget within tolerance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class AllocatorResult:
    bits: list[int]                 # per layer, snapped to allowed set
    continuous: list[float]         # pre-snap value
    lambda_star: float
    achieved_bits_per_w: float
    target_bits_per_w: float


def allocate_bits(
    sensitivities: list[float] | np.ndarray,
    layer_dims: list[int] | np.ndarray,
    target_avg_bits: float,
    *,
    bit_options: tuple[int, ...] = (3, 4, 8),
    tol: float = 1e-3,
    max_iter: int = 64,
) -> AllocatorResult:
    """Distribute bits proportional to log sensitivity, hit target average.

    `target_avg_bits` is the average bit-width per *weight* (not per layer).
    `layer_dims` is the per-layer weight count (m * n).

    Returns AllocatorResult.bits with values from `bit_options`.
    """
    S = np.asarray(sensitivities, dtype=np.float64)
    D = np.asarray(layer_dims, dtype=np.float64)
    assert S.shape == D.shape, "sensitivities and layer_dims must align"
    if S.size == 0:
        return AllocatorResult([], [], 0.0, 0.0, target_avg_bits)

    # Center log-sensitivity at its weighted median so the continuous
    # allocation's natural mean is `target_avg_bits` instead of being
    # dominated by a few very-high-S layers (which would push the
    # remaining majority to the lower clamp).
    log2S = np.log2(np.clip(S, 1e-18, None))
    median_log = float(np.median(log2S))
    log2S = log2S - median_log

    bit_options = tuple(sorted(bit_options))
    bmin, bmax = float(min(bit_options)), float(max(bit_options))

    target_total = target_avg_bits * D.sum()

    def snap(b_continuous: np.ndarray) -> np.ndarray:
        opts = np.asarray(bit_options, dtype=np.float64)
        # nearest in options
        idx = np.argmin(np.abs(b_continuous[:, None] - opts[None, :]), axis=1)
        return opts[idx]

    def total_at(lmb: float, snapped: bool) -> float:
        b = np.clip(0.5 * log2S + lmb, bmin, bmax)
        if snapped:
            b = snap(b)
        return float((b * D).sum())

    # Bracket lambda via continuous total (monotone in lambda)
    lo, hi = -50.0, 50.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        t = total_at(mid, snapped=False)
        if t > target_total:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    lmb = 0.5 * (lo + hi)

    b_cont = np.clip(0.5 * log2S + lmb, bmin, bmax)
    b_snap = snap(b_cont).copy()

    # Greedy correction pass: snapping to {3, 4, 8} is coarse, so we may be
    # several percent off-budget after rounding. Flip layers to adjacent
    # snap options that move us toward target_total, picking flips that
    # cause the largest |error| reduction first.
    opts_arr = np.asarray(bit_options, dtype=np.float64)

    def neighbours(cur: float) -> tuple[float | None, float | None]:
        higher = opts_arr[opts_arr > cur]
        lower = opts_arr[opts_arr < cur]
        return (
            float(higher.min()) if higher.size else None,
            float(lower.max()) if lower.size else None,
        )

    cur_total = float((b_snap * D).sum())
    rel_tol = 1e-3
    for _ in range(8 * S.size):
        err = cur_total - target_total
        if abs(err) / max(target_total, 1.0) < rel_tol:
            break
        best_i, best_new, best_gain = -1, None, 0.0
        for i in range(S.size):
            up, down = neighbours(float(b_snap[i]))
            for new_val in (up, down):
                if new_val is None:
                    continue
                delta = (new_val - b_snap[i]) * D[i]
                new_err = err + delta
                gain = abs(err) - abs(new_err)
                # require strict improvement and direction toward target
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_new = new_val
        if best_i < 0 or best_new is None or best_gain <= 0:
            break
        cur_total += (best_new - b_snap[best_i]) * D[best_i]
        b_snap[best_i] = best_new

    achieved = cur_total / D.sum()
    return AllocatorResult(
        bits=[int(x) for x in b_snap.tolist()],
        continuous=b_cont.tolist(),
        lambda_star=float(lmb),
        achieved_bits_per_w=float(achieved),
        target_bits_per_w=float(target_avg_bits),
    )


def uniform_bits(layer_dims, bits: int) -> AllocatorResult:
    L = len(layer_dims)
    return AllocatorResult([bits] * L, [float(bits)] * L, 0.0, float(bits), float(bits))
