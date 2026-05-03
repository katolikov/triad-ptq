"""Tests for the bit allocator (equation 2 of TRIAD-PTQ)."""
from __future__ import annotations

import numpy as np
import pytest

from triad_ptq.core.allocator import allocate_bits, uniform_bits


def test_target_budget_within_5pct() -> None:
    rng = np.random.default_rng(0)
    L = 30
    sens = rng.lognormal(mean=0.0, sigma=2.0, size=L).tolist()
    dims = [int(rng.integers(64 * 64, 4096 * 4096)) for _ in range(L)]
    target = 4.0

    res = allocate_bits(sens, dims, target_avg_bits=target)
    diff = abs(res.achieved_bits_per_w - target) / target
    assert diff <= 0.05, (
        f"achieved {res.achieved_bits_per_w:.3f} bits/w vs target {target}, diff={diff:.3%}"
    )


def test_higher_sensitivity_gets_more_bits() -> None:
    """Layers in the top sensitivity quartile should average >= bottom quartile."""
    rng = np.random.default_rng(7)
    L = 60
    sens = rng.lognormal(mean=0.0, sigma=2.5, size=L)
    dims = [4096 * 4096] * L  # equal-size layers

    res = allocate_bits(sens.tolist(), dims, target_avg_bits=4.0)
    bits = np.asarray(res.bits)
    sens_arr = np.asarray(sens)

    # top vs bottom quartile by sensitivity
    q1 = np.quantile(sens_arr, 0.75)
    q0 = np.quantile(sens_arr, 0.25)
    top = bits[sens_arr >= q1].mean()
    bot = bits[sens_arr <= q0].mean()
    assert top >= bot, f"top quartile bits {top:.2f} should be >= bottom {bot:.2f}"


def test_target_3_bits() -> None:
    rng = np.random.default_rng(42)
    L = 12
    sens = rng.lognormal(0, 1.5, size=L).tolist()
    dims = [int(rng.integers(1e5, 1e7)) for _ in range(L)]
    res = allocate_bits(sens, dims, target_avg_bits=3.0)
    assert abs(res.achieved_bits_per_w - 3.0) / 3.0 <= 0.10


def test_uniform_alloc() -> None:
    res = uniform_bits([1, 2, 3], 4)
    assert res.bits == [4, 4, 4]
    assert res.achieved_bits_per_w == 4.0


@pytest.mark.parametrize("target", [3.0, 4.0, 5.0, 6.0])
def test_various_targets(target: float) -> None:
    rng = np.random.default_rng(int(target * 17))
    L = 40
    sens = rng.lognormal(0, 2.0, size=L).tolist()
    dims = [int(rng.integers(1e5, 1e7)) for _ in range(L)]
    res = allocate_bits(sens, dims, target_avg_bits=target)
    # Snapping to {3,4,8} only is coarse; allow up to 12% on extreme targets.
    assert abs(res.achieved_bits_per_w - target) / target <= 0.12
