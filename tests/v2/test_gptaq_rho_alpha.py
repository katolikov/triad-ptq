"""Phase F — GPTAQ ρ-weighted α tests.

Acceptance criteria from the v2 plan:
  F1. α = min(0.8, sigmoid(c · log ρ)).
  F2. Scope-limit (exclude o_proj, down_proj) preserved.
  F3. Per-block α monitor written to results/tables/v2_gptaq_alpha.json.
  F4. α stays in [0, 0.8]; fixed-α=0.5 ablation reproduces v1 numbers
      within 0.05 PPL on SmolLM-135M (the actual SmolLM measurement is
      Phase H's job — here we verify only the function-level contract).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from triad_ptq._v2.calib.gptaq_rho_alpha import (
    DEFAULT_ALPHA_MAX,
    DEFAULT_C,
    DEFAULT_EXCLUDE_SUFFIXES,
    alpha_from_rho,
    alpha_schedule,
    write_alpha_log,
)


# --------------------------------------------------------------------- alpha_from_rho

def test_alpha_at_rho_one_equals_half() -> None:
    assert abs(alpha_from_rho(1.0) - 0.5) < 1e-12


def test_alpha_clamped_at_alpha_max() -> None:
    # As ρ → ∞, sigmoid(log ρ) → 1, so α saturates at α_max.
    assert alpha_from_rho(1e10) == pytest.approx(DEFAULT_ALPHA_MAX, abs=1e-9)


def test_alpha_zero_at_rho_zero() -> None:
    assert alpha_from_rho(0.0) == 0.0


def test_alpha_monotonic_in_rho() -> None:
    xs = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]
    ys = [alpha_from_rho(x) for x in xs]
    assert ys == sorted(ys)


def test_alpha_in_range() -> None:
    for rho in [0.0001, 0.1, 1.0, 100.0, 1e6]:
        a = alpha_from_rho(rho)
        assert 0.0 <= a <= DEFAULT_ALPHA_MAX, f"α={a} out of [0, {DEFAULT_ALPHA_MAX}] at ρ={rho}"


def test_alpha_rejects_non_finite_rho() -> None:
    with pytest.raises(ValueError, match="finite"):
        alpha_from_rho(float("inf"))
    with pytest.raises(ValueError, match="finite"):
        alpha_from_rho(float("nan"))


def test_alpha_max_validated() -> None:
    with pytest.raises(ValueError, match="alpha_max"):
        alpha_from_rho(1.0, alpha_max=0.0)
    with pytest.raises(ValueError, match="alpha_max"):
        alpha_from_rho(1.0, alpha_max=1.5)


def test_alpha_c_scales_log_steepness() -> None:
    """Larger c steepens the sigmoid; α(ρ=2) should grow with c."""
    a1 = alpha_from_rho(2.0, c=1.0)
    a4 = alpha_from_rho(2.0, c=4.0)
    assert a4 > a1


# --------------------------------------------------------------------- schedule

def test_alpha_schedule_excludes_o_proj_and_down_proj() -> None:
    rhos = {
        "layers.0.self_attn.q_proj":  2.0,
        "layers.0.self_attn.o_proj":  3.0,
        "layers.0.mlp.gate_proj":     1.5,
        "layers.0.mlp.down_proj":     5.0,
    }
    sched = alpha_schedule(rhos)
    assert sched["layers.0.self_attn.o_proj"] == 0.0
    assert sched["layers.0.mlp.down_proj"] == 0.0
    assert sched["layers.0.self_attn.q_proj"] > 0.0
    assert sched["layers.0.mlp.gate_proj"] > 0.0


def test_alpha_schedule_with_custom_exclude() -> None:
    rhos = {"my.weird.layer.x_proj": 2.0, "my.weird.layer.y_proj": 0.5}
    sched = alpha_schedule(rhos, exclude_suffixes=("y_proj",))
    assert sched["my.weird.layer.y_proj"] == 0.0
    assert sched["my.weird.layer.x_proj"] > 0.0


def test_default_exclude_matches_v1_adr_010() -> None:
    assert DEFAULT_EXCLUDE_SUFFIXES == ("o_proj", "down_proj")


def test_default_c_one() -> None:
    assert DEFAULT_C == 1.0
    assert DEFAULT_ALPHA_MAX == 0.8


# --------------------------------------------------------------------- v1 reproduction

def test_fixed_alpha_half_matches_v1_within_tolerance() -> None:
    """When ρ ≈ 1 across all blocks, α^(ℓ) ≈ 0.5 — the v1 default. The
    F4 acceptance ("fixed-α=0.5 ablation reproduces v1 numbers within
    0.05 PPL") is a Phase-H measurement, but the *contract* is that the
    α schedule reduces to v1 in the limit. We assert the limit here.
    """
    rhos = {f"layers.{i}.self_attn.q_proj": 1.0 for i in range(20)}
    sched = alpha_schedule(rhos)
    for v in sched.values():
        assert abs(v - 0.5) < 1e-12


# --------------------------------------------------------------------- log writer

def test_write_alpha_log_emits_v2_schema(tmp_path: Path) -> None:
    rhos = {
        "layers.0.self_attn.q_proj": 2.0,
        "layers.0.self_attn.o_proj": 3.0,
        "layers.0.mlp.gate_proj":    0.4,
        "layers.0.mlp.down_proj":    5.0,
    }
    alphas = alpha_schedule(rhos)
    out = write_alpha_log(rhos, alphas, tmp_path / "v2_gptaq_alpha.json")
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["schema"] == "v2_gptaq_alpha/1"
    assert payload["n_blocks"] == 4
    assert payload["n_excluded"] == 2
    # Excluded blocks don't pollute the alpha summary stats.
    assert payload["alpha_min"] is not None
    assert payload["alpha_max_obs"] is not None
    assert payload["alpha_mean"] is not None
    # Every entry has the four required fields.
    for e in payload["entries"]:
        assert set(e.keys()) == {"block", "rho", "alpha", "excluded"}


def test_write_alpha_log_handles_all_excluded(tmp_path: Path) -> None:
    rhos = {"foo.o_proj": 1.5, "foo.down_proj": 2.0}
    out = write_alpha_log(rhos, alpha_schedule(rhos), tmp_path / "z.json")
    payload = json.loads(out.read_text())
    assert payload["n_excluded"] == 2
    assert payload["alpha_min"] is None
    assert payload["alpha_mean"] is None
