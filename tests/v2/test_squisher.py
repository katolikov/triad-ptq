"""Phase B — Squisher Fisher diagonal tests.

Acceptance criteria from the v2 plan:
  1. Squisher correlates Pearson ≥ 0.7 with Hutchinson on a 2-layer toy MLP.
  2. ρ ∈ [0.01, 100] for all blocks of TinyLlama-1.1B.

The TinyLlama acceptance gate is gated behind the `TRIAD_RUN_TINYLLAMA_TESTS=1`
environment variable: it requires downloading the HF model and ~1 GB of host
RAM. The toy-MLP test runs unconditionally on CPU.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from triad_ptq._v2.router.hutchinson_check import (
    correlate_squisher_vs_hutchinson,
    hutchinson_diagonal,
    pearson_correlation,
)
from triad_ptq._v2.router.squisher import (
    DEFAULT_GAMMA,
    SquisherAccumulator,
    derive_rho,
    squisher_fisher_diagonal,
)


# --------------------------------------------------------------------- accumulator

def test_accumulator_emas_squared_grads() -> None:
    torch.manual_seed(0)
    lin = nn.Linear(4, 4)
    accum = SquisherAccumulator(gamma=0.5)
    accum.init_from_module(lin)

    x = torch.randn(2, 4)
    y = lin(x).sum()
    y.backward()
    accum.observe(lin)

    g = lin.weight.grad
    expected = (1.0 - 0.5) * g.pow(2)
    assert torch.allclose(accum.state["weight"], expected, atol=1e-6)


def test_accumulator_skips_none_grad() -> None:
    lin = nn.Linear(3, 3)
    accum = SquisherAccumulator()
    accum.init_from_module(lin)
    # No backward call → grads are None.
    accum.observe(lin)
    # Buffers stay zero.
    assert torch.all(accum.state["weight"] == 0)
    assert accum.n_observed == 1


def test_accumulator_diagonal_returns_copy() -> None:
    lin = nn.Linear(2, 2)
    accum = SquisherAccumulator()
    accum.init_from_module(lin)
    d = accum.diagonal()
    d["weight"].fill_(99.0)
    # Modifying the returned copy must not affect the accumulator state.
    assert torch.all(accum.state["weight"] == 0)


# --------------------------------------------------------------------- runner

class _ToyMLP(nn.Module):
    """2-layer MLP with smooth nonlinearity for clean Fisher/Hessian behaviour."""

    def __init__(self, d_in: int = 8, d_hidden: int = 16, d_out: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_squisher_returns_per_param_dict() -> None:
    torch.manual_seed(0)
    block = _ToyMLP()
    X = torch.randn(32, 8)
    Y = torch.randn(32, 8)
    diag = squisher_fisher_diagonal(block, X, Y, n_steps=5, batch_size=4)
    expected_keys = {n for n, _ in block.named_parameters()}
    assert set(diag.keys()) == expected_keys
    for name, t in diag.items():
        assert t.shape == dict(block.named_parameters())[name].shape
        assert torch.all(t >= 0)  # squared grads are non-negative


def test_squisher_history_length_matches_n_steps() -> None:
    torch.manual_seed(0)
    block = _ToyMLP()
    X = torch.randn(16, 8)
    Y = torch.randn(16, 8)
    _, history = squisher_fisher_diagonal(
        block, X, Y, n_steps=7, batch_size=4, return_history=True
    )
    assert len(history) == 7


def test_squisher_default_gamma_is_zero_point_nine() -> None:
    assert DEFAULT_GAMMA == 0.9


# --------------------------------------------------------------------- ρ

def test_derive_rho_returns_finite_positive() -> None:
    torch.manual_seed(0)
    block = _ToyMLP()
    X = torch.randn(16, 8)
    Y = torch.randn(16, 8)
    rho = derive_rho(block, X, Y)
    assert rho > 0
    assert 1e-6 < rho < 1e6  # very loose; the toy-MLP scale is well-behaved


def test_derive_rho_scales_with_output_magnitude() -> None:
    """If we scale the targets up by k, ‖∂L/∂y‖² scales by k²; ‖∂L/∂x‖²
    scales by the same k² through the chain rule. ρ should be approximately
    invariant — verify it stays within an order of magnitude.
    """
    torch.manual_seed(0)
    block = _ToyMLP()
    X = torch.randn(16, 8)
    Y = torch.randn(16, 8)
    rho_1 = derive_rho(block, X, Y)
    rho_k = derive_rho(block, X, Y * 5.0)
    assert 0.1 < rho_k / rho_1 < 10.0


# --------------------------------------------------------------------- Hutchinson + correlation

def test_hutchinson_diagonal_shape_matches() -> None:
    torch.manual_seed(0)
    block = _ToyMLP()
    X = torch.randn(8, 8)
    Y = torch.randn(8, 8)

    def closure() -> torch.Tensor:
        return (block(X) - Y).pow(2).mean()

    diag = hutchinson_diagonal(block, closure, n_samples=5)
    for name, p in block.named_parameters():
        assert diag[name].shape == p.shape


def test_pearson_self_correlation_is_one() -> None:
    a = torch.randn(50)
    assert abs(pearson_correlation(a, a) - 1.0) < 1e-9


def test_pearson_anti_correlation_is_minus_one() -> None:
    a = torch.randn(50)
    assert abs(pearson_correlation(a, -a) - (-1.0)) < 1e-9


def test_squisher_hutchinson_correlation_on_toy_mlp() -> None:
    """Phase B acceptance criterion 1: Pearson ≥ 0.7 between Squisher and
    Hutchinson on a 2-layer toy MLP.

    Setup: BRECQ MSE loss, fixed calibration set, 200 Adam steps for the
    Squisher accumulator (we want the EMA to settle), 50 Hutchinson probes
    against the SAME loss + SAME parameter snapshot.
    """
    torch.manual_seed(0)
    block = _ToyMLP(d_in=12, d_hidden=24, d_out=12)

    # Fixed reference targets — what the FP16 model would output. Using the
    # untrained random init ensures the loss landscape has nontrivial
    # curvature in many directions (we are NOT comparing to a trivial zero
    # gradient setup).
    X = torch.randn(64, 12)
    with torch.no_grad():
        Y = block(X) + 0.3 * torch.randn(64, 12)  # mild target offset

    # 1. Take a SNAPSHOT of the parameters BEFORE Squisher training, since
    #    Hutchinson must run against the same θ that produced g_t² in the
    #    last few EMA observations. We compare on the snapshot post-training.
    squisher = squisher_fisher_diagonal(
        block, X, Y, n_steps=200, lr=1e-3, batch_size=8, gamma=0.9, seed=42
    )

    def closure() -> torch.Tensor:
        return (block(X) - Y).pow(2).mean()

    hutch = hutchinson_diagonal(block, closure, n_samples=50, seed=99)

    corrs = correlate_squisher_vs_hutchinson(squisher, hutch)
    overall = corrs["__overall__"]
    # Acceptance gate: ≥ 0.7. We expect ~0.8–0.95 for this setup; the gate
    # is loose to absorb seed variance.
    assert overall >= 0.7, (
        f"Squisher↔Hutchinson Pearson {overall:.3f} below 0.7 acceptance gate; "
        f"per-param: { {k: round(v, 3) for k, v in corrs.items() if k != '__overall__'} }"
    )


# --------------------------------------------------------------------- TinyLlama gate

@pytest.mark.skipif(
    os.environ.get("TRIAD_RUN_TINYLLAMA_TESTS") != "1",
    reason="TinyLlama-1.1B test gated; set TRIAD_RUN_TINYLLAMA_TESTS=1 to run",
)
def test_rho_in_range_on_tinyllama() -> None:
    """Phase B acceptance criterion 2: ρ ∈ [0.01, 100] for all blocks of
    TinyLlama-1.1B.

    Heavy: requires the HF cache + ~1 GB RAM + a few minutes on M1. Gated
    behind TRIAD_RUN_TINYLLAMA_TESTS=1 so CI doesn't pay the cost.
    """
    from transformers import AutoModelForCausalLM  # noqa: WPS433

    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32
    ).eval()

    d = model.config.hidden_size
    seq_len = 64
    batch = 2
    X = torch.randn(batch, seq_len, d) * 0.1
    # FP16-cascade target := the same forward through the unmodified block.
    out_of_range: dict[str, float] = {}
    for i, block in enumerate(model.model.layers):
        with torch.no_grad():
            Y = block(X)[0] if isinstance(block(X), tuple) else block(X)
        try:
            rho = derive_rho(block, X, Y if isinstance(Y, torch.Tensor) else Y[0])
        except Exception as exc:  # pragma: no cover — surfaces in the gated run
            pytest.fail(f"derive_rho failed on block {i}: {exc}")
        if not (0.01 <= rho <= 100.0):
            out_of_range[f"layer_{i}"] = rho

    assert not out_of_range, f"ρ out-of-range blocks: {out_of_range}"
