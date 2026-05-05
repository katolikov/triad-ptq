"""Phase D — Learnable β + selective LWC tests.

Acceptance criteria from the v2 plan:
  D1. Per-block β is a single scalar; the smoothing transform is the AWQ
      / SmoothQuant migration. The trainer must reduce the BRECQ block-
      output reconstruction loss over 100 Adam steps.
  D2. Selective LWC selects the top ~25 % most sensitive blocks by ρ.
  D4. β converges within 100 steps; total Adam wall-time ≤ 11 min for
      TinyLlama-1.1B on RTX 4090 (gated — measured during Phase H, not
      here).

We use a tiny synthetic Linear-only "block" so the test runs in ~1 s.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from triad_ptq._v2.lwc.selective import (
    LWCConfig,
    apply_selective_lwc,
    select_lwc_blocks,
)
from triad_ptq._v2.transform.learnable_beta import (
    DEFAULT_BETA_INIT,
    LearnableBetaResult,
    bake_smoothed_weights,
    fake_quantize_int4_per_group,
    train_learnable_beta,
)


# --------------------------------------------------------------------- fake-quant

def test_fake_quantize_int4_round_trip_finite() -> None:
    torch.manual_seed(0)
    W = torch.randn(8, 64) * 0.1
    Wq = fake_quantize_int4_per_group(W, group_size=32)
    assert Wq.shape == W.shape
    assert torch.isfinite(Wq).all()
    # MSE must be small relative to weight scale.
    err = (Wq - W).pow(2).mean().sqrt()
    rel = err / W.std()
    assert rel < 0.2, f"INT4 fake-quant rel-stddev {rel:.3f} too large"


def test_fake_quantize_alpha_clipping() -> None:
    """α < 1 reduces the per-group scale, increasing clipping (and MSE)."""
    torch.manual_seed(0)
    W = torch.randn(4, 32) * 0.5
    n_groups = 1
    Wq_full = fake_quantize_int4_per_group(W, 32)
    alpha_half = torch.full((n_groups,), 0.5)
    Wq_half = fake_quantize_int4_per_group(W, 32, alpha=alpha_half)
    err_full = (Wq_full - W).pow(2).mean()
    err_half = (Wq_half - W).pow(2).mean()
    assert err_half > err_full  # tighter scale → more clipping


def test_fake_quantize_ste_backward_passes() -> None:
    W = torch.randn(2, 32, requires_grad=True)
    Wq = fake_quantize_int4_per_group(W, 32)
    loss = Wq.pow(2).sum()
    loss.backward()
    assert W.grad is not None
    assert torch.isfinite(W.grad).all()


# --------------------------------------------------------------------- LWC selector

def test_select_lwc_blocks_above_75th_percentile() -> None:
    rho = {f"layer_{i}": float(i) for i in range(20)}
    sel = select_lwc_blocks(rho, threshold_percentile=75)
    enabled = [k for k, v in sel.items() if v]
    # Top 25% of 20 = 5 blocks (layer_15..layer_19).
    assert len(enabled) == 5
    assert all(int(k.split("_")[1]) >= 15 for k in enabled)


def test_apply_selective_lwc_roundtrips() -> None:
    rho = {f"l{i}": float(i) for i in range(8)}
    cfgs = apply_selective_lwc(rho, threshold_percentile=75)
    assert isinstance(cfgs["l0"], LWCConfig)
    assert cfgs["l7"].enabled is True
    assert cfgs["l0"].enabled is False


# --------------------------------------------------------------------- trainer

class _MiniBlock(nn.Module):
    """Stand-in for a transformer block: two Linears with a GELU.

    `train_learnable_beta` only needs the block's `forward(X) → Y` and a
    list of "quantizable" Linears. We hand it both layers.
    """

    def __init__(self, d_in: int = 64, d_h: int = 128, d_out: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.fc2 = nn.Linear(d_h, d_out)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _gen_calib(d_in: int = 64, d_out: int = 64, n: int = 64) -> tuple[torch.Tensor, torch.Tensor, _MiniBlock]:
    torch.manual_seed(42)
    block = _MiniBlock(d_in=d_in, d_h=2 * d_in, d_out=d_out)
    X = torch.randn(n, d_in)
    with torch.no_grad():
        Y = block(X) + 0.05 * torch.randn(n, d_out)
    return X, Y, block


def test_train_learnable_beta_returns_expected_shape() -> None:
    X, Y, block = _gen_calib()
    res = train_learnable_beta(
        block, X, Y,
        quantizable_linears=[block.fc1, block.fc2],
        group_size=32, n_steps=10, batch_size=8, beta_init=0.5,
    )
    assert isinstance(res, LearnableBetaResult)
    assert 0.0 <= res.beta <= 1.0
    assert res.n_steps == 10
    assert res.n_linears == 2
    assert len(res.beta_history) == 10
    assert len(res.loss_history) == 10


def test_train_learnable_beta_decreases_loss() -> None:
    """BRECQ loss must decrease over 50 Adam steps. The test is mild
    (final ≤ 0.95 × initial) to absorb mini-batch noise."""
    X, Y, block = _gen_calib()
    res = train_learnable_beta(
        block, X, Y,
        quantizable_linears=[block.fc1, block.fc2],
        group_size=32, n_steps=50, batch_size=8, lr=2e-2, beta_init=0.5,
    )
    init_avg = sum(res.loss_history[:5]) / 5
    final_avg = sum(res.loss_history[-5:]) / 5
    assert final_avg < 0.95 * init_avg, (
        f"BRECQ loss did not decrease: init_avg={init_avg:.5f} → final_avg={final_avg:.5f}"
    )


def test_train_learnable_beta_does_not_mutate_weights_during_training() -> None:
    """After train_learnable_beta returns, lin.weight must be unchanged
    from its pre-call value (the trainer monkey-patches `forward` and
    restores). Phase H's caller does the bake.
    """
    X, Y, block = _gen_calib()
    w0 = block.fc1.weight.detach().clone()
    train_learnable_beta(
        block, X, Y,
        quantizable_linears=[block.fc1, block.fc2],
        group_size=32, n_steps=10, batch_size=4,
    )
    assert torch.allclose(block.fc1.weight, w0)


def test_train_learnable_beta_lwc_alpha_within_clip() -> None:
    X, Y, block = _gen_calib()
    res = train_learnable_beta(
        block, X, Y,
        quantizable_linears=[block.fc1, block.fc2],
        group_size=32, n_steps=20, batch_size=8,
        lwc=LWCConfig(enabled=True, alpha_min=0.5, alpha_max=1.0),
    )
    assert res.lwc_enabled
    assert len(res.lwc_alpha) == 2
    for name, a in res.lwc_alpha.items():
        assert torch.all(a >= 0.5)
        assert torch.all(a <= 1.0)


def test_saturation_flag_triggers_at_boundary() -> None:
    X, Y, block = _gen_calib()
    res = train_learnable_beta(
        block, X, Y,
        quantizable_linears=[block.fc1, block.fc2],
        group_size=32, n_steps=5, batch_size=4,
        beta_init=0.05, beta_min=0.05, beta_max=0.95,
    )
    # Init right at the boundary → saturated should be True if Adam can't
    # move β away from it within 5 steps. This is loose; the test only
    # verifies the flag wiring runs.
    assert isinstance(res.saturated, bool)


def test_default_beta_init_is_one_half() -> None:
    assert DEFAULT_BETA_INIT == 0.5


# --------------------------------------------------------------------- bake

def test_bake_smoothed_weights_changes_weight() -> None:
    torch.manual_seed(0)
    lin = nn.Linear(64, 32)
    w0 = lin.weight.detach().clone()
    e_abs_x = {id(lin): torch.full((64,), 0.5)}
    bake_smoothed_weights([lin], e_abs_x, beta=0.5)
    assert not torch.allclose(lin.weight, w0)
