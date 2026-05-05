"""Phase E — Channel-INT8 mixed-precision tests.

Acceptance criteria from the v2 plan:
  E1. κ_j = max_i (|W_ij^rot| · E|X_j|) per output channel; top-r %
      go INT8; rest INT4. Pack into a single bundle with a per-output-
      channel bit indicator.
  E2. Optional FP16 override for true super-weights (Yu et al.).
  E3. Bundle round-trips through pack/unpack; bit indicator survives
      serialisation; MLC-compile smoke is gated.
"""
from __future__ import annotations

import pickle

import pytest
import torch

from triad_ptq._v2.superweight.channel_int8 import (
    DEFAULT_SUPER_CHANNEL_RATE,
    ChannelInt8Bundle,
    channel_kappa,
    detect_true_super_weights,
    pack_channel_int8,
    select_super_channels,
    unpack_channel_int8,
)


# --------------------------------------------------------------------- κ + selector

def test_channel_kappa_shape_and_dtype() -> None:
    W = torch.randn(64, 128)
    e = torch.rand(128) + 0.1
    k = channel_kappa(W, e)
    assert k.shape == (64,)
    assert k.dtype == W.dtype
    assert torch.all(k >= 0)


def test_channel_kappa_rejects_shape_mismatch() -> None:
    W = torch.randn(8, 16)
    with pytest.raises(ValueError, match="must be"):
        channel_kappa(W, torch.rand(8))


def test_select_super_channels_returns_top_r_pct() -> None:
    out, in_ = 200, 64
    W = torch.randn(out, in_)
    e = torch.rand(in_) + 0.1
    idx = select_super_channels(W, e, rate=0.05)
    # 5% of 200 = 10 channels.
    assert idx.numel() == 10
    # Indices must be unique and ascending.
    assert torch.all(idx[:-1] < idx[1:])
    # The selected κ must be >= every non-selected κ.
    kappa = channel_kappa(W, e)
    sel = kappa[idx]
    nonsel = kappa[torch.tensor([i for i in range(out) if i not in idx.tolist()])]
    assert sel.min() >= nonsel.max() - 1e-6


def test_select_super_channels_min_count_floor() -> None:
    """For tiny output dims and small rate, min_count keeps at least one
    super channel (else channel-INT8 is pointless on the layer)."""
    W = torch.randn(8, 32)
    e = torch.rand(32) + 0.1
    idx = select_super_channels(W, e, rate=0.001, min_count=1)
    assert idx.numel() == 1


def test_default_rate_is_one_point_five_pct() -> None:
    assert DEFAULT_SUPER_CHANNEL_RATE == 0.015


# --------------------------------------------------------------------- pack / unpack

def test_pack_channel_int8_returns_bundle() -> None:
    W = torch.randn(64, 128)
    e = torch.rand(128) + 0.1
    b = pack_channel_int8(W, e, group_size=32, rate=0.05)
    assert isinstance(b, ChannelInt8Bundle)
    n_super = int(round(0.05 * 64))
    assert b.super_indices.numel() == n_super
    assert b.bit_indicator.sum().item() == n_super
    assert b.int4_weight.shape == (64 - n_super, 128)
    assert b.int8_weight.shape == (n_super, 128)
    assert b.int4_weight.dtype == torch.int8
    assert b.int8_weight.dtype == torch.int8
    assert b.int4_scale.dtype == torch.float16
    assert b.int8_scale.dtype == torch.float16


def test_pack_unpack_round_trip_preserves_shape_and_finiteness() -> None:
    torch.manual_seed(0)
    W = torch.randn(64, 128) * 0.2
    e = torch.rand(128) + 0.1
    b = pack_channel_int8(W, e, group_size=32, rate=0.03)
    W_rec = unpack_channel_int8(b)
    assert W_rec.shape == W.shape
    assert torch.isfinite(W_rec).all()


def test_int8_super_channels_have_smaller_recon_error_than_int4() -> None:
    """Super-channels are quantised at INT8; non-super at INT4. The MSE
    on the INT8 rows must be < the MSE on the INT4 rows (INT8 has 16×
    finer grid)."""
    torch.manual_seed(0)
    out, in_ = 200, 256
    W = torch.randn(out, in_) * 0.2
    e = torch.rand(in_) + 0.1
    b = pack_channel_int8(W, e, group_size=32, rate=0.05)
    W_rec = unpack_channel_int8(b)

    super_set = set(b.super_indices.tolist())
    super_rows = torch.tensor(sorted(super_set))
    int4_rows = torch.tensor([i for i in range(out) if i not in super_set])

    mse_int8 = (W_rec[super_rows] - W[super_rows]).pow(2).mean()
    mse_int4 = (W_rec[int4_rows] - W[int4_rows]).pow(2).mean()
    assert mse_int8 < mse_int4, f"INT8 MSE {mse_int8:.6f} ≥ INT4 MSE {mse_int4:.6f}"


def test_bit_indicator_round_trips_through_pickle() -> None:
    """The bit indicator (per-output-channel super flag) must survive
    serialisation. We use pickle as a stand-in for the safetensors round
    trip — both share dtype semantics for bool tensors.
    """
    W = torch.randn(32, 64)
    e = torch.rand(64) + 0.1
    b = pack_channel_int8(W, e, group_size=32, rate=0.0625)
    blob = pickle.dumps(b)
    b2 = pickle.loads(blob)
    assert torch.equal(b2.bit_indicator, b.bit_indicator)
    assert torch.equal(b2.super_indices, b.super_indices)
    assert torch.equal(b2.int4_weight, b.int4_weight)
    assert torch.equal(b2.int8_weight, b.int8_weight)
    assert torch.equal(b2.int4_scale, b.int4_scale)
    assert torch.equal(b2.int8_scale, b.int8_scale)


def test_pack_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        pack_channel_int8(torch.randn(8, 8, 8), torch.rand(8))


# --------------------------------------------------------------------- E2 detector

def test_detect_true_super_weights_finds_planted_outlier() -> None:
    """Plant a single weight whose ABSENCE inflates a target-tracking loss
    by >100×. The detector must find it and reject the others.

    Fixture: a Linear with a single nonzero weight [3, 5] = 100. The
    "loss" is the squared deviation of `model(x_unit_at_5)` from a
    target vector that places 100 at index 3. With the weight intact
    the loss is ~0; with the weight zeroed the loss is ~100².
    """
    model = torch.nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        model.weight.zero_()
        model.weight[3, 5] = 100.0  # the planted outlier

    target = torch.zeros(1, 8)
    target[0, 3] = 100.0
    x_unit = torch.zeros(1, 8)
    x_unit[0, 5] = 1.0  # picks out column 5 of the weight

    def loss_fn(m: torch.nn.Module) -> float:
        # baseline (with the weight) → tiny;  perturbed (zeroed) → ~10000.
        return float(((m(x_unit) - target).pow(2).sum() + 1e-6).item())

    candidates = [("weight", i, j) for i in range(8) for j in range(8)]
    detected = detect_true_super_weights(
        model, candidates, loss_fn, threshold=100.0
    )
    names_ij = [(n, i, j) for n, i, j, _ in detected]
    assert ("weight", 3, 5) in names_ij
    # No other weight should clear the threshold.
    assert all(t == ("weight", 3, 5) for t in names_ij)
