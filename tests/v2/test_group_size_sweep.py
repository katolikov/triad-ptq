"""Phase G — Group-size sweep harness tests.

Acceptance criteria from the v2 plan:
  G1. Harness runs the v2 calibration pipeline at G ∈ {32, 64, 128} and
      emits one packed-weight result per G with metadata JSON. (Real
      pipeline dispatch is Phase H runbook; here we test the harness
      contract on a stub callable.)
  G2. Default G recommendation is decoded by `decide_default_group_size`
      and is gated on a measured ON-DEVICE decode_tps; missing data
      raises rather than guessing.
  G4. ADR-015 documents the default-G choice once Mali measurement runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from triad_ptq._v2.groupsize.sweep import (
    GROUP_SIZES_TO_SWEEP,
    RECOMMENDED_DEFAULT_G,
    SweepResult,
    SweepRow,
    decide_default_group_size,
    estimate_disk_mb,
    run_group_size_sweep,
    write_sweep_result,
)


# --------------------------------------------------------------------- estimator

def test_estimate_disk_mb_g64_smaller_than_g32() -> None:
    """G=64 stores half as many fp16 scales per row → smaller bundle.

    With out=2048, in=2048, this is the TinyLlama down-proj approximation.
    """
    out_features, in_features = 2048, 2048
    n_w = out_features * in_features
    mb_g32 = estimate_disk_mb(
        n_weights_total=n_w, super_channel_rate=0.015,
        group_size=32, out_features=out_features,
    )
    mb_g64 = estimate_disk_mb(
        n_weights_total=n_w, super_channel_rate=0.015,
        group_size=64, out_features=out_features,
    )
    mb_g128 = estimate_disk_mb(
        n_weights_total=n_w, super_channel_rate=0.015,
        group_size=128, out_features=out_features,
    )
    assert mb_g64 < mb_g32
    assert mb_g128 < mb_g64
    # Falsification gate H4 #5: G=64 ≤ 0.92 × G=32. Verify our estimator
    # would have a fighting chance — the ratio depends on how many
    # weights vs scales there are; for d=2048 the scale fraction is small.
    ratio = mb_g64 / mb_g32
    assert ratio < 1.0


def test_estimate_disk_mb_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="super_channel_rate"):
        estimate_disk_mb(
            n_weights_total=1024, super_channel_rate=1.5,
            group_size=32, out_features=32,
        )
    with pytest.raises(ValueError, match="positive"):
        estimate_disk_mb(
            n_weights_total=1024, super_channel_rate=0.01,
            group_size=0, out_features=32,
        )
    with pytest.raises(ValueError, match="must equal"):
        estimate_disk_mb(
            n_weights_total=1023, super_channel_rate=0.01,
            group_size=32, out_features=32,  # 32×32=1024, not 1023
        )
    with pytest.raises(ValueError, match="divisible"):
        # in_features = 48; not divisible by group_size=32.
        estimate_disk_mb(
            n_weights_total=48 * 64, super_channel_rate=0.01,
            group_size=32, out_features=64,
        )


def test_estimate_disk_mb_known_constants() -> None:
    """Hand-computable case: out=8, in=64, G=32, super_rate=0.125 (1 super
    channel out of 8). Expected breakdown:
      INT4 main:   7 * 64 * 0.5 = 224 bytes
      INT8 super:  1 * 64 * 1   =  64 bytes
      fp16 scales: 7 * 2 * 2    =  28 bytes  (n_groups = 64/32 = 2)
      fp16 super:  1 * 2 * 2    =   4 bytes
      indicator:   ceil(8/8)    =   1 byte
      Total:                       321 bytes
    """
    mb = estimate_disk_mb(
        n_weights_total=8 * 64, super_channel_rate=0.125,
        group_size=32, out_features=8,
    )
    expected_bytes = 7 * 64 * 0.5 + 1 * 64 + 7 * 2 * 2 + 1 * 2 * 2 + 1
    assert mb == pytest.approx(expected_bytes / (1024 * 1024), rel=1e-9)


# --------------------------------------------------------------------- harness

def test_run_group_size_sweep_aggregates_rows() -> None:
    def calibrate(g: int) -> SweepRow:
        return SweepRow(group_size=g, wt2_ppl=10.0 - 0.1 * g, disk_mb=100.0 / g)

    res = run_group_size_sweep("dummy/model", calibrate)
    assert isinstance(res, SweepResult)
    assert [r.group_size for r in res.rows] == list(GROUP_SIZES_TO_SWEEP)
    assert res.best_by_ppl() == max(GROUP_SIZES_TO_SWEEP)


def test_run_group_size_sweep_writes_json(tmp_path: Path) -> None:
    def calibrate(g: int) -> SweepRow:
        return SweepRow(group_size=g, wt2_ppl=10.0, disk_mb=50.0)

    out = tmp_path / "sweep.json"
    res = run_group_size_sweep("dummy", calibrate, output_path=out)
    payload = json.loads(out.read_text())
    assert payload["schema"] == "v2_group_size_sweep/1"
    assert payload["model_id"] == "dummy"
    assert len(payload["rows"]) == len(GROUP_SIZES_TO_SWEEP)


def test_run_group_size_sweep_rejects_mismatched_g() -> None:
    def bad(_g: int) -> SweepRow:
        return SweepRow(group_size=999)

    with pytest.raises(ValueError, match="group_size=999"):
        run_group_size_sweep("dummy", bad)


# --------------------------------------------------------------------- decision

def test_decide_default_g_picks_64_when_decode_at_least_g32() -> None:
    res = SweepResult(
        model_id="m",
        rows=[
            SweepRow(group_size=32, decode_tps=35.0),
            SweepRow(group_size=64, decode_tps=36.0),
            SweepRow(group_size=128, decode_tps=37.0),
        ],
    )
    assert decide_default_group_size(res) == 64


def test_decide_default_g_reverts_to_32_when_g64_slower() -> None:
    res = SweepResult(
        model_id="m",
        rows=[
            SweepRow(group_size=32, decode_tps=35.0),
            SweepRow(group_size=64, decode_tps=33.5),  # SLOWER
        ],
    )
    assert decide_default_group_size(res) == 32


def test_decide_default_g_requires_measurements() -> None:
    res = SweepResult(
        model_id="m",
        rows=[
            SweepRow(group_size=32, decode_tps=None),
            SweepRow(group_size=64, decode_tps=None),
        ],
    )
    with pytest.raises(ValueError, match="measurement on the target device"):
        decide_default_group_size(res)


def test_decide_default_g_requires_both_g32_g64() -> None:
    res = SweepResult(
        model_id="m",
        rows=[
            SweepRow(group_size=32, decode_tps=35.0),
            SweepRow(group_size=128, decode_tps=37.0),
        ],
    )
    with pytest.raises(ValueError, match="requires both"):
        decide_default_group_size(res)


def test_default_constants() -> None:
    assert GROUP_SIZES_TO_SWEEP == (32, 64, 128)
    assert RECOMMENDED_DEFAULT_G == 64  # candidate; final default decided by decide_default_group_size
