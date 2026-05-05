"""Phase A — plumbing and safety-net tests.

These check ONLY that the v2 scaffolding is wired correctly. No algorithm
behaviour is exercised here (Phases B–H land their own tests).
"""
from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn as nn

from triad_ptq import api as triad_api
from triad_ptq.utils.device import safe_cholesky_inverse


# --------------------------------------------------------------------- A2

V2_PACKAGES = [
    "triad_ptq._v2",
    "triad_ptq._v2.rotation",
    "triad_ptq._v2.transform",
    "triad_ptq._v2.router",
    "triad_ptq._v2.superweight",
    "triad_ptq._v2.lwc",
    "triad_ptq._v2.groupsize",
]
V2_LEAF_MODULES = [
    "triad_ptq._v2.rotation.sign_perm",
    "triad_ptq._v2.transform.learnable_beta",
    "triad_ptq._v2.router.squisher",
    "triad_ptq._v2.superweight.channel_int8",
    "triad_ptq._v2.lwc.selective",
    "triad_ptq._v2.groupsize.sweep",
]


@pytest.mark.parametrize("modname", V2_PACKAGES + V2_LEAF_MODULES)
def test_v2_skeleton_imports(modname: str) -> None:
    importlib.import_module(modname)


@pytest.mark.parametrize("modname", V2_LEAF_MODULES)
def test_v2_leaf_module_marked_unimplemented(modname: str) -> None:
    mod = importlib.import_module(modname)
    assert hasattr(mod, "IMPLEMENTED"), f"{modname} missing IMPLEMENTED flag"
    assert mod.IMPLEMENTED is False


def test_v2_leaf_modules_raise_not_implemented() -> None:
    from triad_ptq._v2.rotation import sign_perm
    from triad_ptq._v2.transform import learnable_beta
    from triad_ptq._v2.router import squisher

    with pytest.raises(NotImplementedError):
        sign_perm.apply_sign_perm_to_llama()
    with pytest.raises(NotImplementedError):
        learnable_beta.train_learnable_beta()
    with pytest.raises(NotImplementedError):
        squisher.squisher_fisher_diagonal()


# --------------------------------------------------------------------- A3

def test_optimize_unknown_algorithm_raises() -> None:
    model = nn.Linear(8, 8)
    with pytest.raises(ValueError, match="unknown algorithm"):
        triad_api.optimize(model, algorithm="v3")


def test_optimize_v2_raises_not_implemented() -> None:
    model = nn.Linear(8, 8)
    with pytest.raises(NotImplementedError, match="SPECTRA-Q"):
        triad_api.optimize(model, algorithm="v2", calibration=[])


def test_optimize_v1_default_unaffected_by_v2_kwargs() -> None:
    """Passing v2-only kwargs at algorithm='v1' should NOT route to compile_model.

    We only verify the kwarg surface — compile_model itself is exercised by
    the existing v1 end-to-end tests.
    """
    sig = triad_api.optimize.__annotations__
    # The new keyword surface must include the v2 knobs even when defaulting
    # to v1, so external callers can stage v2 configs ahead of time.
    import inspect
    params = inspect.signature(triad_api.optimize).parameters
    for name in ("algorithm", "rotation", "super_channel_rate",
                 "gptaq_alpha_c", "lwc_threshold_percentile"):
        assert name in params, f"optimize() missing v2 kwarg {name!r}"
    assert params["algorithm"].default == "v1"


# --------------------------------------------------------------------- A6

def test_safe_cholesky_inverse_round_trip_cpu() -> None:
    torch.manual_seed(0)
    d = 64
    X = torch.randn(256, d, dtype=torch.float32)
    H = X.t() @ X
    H_inv = safe_cholesky_inverse(H, percdamp=0.0, cpu_threshold_dim=None)
    eye = H @ H_inv
    err = (eye - torch.eye(d)).norm() / d ** 0.5
    assert err < 1e-3, f"H · H^-1 should be ~I (err={err:.2e})"


def test_safe_cholesky_inverse_dtype_device_preserved() -> None:
    torch.manual_seed(0)
    H = torch.randn(32, 16) @ torch.randn(16, 32)
    H = H @ H.t() + torch.eye(32)  # PSD
    H = H.to(torch.float32)
    out = safe_cholesky_inverse(H, percdamp=0.01)
    assert out.dtype == H.dtype
    assert out.device == H.device
    assert out.shape == H.shape


def test_safe_cholesky_inverse_ridge_rescues_near_singular() -> None:
    """A nearly-rank-deficient PSD matrix becomes invertible under percdamp."""
    torch.manual_seed(1)
    # Rank-1 PSD: H = v vᵀ.
    v = torch.randn(8, 1, dtype=torch.float32)
    H = v @ v.t() + 1e-6 * torch.eye(8)  # add a tiny floor so mean(diag) > 0
    out = safe_cholesky_inverse(H, percdamp=0.1)
    assert torch.isfinite(out).all()


def test_safe_cholesky_inverse_handles_diag_only() -> None:
    H = torch.diag(torch.tensor([1.0, 2.0, 4.0, 8.0]))
    out = safe_cholesky_inverse(H, percdamp=0.0, cpu_threshold_dim=None)
    expected = torch.diag(torch.tensor([1.0, 0.5, 0.25, 0.125]))
    assert torch.allclose(out, expected, atol=1e-5)
