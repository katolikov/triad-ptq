"""Unit tests for the GPTAQ asymmetric weight transfer (Phase-2).

We verify three properties of `asymmetric_transfer`:

1.  **Identity-cascade reduces to identity**. If the post-quant cascade
    input X equals the FP16 reference X̃, then C = H̃ = H and W_aug = W.

2.  **Closed-form correctness on a random toy**. The transfer is the
    closed-form continuous optimum of  E[‖X̃ Wᵀ − X W_q ᵀ‖²_F]; we
    check that ‖X̃ Wᵀ − X W_augᵀ‖²  ≤  ‖X̃ Wᵀ − X Wᵀ‖²  by a clean
    margin (not a near-tie).

3.  **TRIAD basis-commute**. Applying the transfer in the original basis
    and *then* TRIAD's basis change W' = W·U·Λ^β yields the same result
    (up to rounding) as applying TRIAD first and then the basis-aware
    transfer. This is the property we relied on in compile_model.

We do NOT exercise compile_model end-to-end here — that is the smoke
test in `experiments/18_gptaq_smoke_smollm.py`.
"""
from __future__ import annotations

import math

import torch

from triad_ptq.core.gptaq_asym import GptaqStats, asymmetric_transfer


def _make_random_problem(d_in=32, d_out=24, T=512, seed=0, eps=0.0):
    """Return W (d_out, d_in), X̃ (T, d_in), X = X̃ + eps·N(0,1)."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(d_out, d_in, generator=g, dtype=torch.float32)
    X_pre = torch.randn(T, d_in, generator=g, dtype=torch.float32)
    if eps > 0.0:
        noise = torch.randn(T, d_in, generator=g, dtype=torch.float32)
        X_post = X_pre + eps * noise
    else:
        X_post = X_pre.clone()
    return W, X_pre, X_post


def _grams(X_pre, X_post):
    T = X_post.size(0)
    H = (X_post.t() @ X_post) / T
    C = (X_pre.t() @ X_post) / T
    return GptaqStats(H_post=H, C=C, n_tokens=T)


# ----- 1. Identity cascade reduces to identity -----------------------------

def test_identity_cascade_returns_input_weight():
    W, X_pre, X_post = _make_random_problem(eps=0.0)
    stats = _grams(X_pre, X_post)
    W_aug = asymmetric_transfer(W, stats, percdamp=1e-6)
    # With X = X̃ exactly, the closed form is W (and the damping is the
    # only deviation). Check Frobenius error is ≤ 1e-3 of ‖W‖.
    rel = (W_aug - W).norm() / W.norm().clamp_min(1e-12)
    assert rel.item() < 1e-3, f"identity-cascade transfer drifted: {rel.item():.2e}"


# ----- 2. Closed-form objective decreases ----------------------------------

def test_transfer_decreases_asymmetric_loss_iid_noise():
    """For X = X̃ + ε·noise (i.i.d.) the closed-form gives W_aug ≈ W / (1+ε²).
    Loss reduction is small but deterministic and must be strict."""
    W, X_pre, X_post = _make_random_problem(eps=0.3, seed=11)
    stats = _grams(X_pre, X_post)
    W_aug = asymmetric_transfer(W, stats, percdamp=1e-4)

    Y_pre = X_pre @ W.t()
    loss_W = (Y_pre - X_post @ W.t()).pow(2).mean().item()
    loss_aug = (Y_pre - X_post @ W_aug.t()).pow(2).mean().item()

    assert loss_aug < loss_W, (
        f"asymmetric transfer must decrease loss: "
        f"loss_W={loss_W:.4e}, loss_aug={loss_aug:.4e}"
    )
    # The analytic prediction for i.i.d. noise: loss_aug ≈ loss_W · (ε²/(1+ε²)) ratio
    # is shape-dependent; just require ≥1% reduction (well above fp32 noise).
    rel = (loss_W - loss_aug) / loss_W
    assert rel > 0.01, f"loss decrease below noise floor: rel={rel:.3e}"


def test_transfer_matches_explicit_closed_form_with_asymmetric_C():
    """Catches the C-vs-Cᵀ transpose error: when C = X̃ᵀX is highly
    non-symmetric (X is a non-symmetric mixing of X̃) the wrong transpose
    diverges from the analytic optimum by orders of magnitude."""
    g = torch.Generator().manual_seed(123)
    d_in, d_out, T = 16, 12, 256
    W = torch.randn(d_out, d_in, generator=g, dtype=torch.float32)
    X_pre = torch.randn(T, d_in, generator=g, dtype=torch.float32)
    # Strongly non-symmetric mixing: lower-triangular shift.
    M = torch.tril(torch.randn(d_in, d_in, generator=g, dtype=torch.float32))
    X_post = X_pre @ M

    stats = _grams(X_pre, X_post)
    W_aug = asymmetric_transfer(W, stats, percdamp=1e-6)

    # Analytic closed form W (X̃ᵀX) (XᵀX)⁻¹ from raw activations:
    H = (X_post.t() @ X_post) / T
    C = (X_pre.t() @ X_post) / T
    H_reg = H + 1e-6 * H.diagonal().mean() * torch.eye(d_in)
    W_analytic = W @ C @ torch.linalg.inv(H_reg)

    rel = (W_aug - W_analytic).norm() / W_analytic.norm().clamp_min(1e-12)
    # Tolerance is loose enough to absorb the percdamp ridge but tight enough
    # to catch the C-vs-Cᵀ transpose (wrong-transpose check below verifies the
    # gap is far larger than this tolerance).
    assert rel.item() < 5e-3, (
        f"asymmetric_transfer diverges from analytic closed form: rel={rel.item():.3e}"
    )

    # The wrong transpose (W · Cᵀ · H⁻¹) is meaningfully different on this
    # problem; this is the regression contract.
    W_wrong = W @ C.t() @ torch.linalg.inv(H_reg)
    rel_wrong = (W_aug - W_wrong).norm() / W_wrong.norm().clamp_min(1e-12)
    assert rel_wrong.item() > 0.05, (
        f"test problem is not asymmetric enough: wrong-transpose rel={rel_wrong.item():.3e}"
    )


def test_transfer_decreases_asymmetric_loss_attenuation():
    """Stronger asymmetry: X = α·X̃ + ε·noise with α=0.7 (multiplicative
    attenuation, like quant-cascade output shrinking). Closed form
    should give a substantial (~30%) loss reduction here."""
    g = torch.Generator().manual_seed(99)
    d_in, d_out, T = 32, 24, 1024
    W = torch.randn(d_out, d_in, generator=g, dtype=torch.float32)
    X_pre = torch.randn(T, d_in, generator=g, dtype=torch.float32)
    noise = torch.randn(T, d_in, generator=g, dtype=torch.float32)
    X_post = 0.7 * X_pre + 0.1 * noise

    stats = _grams(X_pre, X_post)
    W_aug = asymmetric_transfer(W, stats, percdamp=1e-4)

    Y_pre = X_pre @ W.t()
    loss_W = (Y_pre - X_post @ W.t()).pow(2).mean().item()
    loss_aug = (Y_pre - X_post @ W_aug.t()).pow(2).mean().item()
    assert loss_aug < 0.5 * loss_W, (
        f"attenuation loss decrease too small: ratio={loss_aug/loss_W:.3f}"
    )


def test_transfer_is_closed_form_optimum():
    """W_aug should be the unique minimiser; perturbing it any direction
    should not lower the asymmetric loss."""
    W, X_pre, X_post = _make_random_problem(eps=0.3, seed=22)
    stats = _grams(X_pre, X_post)
    W_aug = asymmetric_transfer(W, stats, percdamp=1e-4)
    Y_pre = X_pre @ W.t()

    base = (Y_pre - X_post @ W_aug.t()).pow(2).mean().item()
    g = torch.Generator().manual_seed(33)
    for _ in range(8):
        delta = 0.1 * W_aug.norm() / math.sqrt(W_aug.numel()) * \
            torch.randn_like(W_aug, generator=None)
        # generator on tensor doesn't accept Generator object directly here;
        # fall back to seeded torch.randn:
        delta = 0.1 * W_aug.norm() / math.sqrt(W_aug.numel()) * \
            torch.randn(W_aug.shape, generator=g)
        perturbed = W_aug + delta
        loss = (Y_pre - X_post @ perturbed.t()).pow(2).mean().item()
        assert loss >= base * 0.999 - 1e-9, (
            f"perturbation lowered objective: base={base:.4e}, perturbed={loss:.4e}"
        )


# ----- 3. TRIAD basis-commute ----------------------------------------------

def test_transfer_commutes_with_orthogonal_basis_change():
    """W_aug · U   ==   asymmetric_transfer(W·U, stats_in_U_basis).

    Where stats_in_U_basis are the Grams of (X·U) and (X̃·U).
    """
    W, X_pre, X_post = _make_random_problem(d_in=24, d_out=16, eps=0.2, seed=44)
    stats = _grams(X_pre, X_post)

    g = torch.Generator().manual_seed(55)
    A = torch.randn(stats.d_in, stats.d_in, generator=g, dtype=torch.float32)
    U, _ = torch.linalg.qr(A)                 # random orthonormal

    # Path 1: transfer in original basis, then rotate.
    W_aug = asymmetric_transfer(W, stats, percdamp=1e-4)
    W_aug_then_U = W_aug @ U

    # Path 2: rotate inputs into U-basis, recompute Grams, transfer there.
    X_pre_U = X_pre @ U
    X_post_U = X_post @ U
    stats_U = _grams(X_pre_U, X_post_U)
    W_in_U = W @ U
    W_aug_in_U = asymmetric_transfer(W_in_U, stats_U, percdamp=1e-4)

    # The two should agree to fp32 numeric noise.
    diff = (W_aug_then_U - W_aug_in_U).norm() / W_aug_then_U.norm().clamp_min(1e-12)
    assert diff.item() < 1e-3, (
        f"transfer does not commute with orthogonal basis change: rel diff={diff.item():.3e}"
    )


# ----- 4. Lambda^β rescaling commute (TRIAD's full transform) -------------

def test_transfer_commutes_with_diag_rescale():
    W, X_pre, X_post = _make_random_problem(d_in=20, d_out=16, eps=0.2, seed=66)
    stats = _grams(X_pre, X_post)

    # Per-channel positive scale (the Λ^β part of TRIAD).
    g = torch.Generator().manual_seed(77)
    s = torch.rand(stats.d_in, generator=g, dtype=torch.float32) + 0.5  # in [0.5, 1.5]
    s_inv = 1.0 / s

    W_aug = asymmetric_transfer(W, stats, percdamp=1e-4)
    W_aug_scaled = W_aug * s.unsqueeze(0)         # absorb diag(s) into columns

    # Equivalent: scale inputs by 1/s (so X' = X · diag(1/s) → outputs unchanged
    # when W' = W · diag(s)) and recompute stats in that basis.
    X_pre_s  = X_pre  * s_inv.unsqueeze(0)
    X_post_s = X_post * s_inv.unsqueeze(0)
    stats_s = _grams(X_pre_s, X_post_s)
    W_in_s = W * s.unsqueeze(0)
    W_aug_in_s = asymmetric_transfer(W_in_s, stats_s, percdamp=1e-4)

    diff = (W_aug_scaled - W_aug_in_s).norm() / W_aug_scaled.norm().clamp_min(1e-12)
    assert diff.item() < 5e-3, f"diag-rescale commute failed: rel diff={diff.item():.3e}"
