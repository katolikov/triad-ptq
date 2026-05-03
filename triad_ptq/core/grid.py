"""Cross-covariance grid (equations 4 and 5 of TRIAD-PTQ v1.0.0).

Eq (4):  W' = W * U * Lambda^beta,    X' = Lambda^{-beta} * U^T * X
Eq (5):  beta* = (1/2) * sum_k log(lambda_k) * s_k^2  /  sum_k log(lambda_k)^2 * s_k^2
         clamped to [0, 0.5], where s_k = (W @ U)[:, k].pow(2).sum() (squared
         spectral component of the weight matrix along the k-th eigenvector).

Eigendecomposition of A is done on CPU explicitly (PyTorch 2.11 has no MPS
implementation of linalg.eigh; we do not rely on the silent fallback).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..utils.device import safe_eigh


@dataclass
class GridResult:
    U: torch.Tensor          # (n, n)  eigenvectors of A
    eig: torch.Tensor        # (n,)    eigenvalues, ascending
    Lam_pow_beta: torch.Tensor  # (n,)
    beta_star: float
    s_spec: torch.Tensor     # (n,) squared spectral components of W


def closed_form_beta(eig: torch.Tensor, s_spec: torch.Tensor, *,
                     eig_floor: float = 1e-12) -> float:
    """Equation (5) of TRIAD-PTQ. Returns beta* in [0, 0.5]."""
    log_l = torch.log(eig.clamp_min(eig_floor))
    num = (log_l * s_spec).sum()
    den = (log_l.pow(2) * s_spec).sum().clamp_min(eig_floor)
    beta = 0.5 * num / den
    return float(beta.clamp(0.0, 0.5).item())


def compute_grid(
    W: torch.Tensor,
    A: torch.Tensor,
    *,
    eps: float = 1e-6,
    eig_floor: float = 1e-12,
) -> GridResult:
    """Compute (U, Lambda^beta, beta*) for a layer.

    W: (m, n)   weight matrix (out_features, in_features)
    A: (n, n)   activation Gram E[X X^T]
    """
    eig, U = safe_eigh(A, eps=eps)  # eig ascending; U columns are eigenvectors
    # squared spectral components of W along eigenvectors of A
    WU = W.to(U.dtype).to(U.device) @ U  # (m, n)
    s_spec = WU.pow(2).sum(dim=0)         # (n,)
    beta = closed_form_beta(eig, s_spec, eig_floor=eig_floor)
    Lam_b = eig.clamp_min(eig_floor).pow(beta)
    return GridResult(U=U, eig=eig, Lam_pow_beta=Lam_b, beta_star=beta, s_spec=s_spec)


def closed_form_objective(
    eig: torch.Tensor,
    s_spec: torch.Tensor,
    beta: float,
    *,
    eig_floor: float = 1e-12,
) -> float:
    """Quadratic surrogate whose minimizer is exactly eq (5) of the paper.

        F(beta) = sum_k s_k^2 * (1 - 2*beta * log lambda_k)^2

    This is the local linearization of lambda_k^{-2*beta} weighting in the
    Hessian-aligned reconstruction loss; setting dF/dbeta = 0 reproduces
    eq (5).  The closed-form test in tests/test_grid_closed_form.py uses
    this objective for its grid-search reference.
    """
    log_l = torch.log(eig.clamp_min(eig_floor))
    val = (s_spec * (1.0 - 2.0 * beta * log_l).pow(2)).sum()
    return float(val.item())


def hessian_weighted_objective(
    W: torch.Tensor,
    A: torch.Tensor,
    U: torch.Tensor,
    eig: torch.Tensor,
    beta: float,
    *,
    eig_floor: float = 1e-12,
) -> float:
    """Hessian-weighted reconstruction proxy from eq (6) of the paper.

        L(beta) ~ sum_k lambda_k^{1 - 2*beta} * s_k^2

    This is the *exact* Hessian-aligned form before the log-linear surrogate.
    Used as a sanity diagnostic; closed_form_objective above matches eq (5)
    exactly.
    """
    eig_pos = eig.clamp_min(eig_floor)
    s_spec = (W.to(U.dtype).to(U.device) @ U).pow(2).sum(dim=0)
    val = (eig_pos.pow(1.0 - 2.0 * beta) * s_spec).sum()
    return float(val.item())
