"""GPTAQ asymmetric weight-transfer (Phase-2).

Reference: Chen et al., "GPTAQ: Asymmetric Calibration for Improved
Post-Training Quantization", arXiv:2504.02692v3. Available at
https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ.

Standard GPTQ minimizes the per-layer reconstruction error using a
Hessian computed from FP16 inputs to the layer:

    W_q* = argmin_q  || X̃ Wᵀ − X̃ W_qᵀ ||²_F     where  H̃ = X̃ᵀX̃

This implicitly assumes that at inference time layer l receives the
*same* input X̃ that was observed during calibration. For PTQ that is
wrong: layer l receives the output of the cascade of *already-quantized*
layers 0..l−1, which we denote X. The asymmetric (GPTAQ) reconstruction
target is therefore

    W_q* = argmin_q  || X̃ Wᵀ − X W_qᵀ ||²_F                       (1)

with closed-form continuous optimum (∂/∂W_q = 0):

    W_q*  = W · Cᵀ · H⁻¹       where   C = X̃ᵀX  (d_in × d_in)        (2)
                                       H = XᵀX   (d_in × d_in)

Note the orientation of the cross matrix: with `C := X̃ᵀX`, the
gradient is `−2 W (X̃ᵀX) + 2 W_q H` so `W_q* = W · C · H⁻¹`. (NOT
`W · Cᵀ · H⁻¹` — confirmed by the bug-fix in commit "Phase 2: fix
transpose in asymmetric transfer".)

Eq. (2) is the **asymmetric transfer**. We apply it to the FP16 weight
*before* invoking GPTQ rounding. The remaining rounding error is then
handled by the standard GPTQ Cholesky update with H as its Hessian.

Composition with TRIAD's basis transform W' = W·U·Λ^β
─────────────────────────────────────────────────────
TRIAD changes basis via an orthonormal U and a power-diagonal Λ^β.
Substituting the transformed inputs X' = X·U·Λ^{−β} into Eq. (2) and
unfolding shows the asymmetric transfer commutes with the TRIAD basis:

    W'_q*  =  (W · C · H⁻¹) · U · Λ^β   =  W_aug · U · Λ^β

so we may apply the asymmetric transfer **once** in the original basis
and continue with the standard TRIAD pipeline unchanged. See the
derivation in docs/decisions/010-gptaq-phase-2.md.

Numerics
────────
* H is regularised with a `percdamp · mean(diag(H))` ridge before the
  solve (matches the GPTQ damp on the rounding side and keeps both
  sides of the same numerical scale).
* `solve` is preferred over explicit inverse on the assumption that H
  is dense and moderately ill-conditioned.
* fp32 throughout the asymmetric block; the transfer is the only place
  the quantization error of layer l−1 enters the calibration so we keep
  the precision tight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GptaqStats:
    """Stats needed to run the asymmetric weight transfer for one layer."""

    H_post: torch.Tensor      # (d_in, d_in) = E[X_post X_postᵀ] (post-cascade)
    C: torch.Tensor           # (d_in, d_in) = E[X̃ X_postᵀ] (cross)
    n_tokens: int             # for diagnostics

    @property
    def d_in(self) -> int:
        return int(self.H_post.size(0))


def asymmetric_transfer(
    W: torch.Tensor,
    stats: GptaqStats,
    *,
    percdamp: float = 0.01,
    eps: float = 1e-8,
    inplace: bool = False,
) -> torch.Tensor:
    """Apply the GPTAQ asymmetric weight transfer of Eq. (2).

    Args
    ----
    W : (d_out, d_in) fp32 weight (out_features × in_features).
    stats : GptaqStats holding H_post = XᵀX and C = X̃ᵀX with X = post-
            quant-cascade input and X̃ = FP16-cascade input.
    percdamp : ridge as a fraction of mean(diag(H_post)).
    eps : floor for the ridge magnitude.
    inplace : if True, returns a tensor that may alias W (callers must
              treat the input as moved); otherwise allocates fresh.

    Returns
    -------
    W_aug : (d_out, d_in) fp32 weight, ready to feed into TRIAD's
            existing per-layer grid + GPTQ pipeline.
    """
    if W.ndim != 2:
        raise ValueError(f"W must be 2-D (d_out, d_in); got shape {tuple(W.shape)}")
    if stats.H_post.shape != (W.size(1), W.size(1)):
        raise ValueError(
            f"H_post shape {tuple(stats.H_post.shape)} ≠ (d_in, d_in)={W.size(1), W.size(1)}"
        )
    if stats.C.shape != (W.size(1), W.size(1)):
        raise ValueError(
            f"C shape {tuple(stats.C.shape)} ≠ (d_in, d_in)={W.size(1), W.size(1)}"
        )

    H = stats.H_post.to(torch.float32)
    C = stats.C.to(torch.float32)
    Wf = W.to(torch.float32)

    # Damp.
    diag = H.diagonal()
    dead = diag <= 0
    if dead.any():
        H = H.clone()
        H.diagonal()[dead] = 1.0
    damp = percdamp * H.diagonal().mean().clamp_min(eps)
    H = H + damp * torch.eye(H.size(0), device=H.device, dtype=H.dtype)

    # We want W_aug = W · C · H⁻¹ (closed-form optimum of E_asym; see eq (2)
    # in the module docstring). Solve via H · Z = Cᵀ · Wᵀ for Z, then
    # W_aug = Zᵀ:
    #     Z = H⁻¹ · Cᵀ · Wᵀ
    #     W_aug = (H⁻¹ Cᵀ Wᵀ)ᵀ = W · C · (H⁻¹)ᵀ = W · C · H⁻¹  (H is symmetric)
    rhs = C.t() @ Wf.t()            # (d_in, d_out) = Cᵀ Wᵀ
    Z = torch.linalg.solve(H, rhs)  # (d_in, d_out) = H⁻¹ Cᵀ Wᵀ
    W_aug = Z.t().contiguous()      # (d_out, d_in) = W · C · H⁻¹

    if inplace:
        W.copy_(W_aug)
        return W
    return W_aug


# ---- diagnostics ---------------------------------------------------------

def asymmetry_strength(stats: GptaqStats, H_pre: Optional[torch.Tensor] = None) -> dict:
    """Return scalars summarising how much the cascade has shifted the
    layer's input distribution. Used for ADR-quality reporting.

    * `frob_delta`  =  ||H_post − H_pre||_F  /  ||H_pre||_F
       (only computed when H_pre is provided)
    * `cross_off_diag`  =  ||C − diag(C)||_F  /  ||C||_F
       (proxy for input rotation: identity → 0)
    """
    out = {}
    H = stats.H_post.to(torch.float32)
    C = stats.C.to(torch.float32)
    out["d_in"] = int(H.size(0))
    out["n_tokens"] = stats.n_tokens
    out["H_post_trace"] = float(H.diagonal().sum().item())
    out["C_trace"] = float(C.diagonal().sum().item())

    if H_pre is not None:
        Hp = H_pre.to(torch.float32)
        denom = Hp.norm().clamp_min(1e-12)
        out["frob_delta_rel"] = float((H - Hp).norm().div(denom).item())

    diag = torch.diag(C.diagonal())
    off = C - diag
    Cn = C.norm().clamp_min(1e-12)
    out["cross_off_diag_rel"] = float(off.norm().div(Cn).item())
    return out
