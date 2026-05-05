"""Phase B — Hutchinson trace-of-Hessian sanity check.

Cross-validates the Squisher Fisher-diagonal estimate against a true
diagonal-Hessian estimator computed via Hutchinson's trick

    diag(H) ≈  (1/M) · Σ_m  z_m ⊙ (H z_m)        z_m ~ Rademacher / Normal

where each `H z_m` is computed via the standard double-backward:

    Hz = ∂/∂θ  ⟨∂L/∂θ, z⟩

Required by the Phase B acceptance criterion in the v2 plan: Pearson
correlation ≥ 0.7 between the Squisher diagonal and the Hutchinson
diagonal on a 2-layer toy MLP.

This is NOT in the inference path — it runs once at calibration time to
warn if the Squisher accumulator is producing a degenerate signal.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

DEFAULT_HUTCHINSON_M = 50


@torch.enable_grad()
def hutchinson_diagonal(
    module: nn.Module,
    loss_closure: Callable[[], torch.Tensor],
    *,
    n_samples: int = DEFAULT_HUTCHINSON_M,
    distribution: str = "rademacher",
    seed: int = 0xBEEF,
) -> dict[str, torch.Tensor]:
    """Estimate the diagonal of ∇²L w.r.t. `module.parameters()`.

    Parameters
    ----------
    module
        Module whose parameters we differentiate against.
    loss_closure
        Zero-arg callable returning a scalar `loss` tensor with grad enabled
        for `module.parameters()`. Called `n_samples` times.
    n_samples
        Number of Hutchinson probes. Default 50 (the Phase B plan figure).
    distribution
        "rademacher" (±1) or "normal" probe distribution. Rademacher gives
        the lowest-variance estimator for diagonal targets.
    seed
        Probe RNG seed.

    Returns
    -------
    diag
        `{param_name: torch.Tensor}` Hessian-diagonal estimate, same shape
        as each parameter.
    """
    params = [p for p in module.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("hutchinson_diagonal: no trainable parameters in module")

    accum: dict[str, torch.Tensor] = {
        name: torch.zeros_like(p)
        for name, p in module.named_parameters()
        if p.requires_grad
    }
    name_of = {id(p): name for name, p in module.named_parameters() if p.requires_grad}

    g = torch.Generator(device="cpu").manual_seed(int(seed))

    for _ in range(n_samples):
        # Draw probe.
        zs: list[torch.Tensor] = []
        for p in params:
            if distribution == "rademacher":
                z_cpu = torch.randint(0, 2, p.shape, generator=g, dtype=torch.int8) * 2 - 1
                zs.append(z_cpu.to(p.device, p.dtype))
            elif distribution == "normal":
                z_cpu = torch.randn(p.shape, generator=g)
                zs.append(z_cpu.to(p.device, p.dtype))
            else:
                raise ValueError(f"unknown distribution {distribution!r}")

        # First-order grad with create_graph=True.
        loss = loss_closure()
        grads = torch.autograd.grad(loss, params, create_graph=True)
        # ⟨∇L, z⟩ — scalar.
        gz = sum((gi * zi).sum() for gi, zi in zip(grads, zs))
        # Second-order grad → Hz, one tensor per param.
        Hz = torch.autograd.grad(gz, params, retain_graph=False)

        for p, hz, z in zip(params, Hz, zs):
            accum[name_of[id(p)]].add_(hz.detach() * z.detach())

    return {k: v / n_samples for k, v in accum.items()}


def pearson_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson r between two flattened tensors."""
    a = a.detach().reshape(-1).double()
    b = b.detach().reshape(-1).double()
    if a.numel() != b.numel():
        raise ValueError(f"shape mismatch: {a.numel()} ≠ {b.numel()}")
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-30)
    return float((a @ b / denom).item())


def correlate_squisher_vs_hutchinson(
    squisher: dict[str, torch.Tensor],
    hutchinson: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Per-parameter and overall Pearson r between two diagonals.

    Returns
    -------
    {
        "<param_name_1>": float,
        ...
        "__overall__":    float,   # over the concatenation of all params
    }
    """
    if set(squisher) != set(hutchinson):
        raise ValueError(
            "key mismatch: "
            f"squisher∖hutchinson={set(squisher) - set(hutchinson)}, "
            f"hutchinson∖squisher={set(hutchinson) - set(squisher)}"
        )
    out: dict[str, float] = {}
    flat_s, flat_h = [], []
    for name in squisher:
        s = squisher[name]
        h = hutchinson[name]
        # Use absolute Hessian diagonal — sign of H_ii is informative for
        # convexity, but Squisher (g²) is non-negative. The acceptance
        # criterion compares MAGNITUDES of curvature.
        h_abs = h.abs()
        out[name] = pearson_correlation(s, h_abs)
        flat_s.append(s.reshape(-1))
        flat_h.append(h_abs.reshape(-1))
    out["__overall__"] = pearson_correlation(torch.cat(flat_s), torch.cat(flat_h))
    return out


__all__ = [
    "DEFAULT_HUTCHINSON_M",
    "hutchinson_diagonal",
    "pearson_correlation",
    "correlate_squisher_vs_hutchinson",
]
