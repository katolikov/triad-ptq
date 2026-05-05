"""Phase B — Squisher Fisher diagonal sensitivity router.

Reference: arXiv:2507.18807. Replaces v1's empirical-Fisher KFAC + the
noise-injection ρ probe (which the v1 trace router never actually used
because it collapsed to {3, 8} watershed and shipped uniform-only).

Design summary
--------------
During the calibration forward+backward pass over a block-output
reconstruction loss, accumulate

    v_t = γ · v_{t-1} + (1 − γ) · g_t²

per parameter, where g_t is the gradient of the BRECQ block-output
reconstruction loss (we have no labels, so we cannot use LM loss).

For Phase D's 100 Adam steps per block we use γ = 0.9 (the paper's
γ = 0.999 requires longer accumulation). The Hutchinson sanity check
(`triad_ptq/_v2/router/hutchinson_check.py`) must show Pearson ≥ 0.7
between the Squisher diagonal and the true Hessian-diagonal estimate.

ρ^(ℓ) per block is derived as

    ρ^(ℓ) = ‖∇_output L_block‖² / ‖∇_input L_block‖²

This replaces v1's noise-injection probe and feeds Phase F's GPTAQ
ρ-weighted α scheduling.

API
---
- `SquisherAccumulator` — lightweight per-parameter EMA-of-g² state.
- `squisher_fisher_diagonal(block, calib_inputs, fp16_targets, ...)` —
  one-block runner; trains `block.parameters()` with Adam on the BRECQ
  loss for `n_steps` steps, returning the per-parameter Fisher-diagonal
  estimate as a `{param_name: torch.Tensor}` dict.
- `derive_rho(block, calib_inputs, fp16_targets)` — single forward+backward
  over a frozen block, returning ρ as a Python float.
- `squisher_fisher_diagonal_for_model(...)` — wrap the per-block runner
  with a caller-provided block iterator + calibration collector.

Phase B does NOT yet wire these into the calibration loop — that is
Phase D's and Phase F's job. Phase B only ships the primitives + a
sanity correlation test against Hutchinson on a toy MLP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
import torch.nn as nn

IMPLEMENTED = True
DEFAULT_GAMMA = 0.9
DEFAULT_N_STEPS = 100
DEFAULT_LR = 1e-2
DEFAULT_BATCH_SIZE = 4


# --------------------------------------------------------------------- state

@dataclass
class SquisherAccumulator:
    """Per-parameter EMA of squared gradients.

    Holds `v_t = γ · v_{t-1} + (1 − γ) · g_t²` for every leaf parameter
    of a single nn.Module. Buffers live on the same device + dtype as the
    parameters.
    """

    gamma: float = DEFAULT_GAMMA
    state: dict[str, torch.Tensor] = field(default_factory=dict)
    n_observed: int = 0

    def init_from_module(self, module: nn.Module) -> None:
        """Allocate one zero buffer per named parameter (only learnable params)."""
        self.state.clear()
        for name, p in module.named_parameters():
            if p.requires_grad:
                self.state[name] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def observe(self, module: nn.Module) -> None:
        """Fold the current `param.grad` values into the EMA.

        Caller is responsible for invoking `loss.backward()` before this; we
        read `param.grad` (already populated by autograd) and update v_t in
        place. Parameters with `grad is None` are skipped.
        """
        gamma = self.gamma
        one_minus = 1.0 - gamma
        for name, p in module.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            buf = self.state.get(name)
            if buf is None:
                buf = torch.zeros_like(p, memory_format=torch.preserve_format)
                self.state[name] = buf
            # buf ← γ · buf + (1 − γ) · g²
            buf.mul_(gamma).addcmul_(p.grad, p.grad, value=one_minus)
        self.n_observed += 1

    def diagonal(self) -> dict[str, torch.Tensor]:
        """Return the current Fisher-diagonal estimate (a copy, detached)."""
        return {k: v.detach().clone() for k, v in self.state.items()}


# --------------------------------------------------------------------- runner

@torch.enable_grad()
def squisher_fisher_diagonal(
    module: nn.Module,
    calib_inputs: torch.Tensor,
    fp16_targets: torch.Tensor,
    *,
    n_steps: int = DEFAULT_N_STEPS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    gamma: float = DEFAULT_GAMMA,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    seed: int = 0xC0FFEE,
    return_history: bool = False,
):
    """Train `module.parameters()` for `n_steps` Adam steps on the BRECQ
    block-output reconstruction loss, accumulating the Squisher diagonal.

    Parameters
    ----------
    module
        The block to characterise. Its parameters must be trainable; the
        caller is responsible for `requires_grad_(True)`.
    calib_inputs
        Calibration activations, shape (n_calib, ...). The first dim is
        sampled into mini-batches of `batch_size`.
    fp16_targets
        Reference outputs collected from the FROZEN FP16 model on the SAME
        `calib_inputs`. Same leading dim as `calib_inputs`.
    n_steps
        Number of Adam steps. Default 100 matches Phase D's per-block budget.
    lr
        Adam learning rate. Default 1e-2.
    batch_size
        Mini-batch size sampled (with replacement) from `calib_inputs`.
    gamma
        EMA decay for the Squisher accumulator.
    loss_fn
        Optional override; default is mean squared Frobenius reconstruction
        ‖f(X; W) − target‖²_F / numel.
    seed
        Random sampler seed (only the mini-batch sampler — Adam is
        deterministic given init).
    return_history
        If True, also return the per-step loss list.

    Returns
    -------
    diag
        `{param_name: torch.Tensor}` Fisher-diagonal estimate, same shape
        as each parameter.
    history (optional)
        list[float] of length `n_steps`, the BRECQ losses at each step.
    """
    if calib_inputs.size(0) != fp16_targets.size(0):
        raise ValueError(
            f"squisher_fisher_diagonal: calib_inputs[0]={calib_inputs.size(0)} ≠ "
            f"fp16_targets[0]={fp16_targets.size(0)}"
        )
    if loss_fn is None:
        def loss_fn(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
            return (pred - tgt).pow(2).mean()

    device = next(module.parameters()).device
    calib_inputs = calib_inputs.to(device)
    fp16_targets = fp16_targets.to(device)

    accum = SquisherAccumulator(gamma=gamma)
    accum.init_from_module(module)

    params = [p for p in module.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("squisher_fisher_diagonal: no trainable parameters in module")
    opt = torch.optim.Adam(params, lr=lr)

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    n = calib_inputs.size(0)
    history: list[float] = []

    for _ in range(n_steps):
        idx = torch.randint(0, n, (batch_size,), generator=g)
        x = calib_inputs.index_select(0, idx.to(device))
        y_target = fp16_targets.index_select(0, idx.to(device))

        opt.zero_grad(set_to_none=True)
        y_pred = module(x)
        loss = loss_fn(y_pred, y_target)
        loss.backward()

        accum.observe(module)
        opt.step()

        history.append(float(loss.detach().item()))

    diag = accum.diagonal()
    if return_history:
        return diag, history
    return diag


# --------------------------------------------------------------------- ρ

@torch.enable_grad()
def derive_rho(
    module: nn.Module,
    calib_inputs: torch.Tensor,
    fp16_targets: torch.Tensor,
    *,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    eps: float = 1e-12,
) -> float:
    """Compute ρ = ‖∇_output L‖² / ‖∇_input L‖² for one block.

    The block is treated as a frozen black box: gradients flow into the
    INPUT tensor (a `requires_grad=True` clone) and into the OUTPUT
    tensor analytically (since we know the loss). Module parameters are
    NOT touched — this is the per-block sensitivity ρ used by Phase F's
    α scheduling, not the Squisher v_t.

    Numerator and denominator are sums of squares over all elements in
    the activation tensors.

    Returns
    -------
    rho : float
        The block-level ρ. Expected to be in [0.01, 100] for sane LLM
        blocks; outliers indicate a forward implementation issue.
    """
    if loss_fn is None:
        def loss_fn(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
            return (pred - tgt).pow(2).mean()

    device = next(module.parameters()).device
    x = calib_inputs.detach().to(device).clone().requires_grad_(True)
    y_target = fp16_targets.detach().to(device)

    # We need gradients w.r.t. BOTH the input x and a "view" of the output
    # y_pred. Doing this in one autograd graph is awkward (autograd.grad
    # requires the output to be a leaf or to retain its grad). The cleanest
    # approach: compute ∂L/∂x via autograd, and ∂L/∂y by reconstructing the
    # loss from a leaf clone of y_pred.
    y_pred = module(x)
    loss = loss_fn(y_pred, y_target)
    (dL_dx,) = torch.autograd.grad(loss, x, retain_graph=False)

    y_leaf = y_pred.detach().clone().requires_grad_(True)
    l2 = loss_fn(y_leaf, y_target)
    (dL_dy,) = torch.autograd.grad(l2, y_leaf)

    num = float(dL_dy.pow(2).sum().item())
    den = float(dL_dx.pow(2).sum().item())
    return num / max(den, eps)


# --------------------------------------------------------------------- whole-model

def squisher_fisher_diagonal_for_model(
    model: nn.Module,
    block_iter: Callable[[nn.Module], Iterable[tuple[str, nn.Module]]],
    calib_collector: Callable[[nn.Module, str], tuple[torch.Tensor, torch.Tensor]],
    *,
    n_steps: int = DEFAULT_N_STEPS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    gamma: float = DEFAULT_GAMMA,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run `squisher_fisher_diagonal` once per block of `model`.

    Returns a nested dict keyed by block name, each value being the
    per-param Fisher-diagonal dict. Block iteration is delegated to
    `block_iter` (the caller knows the model topology — we don't hardcode
    Llama here).

    `calib_collector(model, block_name) → (X, Y_fp16)` is the bridge to
    the caller's calibration set: it must return a pair of tensors valid
    for the named block (same leading dim, on the model device).
    """
    out: dict[str, dict[str, torch.Tensor]] = {}
    for name, block in block_iter(model):
        X, Y = calib_collector(model, name)
        out[name] = squisher_fisher_diagonal(
            block, X, Y,
            n_steps=n_steps, lr=lr, batch_size=batch_size, gamma=gamma,
        )
    return out


__all__ = [
    "DEFAULT_GAMMA",
    "DEFAULT_N_STEPS",
    "DEFAULT_LR",
    "DEFAULT_BATCH_SIZE",
    "IMPLEMENTED",
    "SquisherAccumulator",
    "squisher_fisher_diagonal",
    "derive_rho",
    "squisher_fisher_diagonal_for_model",
]
