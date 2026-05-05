"""Phase D — Learnable per-block β with BRECQ-style block reconstruction.

Replaces v1's closed-form β* (paper eq. 5). The v1 closed form is provably
only first-order optimal under per-group INT4 because the per-group max is
a non-monotone piecewise function of β.

v2 design
---------
One scalar β^(ℓ) ∈ ℝ per Transformer block, initialised at 0.5 (or a
caller-supplied value, e.g. v1's closed-form β* — D3 free improvement).
The smoothing transform applied to each Linear in the block is the
standard AWQ / SmoothQuant migration::

    s_j = ((E|X_j|) ^ β) / ((max_i |W_ij|) ^ (1 − β))             (D1)
    W   ← W · diag(s)
    norm.gain ← norm.gain / s    (folded into the *preceding* RMSNorm)

Equivalent transform: the FP16 forward output is unchanged; the INT4
group quantizer of W·diag(s) sees a more uniform per-group dynamic
range, which is what the loss is optimising.

β is trained for ``n_steps`` Adam steps minimising the BRECQ block-output
reconstruction loss::

    L_block(β, α) = || f_ℓ(X; W) − f_ℓ(X; Q_G(W·diag(s(β))) ; α) ||²_F

where Q_G is per-group INT4 fake-quantize (STE) and α_g are the optional
OmniQuant LWC clip ratios applied per group (Phase D2).

Saturation detector (D1, second paragraph in plan): if more than 50 % of
the trained β values land at the boundary of [β_min, β_max] (default
[0.05, 0.95]), the trainer logs a `saturated=True` flag in the result;
the caller is expected to fall back to per-input-channel-group β.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn

from triad_ptq._v2.lwc.selective import LWCConfig, LWCParameters

IMPLEMENTED = True

DEFAULT_N_STEPS = 100
DEFAULT_LR = 1e-2
DEFAULT_BATCH_SIZE = 4
DEFAULT_BETA_INIT = 0.5
DEFAULT_BETA_MIN = 0.05
DEFAULT_BETA_MAX = 0.95


# --------------------------------------------------------------------- INT4 fake-quant

def _per_group_max(W: torch.Tensor, group_size: int) -> torch.Tensor:
    """Per-group max(|W|) along the last dim. Shape: (out, n_groups)."""
    out, in_ = W.shape
    if in_ % group_size != 0:
        raise ValueError(f"Linear in_features {in_} not divisible by group_size {group_size}")
    Wg = W.reshape(out, in_ // group_size, group_size)
    return Wg.abs().amax(dim=-1)


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Straight-through-estimator round: forward x.round(), backward identity."""
    return (x.round() - x).detach() + x


def fake_quantize_int4_per_group(
    W: torch.Tensor,
    group_size: int,
    *,
    alpha: torch.Tensor | None = None,
    bits: int = 4,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable per-group symmetric INT4 fake-quant (STE backward).

    Uses the q4f16_1 layout's symmetric scale convention::
        s_g = α_g · max(|W_g|) / (2^{b−1} − 1)
        q_g = clip(round(W_g / s_g), −(2^{b−1}−1), 2^{b−1}−1)
        W_g_dq = q_g · s_g

    α_g defaults to 1.0 (no LWC). When provided it is broadcast over the
    per-group max — α_g ∈ [0.5, 1.0] is the LWC clip-ratio learned in
    Phase D2.
    """
    out, in_ = W.shape
    n_groups = in_ // group_size
    Wg = W.reshape(out, n_groups, group_size)
    max_g = Wg.abs().amax(dim=-1, keepdim=True)  # (out, n_groups, 1)
    if alpha is not None:
        max_g = max_g * alpha.unsqueeze(0).unsqueeze(-1)  # alpha (n_groups,) → (1, n_groups, 1)
    max_int = (1 << (bits - 1)) - 1
    s = (max_g / max_int).clamp(min=eps)
    q = _ste_round(Wg / s).clamp(-max_int, max_int)
    Wq = q * s
    return Wq.reshape(out, in_)


# --------------------------------------------------------------------- patch

@dataclass
class _PatchedLinear:
    """One handle into a swap of `lin.weight` for a learnable-β-driven
    surrogate. `restore()` returns the linear to its original state.

    This is a one-shot training-time wrapper — at inference the smoothed
    quantized weight is simply written back to lin.weight (see
    `bake_into_module` below). No persistent monkey-patch is needed.
    """

    lin: nn.Linear
    e_abs_x: torch.Tensor   # (in_features,)  per-channel E|X|
    max_abs_w: torch.Tensor # (in_features,)  precomputed once
    original_weight: torch.Tensor


def _collect_e_abs_x(block: nn.Module, X: torch.Tensor, linears: list[nn.Linear]) -> dict[int, torch.Tensor]:
    """Forward `X` through `block` once with hooks on each Linear in
    `linears`, collecting E|x| per input channel.

    Returns id(linear) → tensor of shape (in_features,).
    """
    out: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(lin: nn.Linear):
        def hook(module, inputs, output):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            x_flat = x.reshape(-1, x.size(-1)).abs()
            out[id(lin)] = x_flat.mean(dim=0).detach()
        return hook

    for lin in linears:
        handles.append(lin.register_forward_hook(make_hook(lin)))
    try:
        with torch.no_grad():
            block(X)
    finally:
        for h in handles:
            h.remove()

    for lin in linears:
        if id(lin) not in out:
            raise RuntimeError(f"E|X| capture missed linear with id={id(lin)}")
    return out


# --------------------------------------------------------------------- result

@dataclass
class LearnableBetaResult:
    beta: float
    beta_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
    saturated: bool = False
    lwc_alpha: dict[str, torch.Tensor] = field(default_factory=dict)  # name → (n_groups,)
    lwc_enabled: bool = False
    n_steps: int = 0
    n_linears: int = 0


# --------------------------------------------------------------------- trainer

@torch.enable_grad()
def train_learnable_beta(
    block: nn.Module,
    X: torch.Tensor,
    Y_target: torch.Tensor,
    quantizable_linears: Iterable[nn.Linear],
    *,
    group_size: int = 64,
    n_steps: int = DEFAULT_N_STEPS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    beta_init: float = DEFAULT_BETA_INIT,
    beta_min: float = DEFAULT_BETA_MIN,
    beta_max: float = DEFAULT_BETA_MAX,
    lwc: LWCConfig | None = None,
    bits: int = 4,
    seed: int = 0xD0D0,
    return_history: bool = True,
) -> LearnableBetaResult:
    """Train one scalar β (and optional LWC α_g) for the given block.

    Parameters
    ----------
    block
        Transformer block. Must expose `quantizable_linears` (caller passes
        the list — typically [q_proj, k_proj, v_proj, o_proj, gate_proj,
        up_proj, down_proj]).
    X
        Calibration inputs, shape (n_calib, ..., d_in_block).
    Y_target
        FP16 reference outputs, same leading dims, last dim d_out_block.
    quantizable_linears
        Linear modules within `block` whose weights are quantized at INT4.
    group_size
        Per-group quantization granularity G.
    n_steps, lr, batch_size
        Adam hyperparameters.
    beta_init, beta_min, beta_max
        Initial value and clamp bounds for β. Saturation detector flags
        β within 1e-3 of either bound.
    lwc
        Optional LWCConfig — if provided, an α_g parameter is added per
        Linear and trained jointly with β. Pass None to disable.
    bits
        Quantizer bit width. Default 4.

    Returns
    -------
    LearnableBetaResult
    """
    quantizable_linears = list(quantizable_linears)
    if not quantizable_linears:
        raise ValueError("quantizable_linears is empty")

    device = X.device

    # Pre-compute E|X_j| for each linear and snapshot original weights.
    e_abs_x = _collect_e_abs_x(block, X[: min(X.size(0), max(batch_size, 8))], quantizable_linears)

    handles: list[_PatchedLinear] = []
    for lin in quantizable_linears:
        handles.append(_PatchedLinear(
            lin=lin,
            e_abs_x=e_abs_x[id(lin)].to(device),
            max_abs_w=lin.weight.detach().abs().amax(dim=0).to(device).clamp(min=1e-8),
            original_weight=lin.weight.detach().clone(),
        ))

    # The single learnable scalar β.
    beta = nn.Parameter(torch.tensor(float(beta_init), device=device))

    # Optional LWC α_g per linear.
    lwc_params: dict[str, LWCParameters] = {}
    if lwc is not None and lwc.enabled:
        for h in handles:
            in_ = h.lin.in_features
            n_groups = in_ // group_size
            lwc_params[str(id(h.lin))] = LWCParameters(
                alpha=nn.Parameter(torch.full((n_groups,), 1.0, device=device)),
                alpha_min=lwc.alpha_min,
                alpha_max=lwc.alpha_max,
            )

    # Optimiser.
    opt_params: list[nn.Parameter] = [beta]
    for p in lwc_params.values():
        opt_params.append(p.alpha)
    opt = torch.optim.Adam(opt_params, lr=lr)

    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    n = X.size(0)

    beta_hist: list[float] = []
    loss_hist: list[float] = []

    def smooth_quantize(h: _PatchedLinear, beta_v: torch.Tensor, alpha_v: torch.Tensor | None) -> torch.Tensor:
        """Return the (smooth-then-quantize) effective weight for one linear.

        s_j = (E|X_j|)^β / (max_i |W_ij|)^{1−β};   W_eff = Q_G(W_orig · diag(s)).
        We apply the inverse smoothing to the input later (via the fold)
        but for the BRECQ training of β alone we can absorb both sides:
        f(X; W_eff) = f(X · diag(1/s); Q_G(W_orig · diag(s))). The Linear's
        forward is `X @ W_eff^T` so we equivalently compute:

            X' = X / s ;  W' = W_orig · diag(s) ;  W'_q = Q_G(W')
            y = X' @ W'_q^T

        which is what `_BetaSmoothLinear.forward` below does.
        """
        eps = 1e-8
        s = (h.e_abs_x.clamp(min=eps).pow(beta_v)) / (h.max_abs_w.clamp(min=eps).pow(1.0 - beta_v))
        Wp = h.original_weight * s.unsqueeze(0)  # column-scale by s
        if alpha_v is None:
            Wq = fake_quantize_int4_per_group(Wp, group_size, bits=bits)
        else:
            Wq = fake_quantize_int4_per_group(Wp, group_size, alpha=alpha_v.clamp(lwc.alpha_min, lwc.alpha_max), bits=bits)  # type: ignore[arg-type]
        return s, Wp, Wq  # type: ignore[return-value]

    # -- monkey-patch each Linear's forward to use the smoothed-quantized weight,
    #    keeping the parameter state intact so we can restore at the end.
    original_forwards: list[tuple[nn.Linear, callable]] = []  # type: ignore[name-defined]

    def patch_linear(h: _PatchedLinear) -> None:
        lin = h.lin
        original_forwards.append((lin, lin.forward))

        def new_forward(x: torch.Tensor) -> torch.Tensor:
            beta_v = beta.clamp(beta_min, beta_max)
            ap = lwc_params.get(str(id(lin)))
            alpha_v = ap.alpha if ap is not None else None
            s, _Wp, Wq = smooth_quantize(h, beta_v, alpha_v)  # type: ignore[misc]
            x_smooth = x / s
            out = torch.nn.functional.linear(x_smooth, Wq, lin.bias)
            return out

        lin.forward = new_forward  # type: ignore[assignment]

    try:
        for h in handles:
            patch_linear(h)

        for step in range(n_steps):
            idx = torch.randint(0, n, (min(batch_size, n),), generator=rng)
            xb = X.index_select(0, idx.to(device))
            yb = Y_target.index_select(0, idx.to(device))

            opt.zero_grad(set_to_none=True)
            y_pred = block(xb)
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]
            loss = (y_pred - yb).pow(2).mean()
            loss.backward()
            opt.step()

            with torch.no_grad():
                beta.data.clamp_(beta_min, beta_max)
                for ap in lwc_params.values():
                    ap.alpha.data.clamp_(ap.alpha_min, ap.alpha_max)

            if return_history:
                beta_hist.append(float(beta.detach().item()))
                loss_hist.append(float(loss.detach().item()))

    finally:
        # Restore original forward methods on every linear.
        for lin, fwd in original_forwards:
            try:
                del lin.forward  # type: ignore[attr-defined]
            except AttributeError:
                pass

    final_beta = float(beta.detach().item())
    saturated = (final_beta - beta_min) < 1e-3 or (beta_max - final_beta) < 1e-3
    lwc_alpha_out: dict[str, torch.Tensor] = {
        k: v.alpha.detach().clone() for k, v in lwc_params.items()
    }

    return LearnableBetaResult(
        beta=final_beta,
        beta_history=beta_hist,
        loss_history=loss_hist,
        saturated=saturated,
        lwc_alpha=lwc_alpha_out,
        lwc_enabled=bool(lwc_params),
        n_steps=n_steps,
        n_linears=len(handles),
    )


# --------------------------------------------------------------------- bake

@torch.no_grad()
def bake_smoothed_weights(
    quantizable_linears: Iterable[nn.Linear],
    e_abs_x_per_lin: dict[int, torch.Tensor],
    beta: float,
    *,
    fold_into_norm: nn.Module | None = None,
) -> None:
    """Apply the FINAL smoothing transform to each Linear's weight (no fake
    quant — Phase D's job ends at the equivalent transform; the per-group
    INT4 rounding happens in the caller's existing GPTQ-style solver).

    s_j is folded into the preceding `fold_into_norm.weight` (γ ← γ / s)
    if the caller passes one. Otherwise the inverse smoothing must be
    folded into the previous block by the caller.
    """
    eps = 1e-8
    if fold_into_norm is not None:
        gamma = fold_into_norm.weight.data.detach().clone()
    else:
        gamma = None

    for lin in quantizable_linears:
        ex = e_abs_x_per_lin[id(lin)].to(lin.weight.device, lin.weight.dtype).clamp(min=eps)
        mw = lin.weight.detach().abs().amax(dim=0).to(lin.weight.device, lin.weight.dtype).clamp(min=eps)
        s = (ex.pow(beta)) / (mw.pow(1.0 - beta))
        lin.weight.data.mul_(s.unsqueeze(0))
        if gamma is not None:
            # All linears share the same input axis here; we'll only divide
            # the gamma once per call. Caller is responsible for grouping
            # the fold so that all q_proj/k_proj/v_proj see the same s.
            gamma = gamma / s

    if fold_into_norm is not None and gamma is not None:
        fold_into_norm.weight.data.copy_(gamma)


__all__ = [
    "DEFAULT_N_STEPS",
    "DEFAULT_LR",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_BETA_INIT",
    "IMPLEMENTED",
    "LearnableBetaResult",
    "fake_quantize_int4_per_group",
    "train_learnable_beta",
    "bake_smoothed_weights",
]
