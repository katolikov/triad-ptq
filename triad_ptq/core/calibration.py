"""Calibration pass: collect KFAC factors and propagation coefficient.

Implements equation (1) of TRIAD-PTQ v1.0.0:

    S^(l) = tr(A^(l)) * tr(G^(l)) * rho^(l)

where
    A^(l) = E[X X^T]                       (input-side activation Gram)
    G^(l) = E[g g^T] surrogate via output  (Fisher surrogate, see paper §2.1)
    rho^(l) = empirical inter-layer propagation coefficient

For PTQ without backward passes, G is replaced by an empirical Fisher
surrogate built from output activations and a perturbation-noise probe
(noise-substitution argument of [Li et al. 2025], cited in the paper).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

QUANT_TYPES = (nn.Linear, nn.Conv2d)


def quantizable_modules(model: nn.Module, *, allow_depthwise: bool = False) -> list[tuple[str, nn.Module]]:
    """Return (name, module) pairs for quantizable layers.

    Per paper §1.3 we include nn.Linear unconditionally and nn.Conv2d only if
    in_channels >= 32 and out_channels >= 32. Depthwise convs are excluded
    by default (KFAC-A becomes ill-conditioned on degenerate Gram matrices).
    """
    out: list[tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            out.append((name, mod))
        elif isinstance(mod, nn.Conv2d):
            if mod.in_channels < 32 or mod.out_channels < 32:
                continue
            if (not allow_depthwise) and mod.groups == mod.in_channels and mod.in_channels > 1:
                continue
            out.append((name, mod))
    return out


def _flatten_input(mod: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(mod, nn.Linear):
        return x.reshape(-1, x.size(-1))
    if isinstance(mod, nn.Conv2d):
        unfolded = F.unfold(
            x,
            kernel_size=mod.kernel_size,
            dilation=mod.dilation,
            padding=mod.padding,
            stride=mod.stride,
        )
        # (N, C*kh*kw, L) -> (N*L, C*kh*kw)
        unfolded = unfolded.transpose(1, 2).reshape(-1, unfolded.size(1))
        return unfolded
    raise TypeError(type(mod))


def _layer_in_features(mod: nn.Module) -> int:
    if isinstance(mod, nn.Linear):
        return mod.in_features
    return mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1]


@dataclass
class LayerStats:
    name: str
    in_features: int
    out_features: int
    A: torch.Tensor       # (d_in, d_in)
    Y2_mean: float        # E[||y||^2 / out_features]
    abs_X: torch.Tensor   # (d_in,) E[|X_j|]
    n_tokens: int
    rho: float = 1.0
    extras: dict = field(default_factory=dict)

    def trace_A(self) -> float:
        return float(self.A.diagonal().sum().item())

    def sensitivity(self) -> float:
        return self.trace_A() * float(self.Y2_mean) * float(self.rho)


def _default_forward(model, batch, device):
    if isinstance(batch, dict):
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        batch.pop("labels", None)
        return model(**batch)
    if isinstance(batch, (list, tuple)):
        x = batch[0].to(device) if hasattr(batch[0], "to") else batch[0]
        return model(x)
    return model(batch.to(device) if hasattr(batch, "to") else batch)


@torch.no_grad()
def collect_input_stats(
    model: nn.Module,
    batches: Iterable,
    *,
    device: torch.device,
    n_max: int = 128,
    layers: list[tuple[str, nn.Module]] | None = None,
    forward_fn=None,
    a_device: torch.device | None = None,
) -> dict[str, LayerStats]:
    """Single forward pass over `batches`. Collects A, Y2, E[|X|] per layer.

    `a_device` controls where the running A matrix lives. For models that fit
    on MPS, keep it on MPS for fast accumulation. For very large models we
    can override to CPU.
    """
    if layers is None:
        layers = quantizable_modules(model)
    if forward_fn is None:
        forward_fn = _default_forward
    if a_device is None:
        a_device = device

    accum: dict[str, dict] = {}
    out_features: dict[str, int] = {}
    handles = []

    def make_hook(name, mod):
        d_in = _layer_in_features(mod)
        accum[name] = {
            "A": torch.zeros(d_in, d_in, dtype=torch.float32, device=a_device),
            "abs_X": torch.zeros(d_in, dtype=torch.float32, device=a_device),
            "Y2_sum": 0.0,
            "Y2_count": 0,
            "n_tokens": 0,
        }
        if isinstance(mod, nn.Linear):
            out_features[name] = mod.out_features
        else:
            out_features[name] = mod.out_channels

        def hook(_m, inp, out):
            x = inp[0].detach()
            x_flat = _flatten_input(mod, x).to(torch.float32).to(a_device)
            accum[name]["A"].add_(x_flat.t() @ x_flat)
            accum[name]["abs_X"].add_(x_flat.abs().sum(dim=0))
            accum[name]["n_tokens"] += x_flat.size(0)
            y = out.detach().to(torch.float32)
            n_out_el = max(y.numel() / max(y.size(0), 1), 1)
            accum[name]["Y2_sum"] += float((y.pow(2).sum() / n_out_el).item())
            accum[name]["Y2_count"] += y.size(0)

        return hook

    for name, mod in layers:
        handles.append(mod.register_forward_hook(make_hook(name, mod)))

    model.eval()
    seen = 0
    try:
        for batch in batches:
            if seen >= n_max:
                break
            forward_fn(model, batch, device)
            seen += 1
    finally:
        for h in handles:
            h.remove()

    stats: dict[str, LayerStats] = {}
    for name, a in accum.items():
        n_tok = max(a["n_tokens"], 1)
        A = a["A"] / n_tok
        abs_X = a["abs_X"] / n_tok
        y2c = max(a["Y2_count"], 1)
        Y2 = a["Y2_sum"] / y2c
        stats[name] = LayerStats(
            name=name,
            in_features=A.size(0),
            out_features=out_features[name],
            A=A,
            Y2_mean=Y2,
            abs_X=abs_X,
            n_tokens=n_tok,
        )
    return stats


@torch.no_grad()
def estimate_rho(
    model: nn.Module,
    sample_batches: list,
    layer_names: list[str],
    *,
    device: torch.device,
    bits: int = 4,
    forward_fn=None,
    output_fn=None,
    seed: int = 0,
) -> dict[str, float]:
    """Estimate normalized rho^(l) via single noise-injection probe per layer.

    Returns rho in [0, 1] (max-normalized).
    """
    if forward_fn is None:
        forward_fn = _default_forward
    if output_fn is None:
        def output_fn(o):
            if hasattr(o, "logits"):
                return o.logits
            if isinstance(o, torch.Tensor):
                return o
            if isinstance(o, (tuple, list)):
                return o[0]
            return o

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    name_to_mod = dict(model.named_modules())
    rhos: dict[str, float] = {}

    # Reference forward: capture clean outputs and clean per-layer activations.
    clean_outputs: list[torch.Tensor] = []
    clean_layer_acts: dict[str, list[torch.Tensor]] = {n: [] for n in layer_names}

    handles = []
    for n in layer_names:
        m = name_to_mod[n]

        def make(name):
            def hook(_m, _i, o):
                clean_layer_acts[name].append(o.detach().float().cpu().clone())
            return hook
        handles.append(m.register_forward_hook(make(n)))

    model.eval()
    for batch in sample_batches:
        out = forward_fn(model, batch, device)
        clean_outputs.append(output_fn(out).detach().float().cpu().clone())
    for h in handles:
        h.remove()

    # Per-layer perturbation
    for n in layer_names:
        mod = name_to_mod[n]
        W = mod.weight.data
        rng = (W.max() - W.min()).abs().clamp_min(1e-8)
        delta = (rng / (2**bits - 1)).item()
        sigma = (delta**2 / 12.0) ** 0.5
        noise = torch.empty_like(W).normal_(generator=None) * sigma

        local_acts: list[torch.Tensor] = []

        def local_hook(_m, _i, o):
            local_acts.append(o.detach().float().cpu().clone())

        h = mod.register_forward_hook(local_hook)

        W.add_(noise)
        try:
            out_ratios = []
            for i, batch in enumerate(sample_batches):
                pert = output_fn(forward_fn(model, batch, device)).detach().float().cpu()
                d_out = (pert - clean_outputs[i]).pow(2).mean().sqrt().item()
                d_loc = (local_acts[i] - clean_layer_acts[n][i]).pow(2).mean().sqrt().item()
                d_loc = max(d_loc, 1e-12)
                out_ratios.append(d_out / d_loc)
        finally:
            W.add_(-noise)
            h.remove()

        rhos[n] = float(sum(out_ratios) / max(len(out_ratios), 1))

    if rhos:
        m = max(rhos.values())
        if m > 0:
            rhos = {k: v / m for k, v in rhos.items()}
    return rhos


def attach_rho(stats: dict[str, LayerStats], rhos: dict[str, float]) -> None:
    for n, s in stats.items():
        if n in rhos:
            s.rho = rhos[n]
