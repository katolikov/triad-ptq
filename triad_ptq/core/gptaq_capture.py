"""Cascade-aware activation capture for GPTAQ Phase-2.

The asymmetric transfer in `gptaq_asym.asymmetric_transfer` needs two
per-layer Grams: H_post = E[XᵀX] (post-quant cascade) and C = E[X̃ᵀX]
(cross between the FP16 cascade input and the post-quant cascade input).

We collect them with a forward hook that stays attached for the duration
of one calibration sweep through the model. The hook does **streaming**
accumulation so the per-batch flattened activation matrix is freed
immediately and the only thing kept on `a_device` is the (d_in × d_in)
running Gram(s).

For the cross-term C we also need the FP16-cascade input X̃ at the same
layer. We obtain X̃ by running the **same calibration batch** through a
frozen FP16 reference model in lock-step, capturing X̃ via a sibling
hook. The two hooks accumulate XᵀX, X̃ᵀX̃, and X̃ᵀX simultaneously, so
the cross statistic is always paired token-for-token with the post-cascade
statistic — no stale activations.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .calibration import _flatten_input
from .gptaq_asym import GptaqStats


@contextmanager
def _capture_layer(mod: nn.Module, store: dict):
    """Hook that accumulates the flattened layer-input matrix for one
    calibration sweep. `store` is mutated with key 'X' = list of
    (T_i, d_in) fp32 CPU tensors; the caller concatenates after the
    sweep completes. Hook is removed automatically at context exit.
    """
    store["X"] = []

    def hook(_m, inputs, _output):
        x = inputs[0].detach()
        x_flat = _flatten_input(mod, x).to(torch.float32).cpu()
        store["X"].append(x_flat)

    h = mod.register_forward_hook(hook)
    try:
        yield store
    finally:
        h.remove()


@torch.no_grad()
def collect_layer_grams(
    *,
    model_quant: nn.Module,
    model_fp16: nn.Module,
    layer_name: str,
    batches: Iterable,
    device: torch.device,
    forward_fn,
    a_device: torch.device | None = None,
) -> GptaqStats:
    """Run one matched forward sweep over the calibration batches on
    both `model_quant` (rolling-quantized) and `model_fp16` (frozen FP16
    reference) and return the GPTAQ Grams for `layer_name`.

    Both models are assumed to expose a `nn.Linear` (or `nn.Conv2d`)
    submodule named `layer_name`. The two models share architecture but
    their weights differ where layers preceding `layer_name` have been
    replaced with TriadLinear in `model_quant`.

    Memory cost: 2 × T × d_in fp32 CPU tensors during the sweep. After
    the sweep, only the d_in × d_in Grams on `a_device` survive.
    """
    if a_device is None:
        a_device = torch.device("cpu")

    # Resolve the named submodule on each model.
    name_to_q = dict(model_quant.named_modules())
    name_to_f = dict(model_fp16.named_modules())
    if layer_name not in name_to_q or layer_name not in name_to_f:
        raise KeyError(f"layer '{layer_name}' missing on one of the models")

    mod_q = name_to_q[layer_name]
    mod_f = name_to_f[layer_name]

    store_q: dict = {}
    store_f: dict = {}

    model_quant.eval()
    model_fp16.eval()

    with _capture_layer(mod_q, store_q), _capture_layer(mod_f, store_f):
        for batch in batches:
            forward_fn(model_quant, batch, device)
            forward_fn(model_fp16, batch, device)

    if not store_q["X"] or not store_f["X"]:
        raise RuntimeError(f"no activations captured for layer '{layer_name}'")

    X_post = torch.cat(store_q["X"], dim=0)    # (T, d_in)
    X_pre  = torch.cat(store_f["X"], dim=0)    # (T, d_in)

    if X_post.shape != X_pre.shape:
        raise RuntimeError(
            f"shape mismatch for layer '{layer_name}': "
            f"X_post {tuple(X_post.shape)} vs X_pre {tuple(X_pre.shape)}"
        )

    # Move to a_device for the Gram accumulation. For TinyLlama the largest
    # layer has d_in = 5632, so a (d_in × d_in) fp32 Gram is 121 MB — fine
    # to materialise on either CPU or MPS.
    X_post_d = X_post.to(a_device)
    X_pre_d = X_pre.to(a_device)

    T = X_post_d.size(0)
    H_post = (X_post_d.t() @ X_post_d).div_(max(T, 1))
    C      = (X_pre_d.t() @ X_post_d).div_(max(T, 1))

    return GptaqStats(H_post=H_post, C=C, n_tokens=int(T))


@torch.no_grad()
def collect_layer_grams_quantonly(
    *,
    model_quant: nn.Module,
    layer_name: str,
    batches: Iterable,
    device: torch.device,
    forward_fn,
    a_device: torch.device | None = None,
    X_pre: torch.Tensor | None = None,
) -> GptaqStats:
    """Variant that takes a precomputed X̃ (from an earlier FP16 forward
    pass) and only runs `model_quant`. Used by the experimental "cached
    X̃" path that avoids the dual-model footprint at the cost of extra
    activation memory. `X_pre` must be (T, d_in) fp32 on any device.
    """
    if a_device is None:
        a_device = torch.device("cpu")
    if X_pre is None:
        raise ValueError("collect_layer_grams_quantonly: X_pre required")

    name_to_q = dict(model_quant.named_modules())
    if layer_name not in name_to_q:
        raise KeyError(f"layer '{layer_name}' missing on quant model")
    mod_q = name_to_q[layer_name]

    store_q: dict = {}
    model_quant.eval()
    with _capture_layer(mod_q, store_q):
        for batch in batches:
            forward_fn(model_quant, batch, device)

    if not store_q["X"]:
        raise RuntimeError(f"no activations captured for layer '{layer_name}'")
    X_post = torch.cat(store_q["X"], dim=0)
    if X_post.shape != X_pre.shape:
        raise RuntimeError(
            f"shape mismatch for layer '{layer_name}': "
            f"X_post {tuple(X_post.shape)} vs X_pre {tuple(X_pre.shape)}"
        )

    X_post_d = X_post.to(a_device)
    X_pre_d = X_pre.to(a_device).to(torch.float32)
    T = X_post_d.size(0)
    H_post = (X_post_d.t() @ X_post_d).div_(max(T, 1))
    C = (X_pre_d.t() @ X_post_d).div_(max(T, 1))
    return GptaqStats(H_post=H_post, C=C, n_tokens=int(T))
