"""AWQ baseline.

STATUS ON M1 / MPS: NOT FUNCTIONAL.

The `autoawq` package (v0.2.9) installs and quantizes successfully on M1
(takes ~5 minutes for SmolLM-135M with 32 calibration samples). However,
the produced quantized checkpoint relies on the `awq_inference_engine`
C++/CUDA extension for INT4 GEMM at inference time. There is no CPU /
Metal fallback path in autoawq today, so loading and running a quantized
model fails with errors like::

    RuntimeError: Placeholder storage has not been allocated on MPS device

We therefore CANNOT use autoawq end-to-end on M1 hardware. Two options:

  1. Implement AWQ's *algorithm* (activation-aware per-channel search) on
     top of our own quantize/modules stack so it runs on MPS. This would
     produce numbers that are comparable but technically reimplementations.
     We do this as `awq_like_quantize` below.
  2. Run autoawq on a CUDA box and ship the .safetensors back. Out of
     scope for this M1-only project.

The README documents this limitation in the Limitations section. Throughout
the benchmark sweep, the AWQ column is reported as "n/a (CUDA-only)" and
the cheap baseline is RTN-INT4. The TRIAD column remains a real
measurement on real M1 hardware.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..core.calibration import collect_input_stats, quantizable_modules
from ..core.modules import TriadConv2d, TriadLinear
from ..core.quantize import quantize_grouped


def awq_like_quantize(
    model: nn.Module,
    calibration,
    *,
    bits: int = 4,
    group_size: int = 64,
    n_calib: int = 32,
    device: torch.device | None = None,
    n_grid: int = 20,
    forward_fn=None,
):
    """A faithful reimplementation of AWQ's per-output-channel scaling search.

    AWQ scales the per-output-channel weights by  s_j = (E[|X_j|])^alpha
    where alpha in [0, 1] is searched per layer to minimize the layer's
    reconstruction MSE on the calibration set.

    This runs on M1 because we do everything in PyTorch on MPS / CPU and
    do not use autoawq's CUDA kernels. It is not bit-identical to autoawq
    but matches the published algorithm.

    Returns the model with quantizable layers replaced by TriadLinear
    wrappers (with U=identity, Lam_pow_beta=s_j folded in).
    """
    from ..compile import _module_weight2d, _set_module
    from ..utils.device import best_device

    dev = device or best_device()
    model.to(dev).eval()

    layers = quantizable_modules(model)
    cal_list = list(calibration)
    stats = collect_input_stats(
        model, cal_list, device=dev, n_max=n_calib, layers=layers,
        forward_fn=forward_fn, a_device=dev,
    )

    for name, mod in layers:
        s = stats[name]
        W = _module_weight2d(mod).to(dev).float()
        absX = s.abs_X.to(dev).clamp_min(1e-8)

        # search alpha in [0, 1] for the per-channel scale s_j = absX^alpha
        # objective: ||(W_q * s) - (W * s)||^2 weighted by absX
        best_loss = float("inf")
        best_W = W
        best_s = torch.ones_like(absX)
        for k in range(n_grid + 1):
            alpha = k / n_grid
            scale = absX.pow(alpha)
            scale = scale / scale.mean().clamp_min(1e-8)
            W_scaled = W * scale.unsqueeze(0)  # (m, n) * (1, n)
            qw = quantize_grouped(W_scaled, bits=bits, group_size=group_size)
            W_dq = qw.dequantize() / scale.unsqueeze(0)
            err = ((W - W_dq).pow(2) * absX.unsqueeze(0)).sum().item()
            if err < best_loss:
                best_loss = err
                best_W = W_scaled
                best_s = scale

        # Re-quantize with best scale and build the wrapper.
        qw = quantize_grouped(best_W, bits=bits, group_size=group_size)
        # Inference: y = (x / s) @ best_W^T  -- fold 1/s into W via U=eye and Lam=s
        # Easiest: use the U=I, Lam_pow_beta=s path (so x_t = x / s, W_eff = best_W)
        n = best_W.size(1)
        U = torch.eye(n, device=dev, dtype=torch.float32)
        Lam_b = best_s.to(torch.float32)

        if isinstance(mod, nn.Linear):
            new = TriadLinear.from_linear(mod, qw, U=U, Lam_pow_beta=Lam_b, dtype=torch.float32)
        else:
            new = TriadConv2d(mod, qw, U=U, Lam_pow_beta=Lam_b, dtype=torch.float32)
        new.to(dev)
        _set_module(model, name, new)

    return model
