"""Round-to-nearest baseline: per-row grouped asymmetric quantization, no calibration."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..core.modules import TriadConv2d, TriadLinear
from ..core.quantize import rtn_quantize


def quantize_rtn(model: nn.Module, *, bits: int = 4, group_size: int = 64,
                 layer_filter=None, device: torch.device | None = None) -> nn.Module:
    """Replace nn.Linear / nn.Conv2d in place with RTN-quantized TriadLinear."""
    from ..core.calibration import quantizable_modules

    layers = quantizable_modules(model)
    if layer_filter is not None:
        layers = [(n, m) for (n, m) in layers if layer_filter(n)]

    for name, mod in layers:
        if isinstance(mod, nn.Linear):
            W = mod.weight.data
        else:
            W = mod.weight.data.reshape(mod.weight.size(0), -1)
        qw = rtn_quantize(W.to(torch.float32), bits=bits, group_size=group_size)
        if isinstance(mod, nn.Linear):
            new = TriadLinear.from_linear(mod, qw, dtype=torch.float32)
        else:
            new = TriadConv2d(mod, qw, dtype=torch.float32)
        if device is not None:
            new.to(device)
        else:
            new.to(W.device)
        _set_module(model, name, new)
    return model


def _set_module(root: nn.Module, dotted: str, new: nn.Module) -> None:
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)
