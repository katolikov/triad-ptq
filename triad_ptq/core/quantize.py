"""Group-wise asymmetric integer quantization for INT3..INT8.

We use *grouped per-output-channel* asymmetric quantization (group_size along
the input dimension) — the same scheme as AWQ / GPTQ and most modern weight
quantization stacks.

For a weight matrix W in (m, n) and group_size g:
    For each row i and each group [j*g : (j+1)*g]:
        q_min = 0, q_max = 2**bits - 1
        wmin, wmax = group min/max
        scale = (wmax - wmin) / (q_max - q_min)         (float)
        zero  = round(-wmin / scale)                    (int in [0, 2**b-1])
        q     = clamp(round(W / scale + zero), 0, q_max)
        deq   = (q - zero) * scale

Storage: we keep `q` as int32 (so eigh + GEMM + downstream pipelines do not
need bit-packed reads) — this trades disk size for simplicity. We also expose
a packed nibble form for INT4 to compute file-on-disk metrics that match
real INT4 footprints.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QuantizedWeight:
    q: torch.Tensor          # int32, (m, n) -- quantization codes
    scales: torch.Tensor     # fp32,  (m, n_groups)
    zeros: torch.Tensor      # int32, (m, n_groups)
    bits: int
    group_size: int

    def dequantize(self) -> torch.Tensor:
        return _dequantize(self.q, self.scales, self.zeros, self.group_size)

    def storage_bytes(self) -> int:
        n_w = self.q.numel()
        weight_bytes = (n_w * self.bits + 7) // 8
        scale_bytes = self.scales.numel() * 2  # fp16
        zero_bytes = (self.zeros.numel() * self.bits + 7) // 8
        return int(weight_bytes + scale_bytes + zero_bytes)


def _quantize_group(
    W_g: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qmax = (1 << bits) - 1
    wmin = W_g.amin(dim=-1, keepdim=True)
    wmax = W_g.amax(dim=-1, keepdim=True)
    scale = (wmax - wmin).clamp_min(1e-8) / qmax
    zero = (-wmin / scale).round().clamp(0, qmax)
    q = ((W_g / scale) + zero).round().clamp(0, qmax)
    return q.to(torch.int32), scale.squeeze(-1), zero.squeeze(-1).to(torch.int32)


def quantize_grouped(
    W: torch.Tensor,
    bits: int,
    group_size: int,
) -> QuantizedWeight:
    """Asymmetric grouped quant along input dim. W is (m, n)."""
    assert W.ndim == 2, f"expected 2-D weight, got {W.shape}"
    m, n = W.shape
    g = min(group_size, n)
    if n % g != 0:
        # pad temporarily to align groups; we'll truncate after
        pad = g - (n % g)
        W_p = torch.nn.functional.pad(W, (0, pad))
    else:
        W_p = W
    n_p = W_p.size(1)
    n_groups = n_p // g
    Wg = W_p.reshape(m, n_groups, g)
    q, scale, zero = _quantize_group(Wg, bits)  # (m, n_g, g), (m, n_g)
    q = q.reshape(m, n_p)[:, :n]
    return QuantizedWeight(q=q, scales=scale, zeros=zero, bits=bits, group_size=g)


def _dequantize(
    q: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    m, n = q.shape
    g = min(group_size, n)
    if n % g != 0:
        pad = g - (n % g)
        q_p = torch.nn.functional.pad(q, (0, pad))
    else:
        q_p = q
    n_p = q_p.size(1)
    n_groups = n_p // g
    q_grp = q_p.reshape(m, n_groups, g)
    deq = (q_grp.to(scales.dtype) - zeros.unsqueeze(-1).to(scales.dtype)) * scales.unsqueeze(-1)
    return deq.reshape(m, n_p)[:, :n]


# Convenience: round-to-nearest baseline (no calibration) -------------------

def rtn_quantize(W: torch.Tensor, bits: int, group_size: int = 64) -> QuantizedWeight:
    """Plain RTN baseline: identical to quantize_grouped (no transform)."""
    return quantize_grouped(W, bits=bits, group_size=group_size)
