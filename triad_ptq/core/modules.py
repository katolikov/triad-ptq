"""TriadLinear / TriadConv2d wrapper modules.

The fused execution path is

    y = (x @ U / Lam_pow_beta) @ W_dequant.T  +  super_weight_correction(x)

For the M1 prototype we *dequantize on the fly* — there is no INT4 GEMM in
PyTorch on MPS today, so simulated INT4 (dequantize -> fp16 GEMM) is the
honest baseline. Real INT4 throughput numbers must come from MLX-LM and are
explicitly labeled as such in the report.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantize import QuantizedWeight, _dequantize


class TriadLinear(nn.Module):
    """Drop-in replacement for nn.Linear with TRIAD quantized weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        qweight: QuantizedWeight,
        bias: Optional[torch.Tensor] = None,
        U: Optional[torch.Tensor] = None,
        Lam_pow_beta: Optional[torch.Tensor] = None,
        sw_rows: Optional[torch.Tensor] = None,
        sw_cols: Optional[torch.Tensor] = None,
        sw_vals: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = qweight.bits
        self.group_size = qweight.group_size
        self.dtype = dtype

        self.register_buffer("q", qweight.q.contiguous())
        self.register_buffer("scales", qweight.scales.to(torch.float32).contiguous())
        self.register_buffer("zeros", qweight.zeros.to(torch.int32).contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

        # Activation transform fused at export -- keep U / Lam if user wants
        # the un-fused path for diagnostics. For inference we precompute the
        # combined dequantized weight matrix W_hat once and cache it as
        # 'Wcache' (fp16). This gives "simulated INT4" inference timing on MPS.
        if U is not None and Lam_pow_beta is not None:
            self.register_buffer("U", U.contiguous())
            self.register_buffer("Lam_pow_beta", Lam_pow_beta.contiguous())
        else:
            self.U = None
            self.Lam_pow_beta = None

        if sw_rows is not None and sw_rows.numel() > 0:
            self.register_buffer("sw_rows", sw_rows.long())
            self.register_buffer("sw_cols", sw_cols.long())
            self.register_buffer("sw_vals", sw_vals.to(torch.float32))
        else:
            self.sw_rows = None
            self.sw_cols = None
            self.sw_vals = None

        # cached dequantized weight (for fast forward); built lazily on first call
        self._Wcache: Optional[torch.Tensor] = None

    def _build_cache(self) -> torch.Tensor:
        W_dq = _dequantize(self.q, self.scales, self.zeros, self.group_size)  # (m, n)
        # If U / Lam are present, fold them into the effective weight so the
        # forward is a single GEMM. The transformation we applied was
        #   W' = W @ U @ diag(Lam^beta)
        # and at inference x_t = (x @ U) / Lam^beta.
        # We can equivalently fold into W:
        #   y = (x U) / Lam^beta @ W'^T
        #     = x @ (U / Lam^beta @ W'^T)
        # so define W_eff = (U / Lam^beta) @ W'^T  -> shape (n, m), keep transposed.
        # But to keep storage simple, we instead build W_eff_t s.t. y = x @ W_eff_t.
        if self.U is not None:
            # W' is what was quantized, so W_dq ~ W' = W @ U @ diag(Lam^beta).
            # Recover transformed inverse path implicitly:
            # x_t = (x @ U) / Lam^beta  -> y = x_t @ W_dq.T
            # Equivalent: y = x @ (U / Lam^beta) @ W_dq.T
            UtoW = (self.U / self.Lam_pow_beta.unsqueeze(0)) @ W_dq.t()  # (n, m)
            W_eff = UtoW
        else:
            W_eff = W_dq.t()  # (n, m)

        # Apply super-weight correction by adding (sw_vals at (rows,cols))
        # back into W_dq (it was quantized away). The correction must be in
        # the *transformed* basis if U is present, since the cache is built
        # in the W' basis.
        if self.sw_rows is not None and self.sw_rows.numel() > 0:
            # Fill W_dq with FP16 values at sw positions before folding U.
            # We rebuild from scratch with sparse correction:
            corr = torch.zeros_like(W_dq)
            corr[self.sw_rows, self.sw_cols] = self.sw_vals
            W_dq2 = W_dq + corr
            if self.U is not None:
                W_eff = (self.U / self.Lam_pow_beta.unsqueeze(0)) @ W_dq2.t()
            else:
                W_eff = W_dq2.t()
        return W_eff.to(self.dtype).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._Wcache is None:
            self._Wcache = self._build_cache()
        # x: (..., n)  ->  y: (..., m)
        y = x.to(self._Wcache.dtype) @ self._Wcache
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        qweight: QuantizedWeight,
        *,
        U: Optional[torch.Tensor] = None,
        Lam_pow_beta: Optional[torch.Tensor] = None,
        sw_rows=None,
        sw_cols=None,
        sw_vals=None,
        dtype: torch.dtype = torch.float32,
    ) -> "TriadLinear":
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            qweight=qweight,
            bias=linear.bias.detach().clone() if linear.bias is not None else None,
            U=U,
            Lam_pow_beta=Lam_pow_beta,
            sw_rows=sw_rows,
            sw_cols=sw_cols,
            sw_vals=sw_vals,
            device=linear.weight.device,
            dtype=dtype,
        )


class TriadConv2d(nn.Module):
    """1x1-equivalent fused TRIAD conv (treats Conv2d as Linear over im2col)."""

    def __init__(
        self,
        conv: nn.Conv2d,
        qweight: QuantizedWeight,
        U: Optional[torch.Tensor] = None,
        Lam_pow_beta: Optional[torch.Tensor] = None,
        sw_rows=None,
        sw_cols=None,
        sw_vals=None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.bits = qweight.bits
        self.group_size = qweight.group_size
        self.dtype = dtype

        self.register_buffer("q", qweight.q.contiguous())
        self.register_buffer("scales", qweight.scales.to(torch.float32).contiguous())
        self.register_buffer("zeros", qweight.zeros.to(torch.int32).contiguous())
        if conv.bias is not None:
            self.register_buffer("bias", conv.bias.detach().clone())
        else:
            self.bias = None

        if U is not None and Lam_pow_beta is not None:
            self.register_buffer("U", U.contiguous())
            self.register_buffer("Lam_pow_beta", Lam_pow_beta.contiguous())
        else:
            self.U = None
            self.Lam_pow_beta = None

        if sw_rows is not None and sw_rows.numel() > 0:
            self.register_buffer("sw_rows", sw_rows.long())
            self.register_buffer("sw_cols", sw_cols.long())
            self.register_buffer("sw_vals", sw_vals.to(torch.float32))
        else:
            self.sw_rows = None
            self.sw_cols = None
            self.sw_vals = None

        self._Wcache: Optional[torch.Tensor] = None

    def _build_cache(self) -> torch.Tensor:
        W_dq = _dequantize(self.q, self.scales, self.zeros, self.group_size)
        if self.sw_rows is not None and self.sw_rows.numel() > 0:
            corr = torch.zeros_like(W_dq)
            corr[self.sw_rows, self.sw_cols] = self.sw_vals
            W_dq = W_dq + corr
        if self.U is not None:
            W_eff = (self.U / self.Lam_pow_beta.unsqueeze(0)) @ W_dq.t()  # (n, m)
        else:
            W_eff = W_dq.t()
        return W_eff.to(self.dtype).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # im2col -> matmul -> col2im
        if self._Wcache is None:
            self._Wcache = self._build_cache()
        N = x.size(0)
        cols = F.unfold(x, kernel_size=self.kernel_size, dilation=self.dilation,
                        padding=self.padding, stride=self.stride)
        # cols: (N, C*kh*kw, L)
        cols_t = cols.transpose(1, 2)  # (N, L, C*kh*kw)
        out = cols_t.to(self._Wcache.dtype) @ self._Wcache  # (N, L, out_channels)
        out = out.transpose(1, 2)  # (N, out_channels, L)
        H_in, W_in = x.size(2), x.size(3)
        H_out = (H_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        W_out = (W_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        out = out.reshape(N, self.out_channels, H_out, W_out)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1).to(out.dtype)
        return out
