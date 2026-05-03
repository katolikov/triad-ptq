"""End-to-end smoke test: TRIAD-PTQ on a single Linear layer.

We construct a tiny MLP, calibrate on random inputs, run TRIAD-PTQ at
INT4 and check that:
  1. The replaced model has TriadLinear modules.
  2. Forward pass is finite.
  3. Output is within a reasonable distance of the FP32 reference.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from triad_ptq import optimize
from triad_ptq.core.modules import TriadLinear


class TinyMLP(nn.Module):
    def __init__(self, d_in=128, d_h=256, d_out=64):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_h, d_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_smoke_endtoend_int4():
    torch.manual_seed(0)
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = TinyMLP().to(dev).eval()

    # calibration: 8 batches of 16 samples
    calib = [torch.randn(16, 128, device=dev) for _ in range(8)]

    # reference outputs on a fresh input
    x_eval = torch.randn(4, 128, device=dev)
    with torch.no_grad():
        y_ref = model(x_eval).clone()

    qmodel = optimize(
        model,
        bits=4,
        calibration=calib,
        super_weight_frac=1e-3,
        bit_allocator="trace",
        cov_grid="analytic",
        n_calib=8,
        rho_probe_n=2,
        progress=False,
    )

    # both layers replaced
    assert isinstance(qmodel.fc1, TriadLinear)
    assert isinstance(qmodel.fc2, TriadLinear)

    with torch.no_grad():
        y_q = qmodel(x_eval)
    assert torch.isfinite(y_q).all(), "non-finite output after quantization"

    # Quality: L2 distance reasonable (just a sanity bound for random init).
    rel = (y_q - y_ref).pow(2).mean().sqrt() / y_ref.pow(2).mean().sqrt()
    assert rel.item() < 1.0, f"relative error too large: {rel.item():.3f}"
