"""Smoke test for triad_ptq.export.mlc.

Quantizes a tiny synthetic Linear-only model and verifies that
export_to_mlc produces:

  - mlc-chat-config.json
  - ndarray-cache.json
  - params_shard_0.bin

with the expected internal structure (q-codes packed into uint32 lanes,
fp16 scale-zero interleave at 2*n_groups). This catches refactor breaks
in the packing without requiring mlc_llm itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from triad_ptq import optimize
from triad_ptq.core.modules import TriadLinear
from triad_ptq.export.mlc import (
    _interleave_scale_zero_q4f16_1,
    _pack_int4_uint32_lanes,
    export_to_mlc,
)


def test_pack_int4_lanes_round_trip():
    torch.manual_seed(0)
    m, n = 3, 16
    codes = torch.randint(0, 16, (m, n), dtype=torch.int8)
    packed = _pack_int4_uint32_lanes(codes)
    assert packed.shape == (m, n // 8)
    # unpack
    unpacked = torch.zeros_like(codes)
    for k in range(8):
        unpacked[:, k::8] = ((packed >> (4 * k)) & 0xF).to(torch.int8)
    assert torch.equal(unpacked, codes)


def test_interleave_layout():
    s = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float16)
    z = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float16)
    out = _interleave_scale_zero_q4f16_1(s, z)
    assert out.shape == (1, 6)
    assert out[0].tolist() == [1.0, 10.0, 2.0, 20.0, 3.0, 30.0]


class _TinyModel(nn.Module):
    """A minimal HF-compatible-ish model for export_to_mlc smoke testing.

    Has the bare minimum that export_to_mlc reads off `model.config`:
    hidden_size, intermediate_size, num_attention_heads, num_hidden_layers,
    num_key_value_heads, vocab_size, rms_norm_eps, rope_theta,
    max_position_embeddings, tie_word_embeddings.
    """
    class Cfg:
        hidden_size = 64
        intermediate_size = 128
        num_attention_heads = 4
        num_key_value_heads = 4
        num_hidden_layers = 1
        vocab_size = 32
        rms_norm_eps = 1e-5
        rope_theta = 10000.0
        max_position_embeddings = 64
        tie_word_embeddings = False

    def __init__(self):
        super().__init__()
        self.config = self.Cfg()
        self.fc1 = nn.Linear(64, 128, bias=False)
        self.fc2 = nn.Linear(128, 64, bias=False)
        self.embed_tokens = nn.Embedding(32, 64)
        self.norm_weight = nn.Parameter(torch.ones(64))

    def forward(self, x):
        return self.fc2(torch.nn.functional.silu(self.fc1(x)))


def test_export_to_mlc_writes_expected_files(tmp_path: Path):
    torch.manual_seed(0)
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = _TinyModel().to(dev).eval()
    calib = [torch.randn(2, 8, 64, device=dev) for _ in range(4)]

    optimize(
        model, bits=4, calibration=calib,
        super_weight_frac=1e-3,
        bit_allocator="trace",
        cov_grid="analytic",
        n_calib=4, rho_probe_n=2,
        progress=False,
        a_device="cpu",
    )

    # Both fc1 and fc2 should now be TriadLinear.
    assert isinstance(model.fc1, TriadLinear)
    assert isinstance(model.fc2, TriadLinear)

    out_dir = tmp_path / "tiny-mlc"
    summary = export_to_mlc(
        model.cpu(), out_dir,
        hf_model_id="local/tiny-test",
        hf_snapshot_dir=None,  # no tokenizer copy
        fold_U=True,
        fold_super_weights=True,
    )
    assert summary["n_quant_layers"] == 2

    cfg = json.loads((out_dir / "mlc-chat-config.json").read_text())
    assert cfg["quantization"] == "q4f16_1"
    assert cfg["model_config"]["hidden_size"] == 64

    cache = json.loads((out_dir / "ndarray-cache.json").read_text())
    rec_names = {r["name"] for r in cache["records"][0]["records"]}
    assert "fc1.q_weight" in rec_names
    assert "fc2.q_weight" in rec_names
    assert "fc1.q_scale_zero" in rec_names

    shard = out_dir / "params_shard_0.bin"
    assert shard.exists()
    assert shard.stat().st_size == cache["records"][0]["nbytes"]
