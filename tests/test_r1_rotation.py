"""Phase-4 R1 rotation tests.

Three correctness checks:
1.  Hadamard orthonormality.
2.  RMSNorm fold preserves the post-norm output exactly when γ is
    absorbed into the next layer's weight.
3.  The full R1 round-trip on a tiny Llama-style block leaves the block's
    output **unchanged** (cosine ≥ 0.9999) on random input — this is
    the QuaRot computational-invariance property.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from triad_ptq.core.rotate import (
    R1Spec,
    apply_r1_to_block,
    fold_rmsnorm_into_next,
    hadamard_matrix,
    random_signed_hadamard,
    rotate_linear_input,
)


# ----- 1. Hadamard math -----------------------------------------------------

def test_hadamard_is_orthonormal():
    for d in (16, 32, 256, 1024):
        H = hadamard_matrix(d)
        err = (H.t() @ H - torch.eye(d)).norm().item()
        assert err < 1e-4, f"d={d}: ||HᵀH − I|| = {err:.3e}"


def test_random_signed_hadamard_is_orthonormal():
    for d, seed in [(16, 1), (256, 2), (1024, 3)]:
        Q = random_signed_hadamard(d, seed=seed)
        err = (Q.t() @ Q - torch.eye(d)).norm().item()
        assert err < 1e-4, f"d={d}: ||QᵀQ − I|| = {err:.3e}"


def test_hadamard_rejects_non_pow2():
    import pytest
    with pytest.raises(ValueError):
        hadamard_matrix(100)


# ----- 2. RMSNorm fold equivalence -----------------------------------------

class _RMSNorm(nn.Module):
    """Stand-in for HF LlamaRMSNorm (just enough surface for the fold test)."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        v = x.float()
        rms = v.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (v / rms * self.weight.float()).to(x.dtype)


def test_rmsnorm_fold_preserves_output():
    d = 64
    norm = _RMSNorm(d)
    norm.weight.data.copy_(torch.linspace(0.5, 1.5, d))   # non-trivial γ
    lin = nn.Linear(d, 32, bias=False)
    nn.init.normal_(lin.weight)

    g = torch.Generator().manual_seed(7)
    x = torch.randn(3, 5, d, generator=g)

    y_before = lin(norm(x))

    # Fold γ into lin.weight.
    fold_rmsnorm_into_next(norm, [lin])
    assert torch.all(norm.weight.data == 1.0)

    y_after = lin(norm(x))

    diff = (y_before - y_after).norm() / y_before.norm().clamp_min(1e-12)
    assert diff.item() < 1e-5, f"fold changed output: rel diff = {diff.item():.3e}"


# ----- 3. Tiny Llama-block round-trip --------------------------------------

class _TinyBlock(nn.Module):
    """Minimal pre-norm transformer block: norm → q/k/v + identity attn → o
    → residual → norm → gate*up (silu) → down → residual.

    The "attention" here is just a per-head identity (no softmax) — that
    isolates the R1 invariance test from softmax/non-linearity edge cases
    while still keeping the residual-stream / weight-rotation structure.
    """

    def __init__(self, d, ff):
        super().__init__()
        self.input_layernorm = _RMSNorm(d)
        self.input_layernorm.weight.data.copy_(0.5 + torch.rand(d) * 1.0)

        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

        self.post_attention_layernorm = _RMSNorm(d)
        self.post_attention_layernorm.weight.data.copy_(0.5 + torch.rand(d) * 1.0)

        self.gate_proj = nn.Linear(d, ff, bias=False)
        self.up_proj   = nn.Linear(d, ff, bias=False)
        self.down_proj = nn.Linear(ff, d, bias=False)

        for m in (self.q_proj, self.k_proj, self.v_proj, self.o_proj,
                  self.gate_proj, self.up_proj, self.down_proj):
            nn.init.normal_(m.weight, std=0.05)

    def forward(self, x):
        # Pre-attention RMSNorm + identity attention (q·v effective).
        h = self.input_layernorm(x)
        q = self.q_proj(h)
        # k unused in identity attn but kept on graph (rotates with Q).
        _ = self.k_proj(h)
        v = self.v_proj(h)
        attn_out = self.o_proj(q * v)        # placeholder for a real attn op
        x = x + attn_out

        # Pre-MLP RMSNorm + SwiGLU.
        h = self.post_attention_layernorm(x)
        g = torch.nn.functional.silu(self.gate_proj(h))
        u = self.up_proj(h)
        x = x + self.down_proj(g * u)
        return x


def test_r1_invariance_on_tiny_block():
    torch.manual_seed(0)
    d, ff = 64, 128                    # both powers of two
    block = _TinyBlock(d, ff)
    block.eval()

    g = torch.Generator().manual_seed(11)
    x = torch.randn(2, 4, d, generator=g)

    y_before = block(x).detach().clone()

    Q = random_signed_hadamard(d, seed=44)
    spec = R1Spec(
        pre_attn_norm=block.input_layernorm,
        qkv_in=[block.q_proj, block.k_proj, block.v_proj],
        o_out=[block.o_proj],
        post_attn_norm=block.post_attention_layernorm,
        gate_up_in=[block.gate_proj, block.up_proj],
        down_out=[block.down_proj],
    )
    apply_r1_to_block(spec, Q)

    # The ROTATED block expects rotated input and produces rotated output:
    #   y' = R1(block)(x · Q) = (block(x)) · Q
    x_rot = x @ Q
    y_after = block(x_rot)
    # Un-rotate to compare to original output.
    y_after_unrot = y_after @ Q.t()

    cos = torch.nn.functional.cosine_similarity(
        y_before.reshape(-1, d), y_after_unrot.reshape(-1, d), dim=-1
    ).mean().item()
    err = (y_before - y_after_unrot).norm() / y_before.norm().clamp_min(1e-12)
    assert cos > 0.9999, f"R1 invariance broken: cos={cos:.5f}, rel err={err.item():.3e}"
    assert err.item() < 1e-3, f"R1 invariance broken: rel err={err.item():.3e}"
