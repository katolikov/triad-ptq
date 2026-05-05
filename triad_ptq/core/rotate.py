"""Phase-4 — Offline R1 Hadamard pre-rotation.

Implements the R1 (offline, weight-side) rotation of QuaRot
(arXiv:2404.00456 §3.1). The rotation is applied to the model's residual
stream: attention input projections (q/k/v) and MLP input projections
(gate/up) have their **input axis** rotated by Q; attention out_proj
and MLP down_proj have their **output axis** rotated by Qᵀ. Embedding
and lm_head are rotated correspondingly so the network's input/output
behavior is preserved (computational invariance).

The R1 rotation absorbs into the dense fp16 weights — there is **zero
runtime cost** at inference, no kernel changes, no graph rewrite. After
applying R1 we run TRIAD calibration on the rotated weights as if they
were original.

Why this helps INT4 quantisation
────────────────────────────────
A random orthogonal Q (in particular, a random-sign Hadamard) acts as
an "outlier-mixer": each output channel of a Q-multiplied weight is a
sign-flipped sum of all input channels, which heavy-tails the residual
stream activations into a near-Gaussian distribution. This shrinks the
per-group quantisation range and reduces the per-group MSE — the
standard outlier-suppression trick that SmoothQuant, AWQ, OmniQuant,
DuQuant, QuaRot, and SpinQuant all explore.

Computational invariance via RMSNorm
────────────────────────────────────
Llama-style models put an RMSNorm before each attention/MLP block:

    y = γ ⊙ (x / rms(x))    where γ ∈ R^d (learnable per-channel scale)

This is **diagonal** — it commutes with no orthogonal Q. To make Q
absorbable we first **fold γ into the next layer's weight**:

    W ← W · diag(γ)
    γ ← 1                 (RMSNorm scale set to all-ones)

After the fold, the operation `y = (x/rms(x)) · γ ⊙ W` becomes
`y = (x/rms(x)) · W` and Q can be applied to the input axis of W
without violating equivalence.

The lm_head and embedding rotations are handled symmetrically — see
`apply_r1_to_llama` for the full plan.

Power-of-two constraint
──────────────────────
A pure dyadic Hadamard exists for sizes that are powers of two. We
target Llama-class hidden dims (Llama-2-7B: 4096 = 2^12; TinyLlama-1.1B:
2048 = 2^11). Both factor cleanly. For non-power-of-two sizes one would
need a Walsh-Hadamard-Paley construction or fall back to a random
orthogonal matrix; we error out for now.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------- helpers

def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def hadamard_matrix(d: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build a Sylvester-construction Hadamard matrix of size d x d, normalised
    so that QᵀQ = I. Requires d to be a power of two.
    """
    if not _is_pow2(d):
        raise ValueError(f"hadamard_matrix: dimension {d} is not a power of two")
    # Iteratively double up: H_{2k} = [[H_k, H_k], [H_k, -H_k]] / sqrt(2)
    H = torch.tensor([[1.0]], dtype=dtype)
    while H.size(0) < d:
        top = torch.cat([H, H], dim=1)
        bot = torch.cat([H, -H], dim=1)
        H = torch.cat([top, bot], dim=0)
    H = H / (float(d) ** 0.5)
    # Sanity: QᵀQ ≈ I
    return H


def random_signed_hadamard(
    d: int, *, seed: int = 0xACE1, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Random-sign Hadamard:  Q = D · H  where D is a random ±1 diagonal.
    Acts as an effective random orthogonal matrix on residual-stream data
    while keeping the explicit Hadamard structure (relevant if a future
    online-rotation kernel wants to exploit the FWHT structure).
    """
    H = hadamard_matrix(d, dtype=dtype)
    g = torch.Generator().manual_seed(int(seed))
    signs = (torch.randint(0, 2, (d,), generator=g, dtype=torch.int32) * 2 - 1).to(dtype)
    return H * signs.unsqueeze(0)  # equivalent to H · diag(signs)


# --------------------------------------------------------------------- fold

@torch.no_grad()
def fold_rmsnorm_into_next(
    norm: nn.Module,
    weights_to_scale: list[nn.Linear],
) -> None:
    """Multiply each `Linear.weight` along its **input** axis by the RMSNorm's
    learnable scale γ, then reset γ to ones. After this call the network
    output is unchanged but the RMSNorm becomes a pure normalisation
    (no learnable scale), which is the prerequisite for the orthogonal
    rotation absorption.

    Supports any module exposing a 1-D `weight` attribute of length d that
    represents the per-channel multiplicative scale (true for Llama
    `LlamaRMSNorm`).
    """
    if not hasattr(norm, "weight") or norm.weight is None:
        raise TypeError("fold_rmsnorm_into_next: norm has no .weight attribute")
    if norm.weight.ndim != 1:
        raise ValueError(
            f"fold_rmsnorm_into_next: expected 1-D γ, got shape {tuple(norm.weight.shape)}"
        )
    gamma = norm.weight.data.detach().clone()
    d = gamma.numel()
    for lin in weights_to_scale:
        if not isinstance(lin, nn.Linear):
            raise TypeError(f"weights_to_scale entries must be nn.Linear, got {type(lin)}")
        if lin.in_features != d:
            raise ValueError(
                f"in_features {lin.in_features} ≠ γ dim {d} for {lin}"
            )
        # W has shape (out, in). Multiply along the in-axis.
        lin.weight.data.mul_(gamma.to(lin.weight.dtype).to(lin.weight.device))
    norm.weight.data.fill_(1.0)


# --------------------------------------------------------------------- apply

@torch.no_grad()
def rotate_linear_input(lin: nn.Linear, Q: torch.Tensor) -> None:
    """Right-multiply the input axis of a Linear by Q:  W ← W · Q."""
    if Q.shape[0] != Q.shape[1] or Q.size(0) != lin.in_features:
        raise ValueError(f"rotate_linear_input: Q shape {tuple(Q.shape)} does not match "
                         f"in_features {lin.in_features}")
    Q_dev = Q.to(lin.weight.dtype).to(lin.weight.device)
    lin.weight.data.copy_(lin.weight.data @ Q_dev)


@torch.no_grad()
def rotate_linear_output(lin: nn.Linear, Q: torch.Tensor) -> None:
    """Left-multiply the output axis of a Linear by Qᵀ:  W ← Qᵀ · W.

    Equivalent to "the output of this layer is rotated by Qᵀ"."""
    if Q.shape[0] != Q.shape[1] or Q.size(0) != lin.out_features:
        raise ValueError(f"rotate_linear_output: Q shape {tuple(Q.shape)} does not match "
                         f"out_features {lin.out_features}")
    Q_dev = Q.to(lin.weight.dtype).to(lin.weight.device)
    lin.weight.data.copy_(Q_dev.t() @ lin.weight.data)


@torch.no_grad()
def rotate_embedding_output(emb: nn.Embedding, Q: torch.Tensor) -> None:
    """Embedding writes into the residual stream — its output axis is the
    embedding dim. Rotate by Qᵀ: rows ← rows · Q (since each row is the
    column of the writeout)."""
    if Q.shape[0] != Q.shape[1] or Q.size(0) != emb.embedding_dim:
        raise ValueError(f"rotate_embedding_output: Q shape {tuple(Q.shape)} does not "
                         f"match embedding_dim {emb.embedding_dim}")
    Q_dev = Q.to(emb.weight.dtype).to(emb.weight.device)
    emb.weight.data.copy_(emb.weight.data @ Q_dev)


# --------------------------------------------------------------------- spec

@dataclass
class R1Spec:
    """Description of which modules to fold + rotate for one residual stream.

    `qkv_in`, `o_out`, `gate_up_in`, `down_out` are the four GROUPS of
    Linear layers per transformer block. `pre_attn_norm` and
    `post_attn_norm` are the two RMSNorms.

    For a Llama block:
        pre_attn_norm     = block.input_layernorm
        qkv_in            = [block.self_attn.q_proj, k_proj, v_proj]
        o_out             = [block.self_attn.o_proj]
        post_attn_norm    = block.post_attention_layernorm
        gate_up_in        = [block.mlp.gate_proj, block.mlp.up_proj]
        down_out          = [block.mlp.down_proj]
    """

    pre_attn_norm: nn.Module
    qkv_in: list[nn.Linear]
    o_out: list[nn.Linear]
    post_attn_norm: nn.Module
    gate_up_in: list[nn.Linear]
    down_out: list[nn.Linear]


@torch.no_grad()
def apply_r1_to_block(spec: R1Spec, Q: torch.Tensor) -> None:
    """Apply R1 to one transformer block.

    Sequence (per-block):
      1. Fold pre_attn_norm.γ into [q,k,v]_proj (input axis).
      2. Rotate input axis of [q,k,v]_proj by Q.
      3. Rotate output axis of o_proj by Qᵀ.
      4. Fold post_attn_norm.γ into [gate,up]_proj (input axis).
      5. Rotate input axis of [gate,up]_proj by Q.
      6. Rotate output axis of down_proj by Qᵀ.

    The rotation only touches the RESIDUAL stream — neither the
    attention head dim (handled by R2/R3 in the QuaRot taxonomy) nor
    the MLP intermediate (handled by R4) are rotated here.
    """
    fold_rmsnorm_into_next(spec.pre_attn_norm, spec.qkv_in)
    for lin in spec.qkv_in:
        rotate_linear_input(lin, Q)
    for lin in spec.o_out:
        rotate_linear_output(lin, Q)

    fold_rmsnorm_into_next(spec.post_attn_norm, spec.gate_up_in)
    for lin in spec.gate_up_in:
        rotate_linear_input(lin, Q)
    for lin in spec.down_out:
        rotate_linear_output(lin, Q)


@torch.no_grad()
def apply_r1_to_llama(model: nn.Module, *, seed: int = 0xACE1) -> dict:
    """Convenience wrapper for HF Llama-family models.

    Walks `model.model.layers` and applies R1 with a single (cached)
    rotation matrix Q across all blocks. Also rotates the embedding
    output and lm_head input + folds `model.model.norm` into lm_head.
    Returns a dict of diagnostics (Q hash, n_blocks, ...).
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("apply_r1_to_llama: not an HF Llama-family model")

    cfg = model.config
    d = int(cfg.hidden_size)
    Q = random_signed_hadamard(d, seed=seed)

    # Embedding output → rotate by Q (input to first block's residual stream).
    emb = getattr(model.model, "embed_tokens", None)
    if emb is None:
        raise RuntimeError("apply_r1_to_llama: cannot find model.model.embed_tokens")
    rotate_embedding_output(emb, Q)

    # Per-block fold + rotate.
    n_blocks = 0
    for blk in model.model.layers:
        spec = R1Spec(
            pre_attn_norm=blk.input_layernorm,
            qkv_in=[blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj],
            o_out=[blk.self_attn.o_proj],
            post_attn_norm=blk.post_attention_layernorm,
            gate_up_in=[blk.mlp.gate_proj, blk.mlp.up_proj],
            down_out=[blk.mlp.down_proj],
        )
        apply_r1_to_block(spec, Q)
        n_blocks += 1

    # Final RMSNorm + lm_head: fold γ into lm_head's input axis, then rotate
    # lm_head's input axis by Q. (lm_head reads from residual stream, so
    # post-rotation input is Q⁻¹·X, but Q is orthogonal so we use Qᵀ on the
    # weight: equivalent.)
    final_norm = getattr(model.model, "norm", None)
    lm_head = getattr(model, "lm_head", None)
    if final_norm is None or lm_head is None:
        raise RuntimeError("apply_r1_to_llama: missing model.model.norm or model.lm_head")
    fold_rmsnorm_into_next(final_norm, [lm_head])
    rotate_linear_input(lm_head, Q)

    return {
        "hidden_size": d,
        "n_blocks": n_blocks,
        "Q_seed": int(seed),
        "Q_frob": float(Q.norm().item()),
        "Q_orthogonality_err": float((Q.t() @ Q - torch.eye(d)).norm().item()),
    }
