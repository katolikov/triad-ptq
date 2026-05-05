"""Phase C — Offline block-diagonal random sign-flip + permutation rotation.

Reference: arXiv:2511.04214 ("Block Rotation is All You Need for MXFP4
Quantization"). Replaces v1's R1 Hadamard (`triad_ptq/core/rotate.py`).

Why a block-diagonal rotation
-----------------------------
v1's R1 is a *full-d* random-sign Hadamard. After the fold it is exactly
absorbed into the FP16 weights, but it is **not group-aligned**: a
per-group INT4 quantizer (group size G) sees a global max of a ``W·Q``
row, mixing outliers across all groups. That suits per-tensor
quantization but is suboptimal under MXFP4 / per-group INT4 (the case
we ship).

A block-diagonal Q with block size G has the property::

      |Q · x|  =  permuted_signed( |x| )    block-by-block

so the per-group max is **preserved exactly** (we don't trade scale
range for incoherence). The within-group columns are still mixed,
which gives the same outlier-flattening behaviour for activation
quantization that QuaRot exploits, but the weight-side per-group MSE
budget for the INT4 quantizer is unchanged. arXiv:2511.04214 reports
this Pareto-dominates global rotations under grouped quantization.

Two constructions
-----------------
1. **sign_perm** (default): each G×G block is ``Π · diag(ε)`` — a
   permutation composed with a random ±1 diagonal. Orthogonal,
   integer-valued, no FP rounding cost when applied; cheap to fold.
2. **block_hadamard**: each G×G block is ``H_G · diag(ε)`` (Sylvester
   Hadamard scaled by 1/√G). More within-group mixing but introduces
   FP rounding in the fold; reserved for the fallback if sign_perm
   under-performs.

Both share the same :func:`apply_block_rotation_to_llama` walker.

Group-size constraint
---------------------
``hidden_size`` must be divisible by ``group_size``. We only rotate the
RESIDUAL stream — head-dim and MLP-intermediate are not touched (those
correspond to QuaRot R2/R3/R4 and require online kernels we do not
ship). For MLC q4f16_1 the canonical G is 32; v2 also exercises G=64
(Phase G). All current target models satisfy ``hidden % G == 0`` for
G ∈ {32, 64, 128}.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from triad_ptq.core.rotate import (
    fold_rmsnorm_into_next,
    rotate_embedding_output,
    rotate_linear_input,
    rotate_linear_output,
)

IMPLEMENTED = True
RotationKind = Literal["sign_perm", "block_hadamard"]


# --------------------------------------------------------------------- builders

def _hadamard_block(g: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Sylvester Hadamard of size g × g, normalised so HᵀH = I.
    Requires g to be a power of two.
    """
    if g <= 0 or (g & (g - 1)) != 0:
        raise ValueError(f"hadamard requires g power of two, got {g}")
    H = torch.tensor([[1.0]], dtype=dtype)
    while H.size(0) < g:
        top = torch.cat([H, H], dim=1)
        bot = torch.cat([H, -H], dim=1)
        H = torch.cat([top, bot], dim=0)
    return H / (float(g) ** 0.5)


def block_signed_permutation(
    d: int,
    group_size: int,
    *,
    seed: int = 0xACE1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Construct R of shape (d, d), block-diagonal at block size ``group_size``,
    where each block is ``Π · diag(ε)`` with random Π and Bernoulli ε.

    R is orthogonal: each row and each column has exactly one ±1, all in
    distinct positions within the block, zeros elsewhere.
    """
    if d % group_size != 0:
        raise ValueError(f"d={d} not divisible by group_size={group_size}")
    n_blocks = d // group_size
    Q = torch.zeros(d, d, dtype=dtype)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    for b in range(n_blocks):
        perm = torch.randperm(group_size, generator=g)
        signs = (torch.randint(0, 2, (group_size,), generator=g, dtype=torch.int32) * 2 - 1).to(dtype)
        offset = b * group_size
        for i in range(group_size):
            Q[offset + i, offset + int(perm[i])] = signs[i]
    return Q


def block_hadamard_rotation(
    d: int,
    group_size: int,
    *,
    seed: int = 0xACE1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Construct R of shape (d, d), block-diagonal at block size ``group_size``,
    where each block is ``H_G · diag(ε)`` (Sylvester Hadamard composed with
    random ±1 diagonal). Requires ``group_size`` to be a power of two.
    """
    if d % group_size != 0:
        raise ValueError(f"d={d} not divisible by group_size={group_size}")
    H = _hadamard_block(group_size, dtype=dtype)
    n_blocks = d // group_size
    Q = torch.zeros(d, d, dtype=dtype)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    for b in range(n_blocks):
        signs = (torch.randint(0, 2, (group_size,), generator=g, dtype=torch.int32) * 2 - 1).to(dtype)
        block = H * signs.unsqueeze(0)  # right-multiply by diag(signs)
        offset = b * group_size
        Q[offset:offset + group_size, offset:offset + group_size] = block
    return Q


def build_rotation(
    d: int,
    group_size: int,
    *,
    kind: RotationKind = "sign_perm",
    seed: int = 0xACE1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dispatcher used by :func:`apply_block_rotation_to_llama` and the
    config flag ``rotation: 'sign_perm' | 'block_hadamard'``.
    """
    if kind == "sign_perm":
        return block_signed_permutation(d, group_size, seed=seed, dtype=dtype)
    if kind == "block_hadamard":
        return block_hadamard_rotation(d, group_size, seed=seed, dtype=dtype)
    raise ValueError(f"unknown rotation kind {kind!r} (expected sign_perm | block_hadamard)")


# --------------------------------------------------------------------- LLM walker

@dataclass
class BlockRotationDiagnostics:
    hidden_size: int
    group_size: int
    kind: RotationKind
    seed: int
    n_blocks: int
    n_layers: int
    Q_orthogonality_err: float
    is_block_diagonal: bool


def _check_block_diagonal(Q: torch.Tensor, group_size: int) -> bool:
    """True iff the only non-zero entries of Q live in the diagonal G×G
    blocks. Used as a structural assertion in the diagnostics — Phase G's
    group-size sweep relies on this property holding strictly.
    """
    d = Q.size(0)
    if d % group_size != 0:
        return False
    mask = torch.zeros_like(Q, dtype=torch.bool)
    for b in range(d // group_size):
        off = b * group_size
        mask[off:off + group_size, off:off + group_size] = True
    return bool((Q[~mask] == 0).all().item())


@torch.no_grad()
def apply_block_rotation_to_llama(
    model: nn.Module,
    *,
    group_size: int,
    kind: RotationKind = "sign_perm",
    seed: int = 0xACE1,
) -> BlockRotationDiagnostics:
    """Apply a single block-diagonal rotation Q to every residual stream
    interface of an HF Llama-family model.

    Sequence (per block, identical to QuaRot R1 except Q is block-diagonal):
      1. Fold pre-attn RMSNorm γ into [q,k,v]_proj input axis.
      2. Rotate input axis of [q,k,v]_proj by Q.
      3. Rotate output axis of o_proj by Qᵀ.
      4. Fold post-attn RMSNorm γ into [gate,up]_proj input axis.
      5. Rotate input axis of [gate,up]_proj by Q.
      6. Rotate output axis of down_proj by Qᵀ.
    Plus: rotate embedding output by Q, fold final RMSNorm γ into lm_head,
    rotate lm_head input axis by Q.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("apply_block_rotation_to_llama: not an HF Llama-family model")

    cfg = model.config
    d = int(cfg.hidden_size)
    if d % group_size != 0:
        raise ValueError(
            f"apply_block_rotation_to_llama: hidden_size {d} is not divisible by "
            f"group_size {group_size}"
        )

    Q = build_rotation(d, group_size, kind=kind, seed=seed)

    emb = getattr(model.model, "embed_tokens", None)
    if emb is None:
        raise RuntimeError("apply_block_rotation_to_llama: missing model.model.embed_tokens")
    rotate_embedding_output(emb, Q)

    n_layers = 0
    for blk in model.model.layers:
        fold_rmsnorm_into_next(
            blk.input_layernorm,
            [blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj],
        )
        for lin in (blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj):
            rotate_linear_input(lin, Q)
        rotate_linear_output(blk.self_attn.o_proj, Q)

        fold_rmsnorm_into_next(
            blk.post_attention_layernorm,
            [blk.mlp.gate_proj, blk.mlp.up_proj],
        )
        for lin in (blk.mlp.gate_proj, blk.mlp.up_proj):
            rotate_linear_input(lin, Q)
        rotate_linear_output(blk.mlp.down_proj, Q)
        n_layers += 1

    final_norm = getattr(model.model, "norm", None)
    lm_head = getattr(model, "lm_head", None)
    if final_norm is None or lm_head is None:
        raise RuntimeError("apply_block_rotation_to_llama: missing model.model.norm or lm_head")
    fold_rmsnorm_into_next(final_norm, [lm_head])
    rotate_linear_input(lm_head, Q)

    err = float((Q.t() @ Q - torch.eye(d)).norm().item())
    return BlockRotationDiagnostics(
        hidden_size=d,
        group_size=group_size,
        kind=kind,
        seed=seed,
        n_blocks=d // group_size,
        n_layers=n_layers,
        Q_orthogonality_err=err,
        is_block_diagonal=_check_block_diagonal(Q, group_size),
    )


# Backwards-compatible alias for the original Phase-A placeholder name.
def apply_sign_perm_to_llama(model: nn.Module, *, group_size: int, seed: int = 0xACE1):
    return apply_block_rotation_to_llama(
        model, group_size=group_size, kind="sign_perm", seed=seed
    )


__all__ = [
    "IMPLEMENTED",
    "RotationKind",
    "BlockRotationDiagnostics",
    "block_signed_permutation",
    "block_hadamard_rotation",
    "build_rotation",
    "apply_block_rotation_to_llama",
    "apply_sign_perm_to_llama",
]
