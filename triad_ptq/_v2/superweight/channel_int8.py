"""Phase E — Channel-grained INT8 mixed precision for super-weights.

Replaces v1's FP16 sparse super-weights (which would require a sparse-FP16
kernel at inference). Top-r % of OUTPUT channels by per-channel salience
κ_j are split off and stored at higher precision; the rest stay INT4.

Salience metric
---------------
Per Yu et al. arXiv:2411.07191 / DuQuant / AWQ: the "damage" a quantizer
inflicts on output channel j is bounded above by the product of the row
max and the activation expectation::

    κ_j = max_i (|W_ij^{rot}| · E|X_j|)             (E1)

Computed AFTER the Phase-C rotation so it sees the rotated weight (the
rotation can change which channels are most damaging). Selecting the
top-r % output channels of the rotated weight is the "channel-INT8" set.

Two deployment options (per the v2 plan)
----------------------------------------
* **Option A — no kernel change (v2.0 default).** Super-channels become a
  small DENSE INT8 tensor that is dequantised ONCE at load time and run
  as a separate FP16 GEMV. Decode-time cost ≈ 1.5 % × d × 2 bytes / cycle
  ≪ 0.5 % of the main INT4 GEMV. No sparse storage, no kernel addition.
* **Option B — fused INT4+INT8 TIR kernel.** Deferred to v2.1.

The "**same** packed tensor" requirement from the v2 plan is honoured by
keeping the master file layout identical — the per-output-channel
1-bit indicator + INT8 sub-tensor live alongside the INT4 packed tensor
inside one `.safetensors` shard. They are split *at runtime load* into
two separate dense matmul ops; the on-disk container is unified.

Note on the constraint "no sparse FP16 path": a 1.5 % × d dense FP16
GEMV is **not** sparse — every output channel of the small tensor is
present and computed densely. We never use a SparseTensor / CSR layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

IMPLEMENTED = True
DEFAULT_SUPER_CHANNEL_RATE = 0.015
INT8_MAX = 127


# --------------------------------------------------------------------- κ + select

def channel_kappa(
    W_rot: torch.Tensor,
    e_abs_x: torch.Tensor,
) -> torch.Tensor:
    """Compute κ_j = max_i (|W_ij| · E|X_j|) for the rotated weight.

    Parameters
    ----------
    W_rot : torch.Tensor
        Linear weight after Phase-C rotation, shape (out, in).
    e_abs_x : torch.Tensor
        Per-input-channel E|X|, shape (in,).

    Returns
    -------
    kappa : torch.Tensor of shape (out,)
    """
    if W_rot.ndim != 2:
        raise ValueError(f"channel_kappa: W_rot must be 2-D, got {W_rot.shape}")
    if e_abs_x.ndim != 1 or e_abs_x.numel() != W_rot.size(1):
        raise ValueError(
            f"channel_kappa: e_abs_x shape {tuple(e_abs_x.shape)} must be (in={W_rot.size(1)},)"
        )
    weighted = W_rot.detach().abs() * e_abs_x.detach().abs().unsqueeze(0)
    return weighted.amax(dim=1)


def select_super_channels(
    W_rot: torch.Tensor,
    e_abs_x: torch.Tensor,
    *,
    rate: float = DEFAULT_SUPER_CHANNEL_RATE,
    min_count: int = 1,
) -> torch.Tensor:
    """Return the indices (sorted ascending) of output channels in the
    top-`rate` fraction by κ.

    Always returns at least `min_count` channels.
    """
    if not (0.0 < rate < 1.0):
        raise ValueError(f"rate must be in (0, 1), got {rate}")
    kappa = channel_kappa(W_rot, e_abs_x)
    out = W_rot.size(0)
    n_super = max(min_count, int(round(rate * out)))
    n_super = min(n_super, out)
    # Indices of the n_super largest κ.
    _, idx = torch.topk(kappa, n_super, largest=True, sorted=True)
    return idx.sort().values  # ascending order


# --------------------------------------------------------------------- packing

@dataclass
class ChannelInt8Bundle:
    """One Linear's worth of mixed-precision packed weights.

    Layout
    ------
    * `int4_weight` — shape (out_int4, in), dtype int8 in [-8, 7]; the
      INT4 codes for the non-super output channels. Caller is
      responsible for packing two int4 codes per int8 byte at MLC export
      time.
    * `int4_scale` — shape (out_int4, n_groups), dtype float16; per-group
      INT4 fp16 scales.
    * `int8_weight` — shape (out_super, in), dtype int8 in [-127, 127];
      INT8 codes for super channels.
    * `int8_scale` — shape (out_super, n_groups), dtype float16; per-group
      INT8 fp16 scales (kept group-aligned with INT4 for layout
      uniformity, even though INT8 doesn't strictly require it).
    * `super_indices` — shape (out_super,), dtype int32; original output
      indices of the super channels.
    * `bit_indicator` — shape (out,), dtype bool; True at super positions,
      False elsewhere. Survives serialisation as a uint8 packed array.

    Reconstruction: the dequantised weight at row j is
        - int4_weight[map_int4[j]] * int4_scale[map_int4[j]]   if not super
        - int8_weight[map_int8[j]] * int8_scale[map_int8[j]]   if super
    """

    int4_weight: torch.Tensor
    int4_scale: torch.Tensor
    int8_weight: torch.Tensor
    int8_scale: torch.Tensor
    super_indices: torch.Tensor
    bit_indicator: torch.Tensor
    group_size: int

    @property
    def out_features(self) -> int:
        return int(self.bit_indicator.numel())

    @property
    def in_features(self) -> int:
        return int(self.int4_weight.size(1)) if self.int4_weight.numel() else int(self.int8_weight.size(1))


def _quantize_per_group_int(
    W: torch.Tensor,
    group_size: int,
    *,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-group integer quantize-then-dequantize. Returns (codes,
    scale). `codes.dtype = int8`; `scale.dtype = float16`.
    """
    out, in_ = W.shape
    if in_ % group_size != 0:
        raise ValueError(f"in_features {in_} not divisible by group_size {group_size}")
    n_groups = in_ // group_size
    Wg = W.reshape(out, n_groups, group_size).float()
    max_g = Wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    max_int = (1 << (bits - 1)) - 1
    scale = (max_g / max_int).to(torch.float16).float()  # round-trip via fp16 to match runtime
    q = (Wg / scale).round().clamp(-max_int, max_int).to(torch.int8)
    return q.reshape(out, in_), scale.reshape(out, n_groups).to(torch.float16)


def pack_channel_int8(
    W_rot: torch.Tensor,
    e_abs_x: torch.Tensor,
    *,
    group_size: int = 64,
    rate: float = DEFAULT_SUPER_CHANNEL_RATE,
) -> ChannelInt8Bundle:
    """Build a :class:`ChannelInt8Bundle` from a rotated weight + activation
    statistic. Pure storage operation — no inference path involved.
    """
    if W_rot.ndim != 2:
        raise ValueError(f"W_rot must be 2-D, got {W_rot.shape}")
    out, in_ = W_rot.shape
    super_idx = select_super_channels(W_rot, e_abs_x, rate=rate)
    bit_ind = torch.zeros(out, dtype=torch.bool)
    bit_ind[super_idx] = True

    int4_idx = torch.where(~bit_ind)[0]
    int8_idx = torch.where(bit_ind)[0]

    W_int4 = W_rot[int4_idx]
    W_int8 = W_rot[int8_idx]

    if W_int4.numel():
        q4, s4 = _quantize_per_group_int(W_int4, group_size, bits=4)
    else:
        q4 = torch.zeros(0, in_, dtype=torch.int8)
        s4 = torch.zeros(0, in_ // group_size, dtype=torch.float16)
    if W_int8.numel():
        q8, s8 = _quantize_per_group_int(W_int8, group_size, bits=8)
    else:
        q8 = torch.zeros(0, in_, dtype=torch.int8)
        s8 = torch.zeros(0, in_ // group_size, dtype=torch.float16)

    return ChannelInt8Bundle(
        int4_weight=q4,
        int4_scale=s4,
        int8_weight=q8,
        int8_scale=s8,
        super_indices=int8_idx.to(torch.int32),
        bit_indicator=bit_ind,
        group_size=group_size,
    )


def unpack_channel_int8(bundle: ChannelInt8Bundle) -> torch.Tensor:
    """Reconstruct the dequantised FP32 weight from a bundle. Lossy in
    general — the round-trip error is the per-group MSE inherent to INT4
    / INT8.
    """
    out = bundle.out_features
    in_ = bundle.in_features
    G = bundle.group_size
    n_groups = in_ // G
    W = torch.zeros(out, in_, dtype=torch.float32)

    super_set = set(bundle.super_indices.tolist())
    int4_rows: list[int] = [j for j in range(out) if j not in super_set]
    int8_rows: list[int] = bundle.super_indices.tolist()

    if bundle.int4_weight.numel():
        q4 = bundle.int4_weight.float()
        s4 = bundle.int4_scale.float().unsqueeze(-1)  # (out_int4, n_groups, 1)
        Wg4 = (q4.reshape(-1, n_groups, G) * s4).reshape(-1, in_)
        for new_i, orig_i in enumerate(int4_rows):
            W[orig_i] = Wg4[new_i]
    if bundle.int8_weight.numel():
        q8 = bundle.int8_weight.float()
        s8 = bundle.int8_scale.float().unsqueeze(-1)
        Wg8 = (q8.reshape(-1, n_groups, G) * s8).reshape(-1, in_)
        for new_i, orig_i in enumerate(int8_rows):
            W[orig_i] = Wg8[new_i]
    return W


# --------------------------------------------------------------------- E2 (optional)

@torch.no_grad()
def detect_true_super_weights(
    model: torch.nn.Module,
    candidate_indices: Iterable[tuple[str, int, int]],
    eval_loss_fn,
    *,
    threshold: float = 100.0,
) -> list[tuple[str, int, int, float]]:
    """For each (param_name, i, j) in `candidate_indices`, zero that single
    weight and call `eval_loss_fn(model)` → loss; restore. If loss
    increases by > `threshold` × baseline, mark the weight as a "true
    super-weight" (Yu et al. arXiv:2411.07191). Returns a list of
    `(name, i, j, loss_ratio)` tuples for the marked weights.

    The caller is responsible for picking the candidate set (typically
    the top-0.001 % weights by κ) and for providing an `eval_loss_fn`
    that returns a single scalar — usually a 64-token PPL surrogate so
    the screen takes seconds, not minutes.

    The detected super-weights are stored as FP16 scalars in the lm_head
    metadata at MLC export time and applied at inference via a one-time
    additive correction.
    """
    name_to_param = dict(model.named_parameters())
    base_loss = float(eval_loss_fn(model))
    if base_loss <= 0:
        raise ValueError("baseline loss must be > 0")
    detected: list[tuple[str, int, int, float]] = []
    for name, i, j in candidate_indices:
        p = name_to_param.get(name)
        if p is None or p.ndim != 2:
            continue
        original = p[i, j].item()
        p[i, j] = 0.0
        try:
            new_loss = float(eval_loss_fn(model))
        finally:
            p[i, j] = original
        ratio = new_loss / base_loss
        if ratio > threshold:
            detected.append((name, int(i), int(j), ratio))
    return detected


__all__ = [
    "DEFAULT_SUPER_CHANNEL_RATE",
    "INT8_MAX",
    "IMPLEMENTED",
    "ChannelInt8Bundle",
    "channel_kappa",
    "select_super_channels",
    "pack_channel_int8",
    "unpack_channel_int8",
    "detect_true_super_weights",
]
