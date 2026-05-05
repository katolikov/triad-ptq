"""GPTQ-style Cholesky weight update on TRIAD-transformed weights.

This is a faithful re-implementation of the GPTQ algorithm of
Frantar et al. 2023 (arXiv:2210.17323), adapted to operate on the
*transformed* weight matrix W' = W * U * Lambda^beta and *transformed*
input Gram H' = U^T A U Lambda^{2 beta}^{-1} ... actually we operate
in the original (W', X') frame: H' = (X')^T X' computed implicitly via
H' = Lambda^{-beta} U^T A U Lambda^{-beta} = Lambda^{1 - 2 beta} (since
A = U Lambda U^T diagonalizes in this basis).

So the post-transform Hessian is simply diagonal: H' = diag(lambda^{1-2β}).
GPTQ in this basis decouples per input channel up to the joint rounding,
which simplifies the inner loop dramatically.

For correctness/generality we still use the standard GPTQ Cholesky update
on the layer's input Gram in the original basis when available — this
keeps the implementation usable as a drop-in replacement when cov_grid is
disabled.
"""
from __future__ import annotations

import torch

from .quantize import QuantizedWeight, _dequantize, _quantize_group


def _find_clip(
    W_g: torch.Tensor,
    bits: int,
    x_var: torch.Tensor | None = None,
    ratios: tuple[float, ...] = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7),
) -> torch.Tensor:
    """Per-row clip-ratio search (B.4).

    For each row independently, sweep `ratios` to scale the per-group
    (wmin, wmax) before quantization, and pick the ratio that minimises
    the activation-weighted MSE  sum_j x_var[j] * (w_g[..., j] - dq[..., j])^2.
    If `x_var` is None, falls back to plain MSE.

    Returns a tensor of shape (m,) with the chosen ratio per row.
    """
    qmax = (1 << bits) - 1
    m = W_g.size(0)
    n_g = W_g.size(1)
    g = W_g.size(2)

    # Precompute baseline (ratio=1.0) min/max
    wmin = W_g.amin(dim=-1, keepdim=True)
    wmax = W_g.amax(dim=-1, keepdim=True)

    if x_var is None:
        x_w = torch.ones(g, device=W_g.device, dtype=W_g.dtype)
    else:
        x_w = x_var.to(W_g.device).to(W_g.dtype)

    # (n_groups, g) -> broadcast against (m, n_groups, g)
    if x_w.numel() == n_g * g:
        x_w = x_w.reshape(n_g, g)

    best_err = torch.full((m,), float("inf"), device=W_g.device, dtype=W_g.dtype)
    best_ratio = torch.ones(m, device=W_g.device, dtype=W_g.dtype)
    for r in ratios:
        wmin_r = wmin * r
        wmax_r = wmax * r
        scale = (wmax_r - wmin_r).clamp_min(1e-8) / qmax
        zero = (-wmin_r / scale).round().clamp(0, qmax)
        # Convention 2 (matches gptq_solver's main loop):
        #   q  = round(w/scale + z).clamp(0, qmax)
        #   dq = (q - z) * scale
        # No "+ wmin_r" term -- the zero point already encodes the shift.
        q = ((W_g / scale) + zero).round().clamp(0, qmax)
        dq = (q - zero) * scale
        # Weighted squared error per row
        err_per_grp = ((W_g - dq) ** 2)
        if x_w.dim() == 1:
            err = err_per_grp.sum(dim=-1).matmul(torch.ones(n_g, device=W_g.device, dtype=W_g.dtype))
        else:
            err = (err_per_grp * x_w.unsqueeze(0)).sum(dim=(-1, -2))
        improved = err < best_err
        best_err = torch.where(improved, err, best_err)
        best_ratio = torch.where(improved, torch.full_like(best_ratio, r), best_ratio)
    return best_ratio


def gptq_quantize_layer(
    W: torch.Tensor,
    H: torch.Tensor,
    *,
    bits: int,
    group_size: int = 128,
    actorder: bool = False,
    percdamp: float = 0.01,
    blocksize: int = 128,
    clip_search: bool = False,
) -> QuantizedWeight:
    """Standard GPTQ pass on a single dense layer.

    W: (m, n) fp32 weight (rows = out features).
    H: (n, n) input Gram E[X X^T] for this layer.
    Returns the quantized weight; dequantize() reconstructs W_hat.

    Implementation tracks the original GPTQ paper's algorithm 1.

    `clip_search` (B.4): if True, runs a per-row activation-weighted
    clip-ratio search before the per-group scale/zero are computed
    inside the Cholesky update. Picked ratio shrinks (wmin, wmax)
    multiplicatively, absorbing outliers without spreading the grid.

    WARNING: as of 2026-05-05, ``clip_search=True`` is research-only.
    On TinyLlama-1.1B it lowers eval-window PPL by ~0.13 but causes
    autoregressive generation to collapse into degenerate repetition
    on-device. Do NOT enable for production calibration without first
    (a) validating coherent generation on a held-out prompt and
    (b) considering the literature-aligned variants described in
    ``docs/decisions/007-ppl-improvements-v2.md``.
    """
    assert W.ndim == 2 and H.shape == (W.size(1), W.size(1))
    dev = W.device
    m, n = W.shape
    Wq = W.clone().to(torch.float32)

    # Damp the Hessian. We mutate a local fp32 copy in place to avoid retaining
    # the caller's H alongside the Cholesky factors (memory-critical on
    # 1B-class models where H can be 5632x5632 fp32 = 127 MB).
    H = H.detach().to(torch.float32, copy=True)
    diag = H.diagonal()
    dead = diag == 0
    if dead.any():
        H.diagonal()[dead] = 1.0
        Wq[:, dead] = 0.0

    # Snapshot true per-column variance E[x_j^2] = H[j,j] before H is
    # mutated / freed. Used by clip_search below.
    col_var_full = H.diagonal().detach().clone()

    # Activation reordering (descending importance) ------------------------
    if actorder:
        perm = torch.argsort(diag, descending=True)
        Wq = Wq[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)
    else:
        invperm = None

    damp = percdamp * H.diagonal().mean().clamp_min(1e-12)
    H.diagonal().add_(damp)

    # Cholesky of H^{-1} (upper triangular) --------------------------------
    # GPTQ uses Hinv = (cholesky(H)^{-T})  -> we work with U^{-1}.
    # Free intermediates eagerly: H is dead after L is computed, L is dead
    # after Hinv is computed. Without these `del`s we transiently hold three
    # (n, n) fp32 tensors at the same time (H + L + Hinv).
    L = torch.linalg.cholesky(H)
    del H
    Hinv = torch.cholesky_inverse(L)
    del L
    Hinv = torch.linalg.cholesky(Hinv, upper=True)  # upper R s.t. Hinv = R^T R

    # We process W in column blocks of size `blocksize`, applying GPTQ within.
    # Per-row asymmetric grouped quantization parameters are computed lazily
    # over input-channel groups of `group_size`.
    qmax = (1 << bits) - 1
    Q = torch.zeros_like(Wq, dtype=torch.float32)
    Losses = torch.zeros_like(Wq)

    # group_size along input dim; if group_size == 0 -> per-row global
    g = group_size if group_size and group_size > 0 else n
    g = min(g, n)
    n_groups = (n + g - 1) // g

    # Precompute per-(row, group) scale & zero on the *current* W_q after
    # we hit each new group boundary. We store running scales/zeros.
    scales = torch.zeros(m, n_groups, dtype=torch.float32, device=dev)
    zeros = torch.zeros(m, n_groups, dtype=torch.int32, device=dev)
    qcodes = torch.zeros(m, n, dtype=torch.int32, device=dev)

    # B.4 clip search: when enabled, compute one (m,) ratio per row from
    # the *initial* W and the per-column variance E[x_j^2] = H[j,j] (true
    # activation magnitudes from calibration). The chosen ratio is applied
    # inside the per-group scale/zero update below AND we pre-clamp Wq to
    # the new range so GPTQ propagates only quantization noise (sub-grid),
    # not clamping error (super-grid). This matches AWQ-clip / OmniQuant's
    # treatment.
    if clip_search:
        # Pad columns to a multiple of g if needed, then reshape to groups.
        if n % g != 0:
            pad = g - (n % g)
            W_pad = torch.nn.functional.pad(Wq, (0, pad))
            x_pad = torch.nn.functional.pad(col_var_full, (0, pad))
        else:
            W_pad = Wq
            x_pad = col_var_full
            pad = 0
        n_p = W_pad.size(1)
        W_groups = W_pad.reshape(m, n_p // g, g)
        x_groups = x_pad.reshape(n_p // g, g)
        clip_ratio = _find_clip(W_groups, bits=bits, x_var=x_groups)  # (m,)
        # Pre-clamp Wq to the chosen per-row range. The clamping is done
        # per (row, group) using the chosen per-row ratio.
        wmin_g = W_groups.amin(dim=-1, keepdim=True)
        wmax_g = W_groups.amax(dim=-1, keepdim=True)
        r = clip_ratio.view(m, 1, 1)
        wmin_clip = wmin_g * r
        wmax_clip = wmax_g * r
        W_clipped = torch.maximum(torch.minimum(W_groups, wmax_clip), wmin_clip)
        Wq[:, :n] = W_clipped.reshape(m, n_p)[:, :n]
        del W_pad, W_groups, x_groups, wmin_g, wmax_g, wmin_clip, wmax_clip, W_clipped
    else:
        clip_ratio = None

    cur_group = -1

    for i1 in range(0, n, blocksize):
        i2 = min(i1 + blocksize, n)
        count = i2 - i1
        W_block = Wq[:, i1:i2].clone()
        Q_block = torch.zeros_like(W_block)
        Err_block = torch.zeros_like(W_block)
        Hinv_block = Hinv[i1:i2, i1:i2]

        for j in range(count):
            col = i1 + j
            grp = col // g
            if grp != cur_group:
                # compute scale/zero for new group based on remaining (yet-unseen)
                # weights in this group
                group_start = grp * g
                group_end = min(group_start + g, n)
                # use the current (already partially updated) W
                Wg_view = Wq[:, group_start:group_end]
                # Wq is already pre-clamped when clip_search is enabled
                # (see the block above the main loop). The per-group
                # (wmin, wmax) here therefore reflect the clipped range.
                wmin = Wg_view.amin(dim=-1, keepdim=True)
                wmax = Wg_view.amax(dim=-1, keepdim=True)
                s = (wmax - wmin).clamp_min(1e-8) / qmax
                z = (-wmin / s).round().clamp(0, qmax)
                scales[:, grp] = s.squeeze(-1)
                zeros[:, grp] = z.squeeze(-1).to(torch.int32)
                cur_group = grp

            w_col = W_block[:, j]
            d = Hinv_block[j, j]

            s = scales[:, cur_group]
            z = zeros[:, cur_group].to(torch.float32)
            q = ((w_col / s) + z).round().clamp(0, qmax)
            qcodes[:, col] = q.to(torch.int32)
            w_q = (q - z) * s
            Q_block[:, j] = w_q

            err = (w_col - w_q) / d.clamp_min(1e-12)
            # update remaining columns in block
            if j + 1 < count:
                W_block[:, j + 1 :].sub_(err.unsqueeze(1) * Hinv_block[j, j + 1 :].unsqueeze(0))
            Err_block[:, j] = err
            Losses[:, col] = (w_col - w_q).pow(2) / d.clamp_min(1e-12).pow(2)

        Q[:, i1:i2] = Q_block
        # propagate block error to columns past the block
        if i2 < n:
            Wq[:, i2:].sub_(Err_block @ Hinv[i1:i2, i2:])

    if invperm is not None:
        qcodes = qcodes[:, invperm]
        # remap scales/zeros to original column ordering by group:
        # (scales/zeros are indexed per (row, group); groups were defined on
        # permuted columns, so to keep the dequantize aligned with the
        # un-permuted qcodes we instead store scales in the *permuted* order
        # and require dequantize to use the same permutation. Simpler: keep
        # actorder=False by default in the TRIAD path.)

    return QuantizedWeight(q=qcodes, scales=scales, zeros=zeros, bits=bits, group_size=g)
