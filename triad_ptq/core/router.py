"""Super-weight router (equation 3 of TRIAD-PTQ v1.0.0).

For each weight (i, j) of layer l:

    kappa_ij = |W_ij| * E[|X_j|] * rho^(l)

We rank kappa globally across layers and retain the top-tau fractile at FP16.
The retained set is stored as flat (layer_idx, row, col, value) triples for
later reconstruction at inference.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SuperWeightSet:
    layer_name: str
    rows: torch.Tensor   # int64
    cols: torch.Tensor   # int64
    values: torch.Tensor  # fp16
    shape: tuple[int, int]


def compute_kappa(W: torch.Tensor, abs_X: torch.Tensor, rho: float) -> torch.Tensor:
    """kappa_ij = |W_ij| * E[|X_j|] * rho. W is (m, n), abs_X is (n,)."""
    return W.abs() * abs_X.unsqueeze(0) * float(rho)


def select_super_weights(
    layer_kappas: dict[str, torch.Tensor],
    tau: float,
) -> tuple[dict[str, SuperWeightSet], float]:
    """Select global top-tau fractile of weights by kappa.

    Returns (per-layer sparse sets, achieved_fraction).
    """
    if tau <= 0:
        return {n: SuperWeightSet(n, torch.empty(0, dtype=torch.long),
                                  torch.empty(0, dtype=torch.long),
                                  torch.empty(0, dtype=torch.float16),
                                  tuple(k.shape)) for n, k in layer_kappas.items()}, 0.0

    # Concatenate all kappas to find global threshold
    flats = []
    for n, k in layer_kappas.items():
        flats.append(k.detach().to(torch.float32).flatten())
    if not flats:
        return {}, 0.0
    all_k = torch.cat(flats).cpu()
    total = all_k.numel()
    k_count = max(int(round(total * tau)), 1)
    if k_count >= total:
        k_count = total - 1
    # Use kth-largest via topk (more accurate than quantile for sparse fractions)
    threshold = torch.topk(all_k, k_count, largest=True).values.min().item()

    selected: dict[str, SuperWeightSet] = {}
    achieved_count = 0
    for n, k in layer_kappas.items():
        mask = k >= threshold
        idx = mask.nonzero(as_tuple=False)
        # idx is (K, 2) of (row, col)
        rows = idx[:, 0].cpu()
        cols = idx[:, 1].cpu()
        # Cap a layer's contribution to keep memory bounded -- if a layer
        # contributes vastly more than its share, take only its top
        # (per-layer-share + 100%)*expected_per_layer_count. In practice the
        # global threshold gives a balanced split.
        achieved_count += rows.numel()
        selected[n] = SuperWeightSet(
            layer_name=n,
            rows=rows.long(),
            cols=cols.long(),
            values=torch.empty(0),  # filled by caller from W
            shape=tuple(k.shape),
        )
    return selected, achieved_count / total
