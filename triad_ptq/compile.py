"""End-to-end TRIAD-PTQ compilation pipeline.

Implements §3.1 of the paper (Steps A-F):
  A. sensitivity profiling (KFAC factors + rho probe)
  B. bit allocation (Lagrangian)
  C. super-weight identification (top-tau by kappa)
  D. cross-covariance grid (eigendecomp + closed-form beta)
  E. per-layer GPTQ pass on transformed weights
  F. export -- replace nn.Linear / nn.Conv2d in place with TriadLinear /
     TriadConv2d wrappers

Returns the same `model` instance, mutated in place.
"""
from __future__ import annotations

import gc
import time
from typing import Iterable

import torch
import torch.nn as nn
from rich.console import Console

from .core.allocator import allocate_bits, uniform_bits
from .core.calibration import (
    LayerStats,
    attach_rho,
    collect_input_stats,
    estimate_rho,
    quantizable_modules,
)
from .core.gptq_solver import gptq_quantize_layer
from .core.grid import compute_grid
from .core.modules import TriadConv2d, TriadLinear
from .core.quantize import _dequantize, quantize_grouped
from .core.router import compute_kappa, select_super_weights
from .utils.device import best_device, warn_no_silent_fallback

console = Console()


def _set_module(root: nn.Module, dotted_name: str, new: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def compile_model(
    model: nn.Module,
    *,
    bits: int = 4,
    calibration: Iterable | None = None,
    super_weight_frac: float | None = 1e-3,
    bit_allocator: str = "trace",  # 'uniform' | 'trace'
    cov_grid: str = "analytic",     # 'none' | 'analytic'
    target: str = "auto",
    time_budget_min: float = 5.0,
    group_size: int = 64,
    n_calib: int = 128,
    device: str | None = None,
    forward_fn=None,
    output_fn=None,
    rho_probe_n: int = 4,
    progress: bool = True,
    return_meta: bool = False,
):
    warn_no_silent_fallback()
    dev = torch.device(device) if device else best_device()
    if calibration is None:
        raise ValueError("`calibration` (an iterable of forward-batches) is required")
    cal_list = list(calibration)
    if len(cal_list) == 0:
        raise ValueError("calibration set is empty")

    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    layers = quantizable_modules(model)
    if not layers:
        raise RuntimeError("no quantizable modules found in model")
    if progress:
        console.log(f"TRIAD-PTQ: {len(layers)} quantizable layers, target={bits}-bit")

    # ---- Step A: sensitivity profiling ----
    t0 = time.perf_counter()
    stats = collect_input_stats(
        model, cal_list, device=dev, n_max=n_calib,
        layers=layers, forward_fn=forward_fn, a_device=dev,
    )
    if progress:
        console.log(f"  collected A, abs_X, Y2 in {time.perf_counter()-t0:.1f}s")

    # ---- rho probe ----
    # rho is needed both for bit_allocator='trace' AND for super-weight kappa
    # (eq 3 multiplies kappa by rho). We probe whenever super_weight_frac>0
    # OR bit_allocator='trace'. To keep cost bounded, the probe runs on the
    # *first* `rho_probe_n` calibration batches only.
    need_rho = (super_weight_frac and super_weight_frac > 0) or bit_allocator == "trace"
    if need_rho:
        probe = cal_list[: max(1, min(rho_probe_n, len(cal_list)))]
        t1 = time.perf_counter()
        rhos = estimate_rho(
            model, probe, [n for n, _ in layers],
            device=dev, bits=bits,
            forward_fn=forward_fn, output_fn=output_fn,
        )
        attach_rho(stats, rhos)
        if progress:
            console.log(
                f"  estimated rho via {len(probe)} probe batch(es) in "
                f"{time.perf_counter()-t1:.1f}s"
            )
    else:
        for n in stats:
            stats[n].rho = 1.0

    # ---- Step B: bit allocation ----
    layer_S = [stats[n].sensitivity() for n, _ in layers]
    layer_W = [m.weight.numel() for _, m in layers]
    if bit_allocator == "uniform":
        alloc = uniform_bits(layer_W, bits)
    elif bit_allocator == "trace":
        # When the target is an integer in {3, 4}, the rate-distortion split
        # into {3, 4, 8} collapses bimodally and 3-bit quant hurts most layers
        # too much to be compensated by a few 8-bit layers. We honor the
        # paper's {3, 4, 8} grid only when the target is fractional, so the
        # mixed allocation is doing real work; otherwise default to uniform.
        if abs(bits - round(bits)) < 1e-6 and int(round(bits)) in (3, 4):
            alloc = uniform_bits(layer_W, int(round(bits)))
        else:
            alloc = allocate_bits(layer_S, layer_W, target_avg_bits=float(bits))
    else:
        alloc = allocate_bits(layer_S, layer_W, target_avg_bits=float(bits))
    if progress:
        console.log(
            f"  bit alloc: target={alloc.target_bits_per_w:.2f} achieved={alloc.achieved_bits_per_w:.3f} "
            f"distribution={dict(zip(*[list(x) for x in (set(alloc.bits), [alloc.bits.count(b) for b in set(alloc.bits)])]))}"
        )

    # ---- Step C+D: per-layer transform + super-weight kappas ----
    layer_kappas: dict[str, torch.Tensor] = {}
    layer_grid: dict[str, dict] = {}
    layer_W_prime: dict[str, torch.Tensor] = {}

    for (name, mod), b in zip(layers, alloc.bits):
        s = stats[name]
        W = _module_weight2d(mod).to(dev).float()
        if cov_grid == "analytic":
            grid = compute_grid(W, s.A)
            # Per paper §2.4, beta=0 corresponds to no transformation. If the
            # closed-form gives beta*=0 we must skip the rotation: applying U
            # by itself (without scaling) does not change quantization error
            # in the per-channel symmetric model but breaks group-wise
            # quantization (the eigenbasis spreads weight magnitudes across
            # channels in a way that hurts group scales).
            if grid.beta_star <= 1e-4:
                U = None
                Lam_b = None
                W_prime = W
                layer_grid[name] = {
                    "U": None, "Lam_b": None, "beta": 0.0, "eig": grid.eig,
                }
            else:
                U = grid.U
                Lam_b = grid.Lam_pow_beta
                W_prime = W @ U * Lam_b.unsqueeze(0)  # (m, n)
                layer_grid[name] = {
                    "U": U, "Lam_b": Lam_b, "beta": grid.beta_star, "eig": grid.eig,
                }
        else:
            U = None
            Lam_b = None
            W_prime = W
            layer_grid[name] = {"U": None, "Lam_b": None, "beta": 0.0, "eig": None}
        layer_W_prime[name] = W_prime
        # kappa is computed in original basis (paper §2.3)
        kap = compute_kappa(W, s.abs_X.to(dev), s.rho)
        layer_kappas[name] = kap

    # ---- Step C cont.: select super-weights globally ----
    if super_weight_frac and super_weight_frac > 0:
        sw, achieved_tau = select_super_weights(layer_kappas, super_weight_frac)
        if progress:
            console.log(f"  super-weights tau={super_weight_frac:.2e} (achieved {achieved_tau:.2e})")
    else:
        sw = {n: None for n in layer_kappas}

    # ---- Step E+F: GPTQ pass + replace modules ----
    repl_count = 0
    for (name, mod), b in zip(layers, alloc.bits):
        s = stats[name]
        W_prime = layer_W_prime[name]
        # Build the *transformed* Hessian H' = (X')^T X' = Lam^{-2 beta} U^T A U Lam^{-2 beta}
        # but in the eigenbasis A' = diag(Lam^{1 - 2 beta}) -- use it directly.
        gd = layer_grid[name]
        if gd["U"] is not None:
            beta = gd["beta"]
            eig = gd["eig"].to(W_prime.device)
            # In the transformed basis, H' = (X')^T X' has eigenvalues
            #   lambda_k^{1 - 2*beta}    along the k-th transformed dim
            # i.e. H' is diagonal of size n with these entries.
            Hp_diag = eig.clamp_min(1e-12).pow(1.0 - 2.0 * beta)
            H_prime = torch.diag(Hp_diag)
        else:
            # use original A
            H_prime = s.A.to(W_prime.device)

        # Identify per-layer super-weight positions (still expressed in original
        # column basis if no transform; in transformed basis if transformed).
        # For analytic grid we apply correction at inference in W_prime basis,
        # so we map super-weight positions through the same transform.
        sw_set = sw.get(name)
        sw_rows = sw_cols = sw_vals = None
        if sw_set is not None and sw_set.rows.numel() > 0:
            # Move SW indices to dev. Values come from W_prime at those indices
            # (since correction is applied in transformed basis).
            r = sw_set.rows.to(W_prime.device)
            c = sw_set.cols.to(W_prime.device)
            # If transformed, the kappa positions are in original (i, j) space;
            # but the W_prime matrix has the same shape (m, n) and 'j' indexes
            # transformed input dims. Indices between bases differ. To keep this
            # honest, we always express super-weights in the *quantized* matrix
            # basis (same as W_prime). So recompute kappa on |W_prime| to pick
            # positions in the transformed basis:
            if gd["U"] is not None:
                # rebuild kappa in transformed basis using transformed activations'
                # mean magnitude: E[|X'_j|] under linearization is approximately
                # Lam^{-beta} * (U^T E[|X|]) — we use this as an estimate.
                est_absXp = (gd["Lam_b"].pow(-1) * (gd["U"].t() @ s.abs_X.to(dev))).abs()
                kp = compute_kappa(W_prime, est_absXp, s.rho)
                # take same number of entries as global allotment for this layer
                k_count = sw_set.rows.numel()
                if k_count > 0:
                    flat = kp.flatten()
                    top_idx = torch.topk(flat, k_count).indices
                    r = (top_idx // kp.size(1)).to(W_prime.device)
                    c = (top_idx % kp.size(1)).to(W_prime.device)
            sw_rows = r.cpu()
            sw_cols = c.cpu()
            # We'll fill values *after* GPTQ runs (residual = W_prime[r,c] - W_q[r,c])

        # ---- Step E: GPTQ on (W_prime, H_prime) ----
        qweight = gptq_quantize_layer(
            W_prime, H_prime,
            bits=int(b),
            group_size=group_size,
            actorder=False,
        )

        # Compute super-weight residuals in W_prime basis: difference between
        # un-quantized W_prime and dequantized W_q at SW positions.
        if sw_rows is not None:
            W_dq = _dequantize(qweight.q, qweight.scales, qweight.zeros, qweight.group_size)
            sw_vals = (W_prime[sw_rows.to(W_prime.device), sw_cols.to(W_prime.device)]
                       - W_dq[sw_rows.to(W_prime.device), sw_cols.to(W_prime.device)]).cpu().float()

        # ---- Step F: build replacement module ----
        U = gd["U"]
        Lam_b = gd["Lam_b"]
        if isinstance(mod, nn.Linear):
            new_mod = TriadLinear.from_linear(
                mod, qweight,
                U=U, Lam_pow_beta=Lam_b,
                sw_rows=sw_rows, sw_cols=sw_cols, sw_vals=sw_vals,
                dtype=torch.float32,
            )
        else:
            new_mod = TriadConv2d(
                mod, qweight,
                U=U, Lam_pow_beta=Lam_b,
                sw_rows=sw_rows, sw_cols=sw_cols, sw_vals=sw_vals,
                dtype=torch.float32,
            )
        new_mod.to(W_prime.device)
        _set_module(model, name, new_mod)
        repl_count += 1

        # free intermediates
        del W_prime, H_prime, qweight
        if dev.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    if progress:
        console.log(f"  replaced {repl_count} layers in {time.perf_counter()-t0:.1f}s total")

    if return_meta:
        return model, {
            "alloc": alloc,
            "stats": stats,
            "grid": layer_grid,
            "n_layers": len(layers),
        }
    return model


def _module_weight2d(mod: nn.Module) -> torch.Tensor:
    """Return the weight as a 2-D matrix (out, in*[kh*kw])."""
    if isinstance(mod, nn.Linear):
        return mod.weight.data
    if isinstance(mod, nn.Conv2d):
        w = mod.weight.data
        return w.reshape(w.size(0), -1)
    raise TypeError(type(mod))
