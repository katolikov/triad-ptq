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

Memory layout (`a_device='cpu'`, the low-mem path used for >=1B params on
M1 8 GB):

  - Stage A allocates one Gram matrix `A` per quantizable layer on `a_device`
    (host RAM). For TinyLlama-1.1B this is ~5 GB on CPU but never lives on
    MPS simultaneously.
  - Stage B is scalar-only.
  - Stage C runs a single per-layer top-K kappa pass; it transiently
    materialises one (m, n) tensor on `dev` (peak ~46 MB for TinyLlama
    down_proj) and discards it before the next layer.
  - Stage D+E+F fuse into a single per-layer streaming loop. Each iteration:
      1. Pull A_l from `a_device` to `dev`.
      2. eigh A_l on CPU -> U_l, eig_l (CPU, fp64) -> move to dev as fp32.
      3. Compute beta*, Lam_b, W_prime, H_prime.
      4. GPTQ -> qweight.
      5. Build TriadLinear/TriadConv2d, place on dev, replace in model.
      6. del A_l, U_l, eig_l, W_prime, H_prime; empty_cache().
    Peak transient is dominated by U (n^2) and W_prime (m*n); both freed
    before the next layer.
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
    _default_forward,
    attach_rho,
    collect_input_stats,
    estimate_rho,
    quantizable_modules,
)
from .core.gptq_solver import gptq_quantize_layer
from .core.gptaq_asym import asymmetric_transfer, asymmetry_strength
from .core.gptaq_capture import collect_layer_grams
from .core.grid import compute_grid
from .core.modules import TriadConv2d, TriadLinear
from .core.quantize import _dequantize, quantize_grouped
from .core.router import SuperWeightSet, compute_kappa, compute_kappa_topk
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


def _module_weight2d(mod: nn.Module) -> torch.Tensor:
    """Return the weight as a 2-D matrix (out, in*[kh*kw])."""
    if isinstance(mod, nn.Linear):
        return mod.weight.data
    if isinstance(mod, nn.Conv2d):
        w = mod.weight.data
        return w.reshape(w.size(0), -1)
    raise TypeError(type(mod))


def _empty_cache(dev: torch.device) -> None:
    if dev.type == "mps":
        torch.mps.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


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
    a_device: str | None = None,
    forward_fn=None,
    output_fn=None,
    rho_probe_n: int = 4,
    progress: bool = True,
    return_meta: bool = False,
    clip_search: bool = False,
    asymmetric_calib: bool = False,
):
    warn_no_silent_fallback()
    dev = torch.device(device) if device else best_device()
    a_dev = torch.device(a_device) if a_device else dev
    if calibration is None:
        raise ValueError("`calibration` (an iterable of forward-batches) is required")
    cal_list = list(calibration)
    if len(cal_list) == 0:
        raise ValueError("calibration set is empty")

    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # GPTAQ Phase-2: keep an immutable FP16 deepcopy of the original model so
    # we can compute per-layer cross-Grams C = E[X̃ᵀ X_post] during the
    # streaming GPTQ loop. We do this BEFORE any layer replacement.
    model_fp16_ref: nn.Module | None = None
    if asymmetric_calib:
        import copy as _copy
        if progress:
            console.log("  asymmetric_calib=True → cloning FP16 reference model "
                        "(memory ≈ 2× model size)")
        model_fp16_ref = _copy.deepcopy(model)
        model_fp16_ref.eval()
        for p in model_fp16_ref.parameters():
            p.requires_grad_(False)
        # FP16 reference can stay on the same compute device — it just runs
        # forward passes alongside the rolling-quant model.
        model_fp16_ref.to(dev)

    layers = quantizable_modules(model)
    if not layers:
        raise RuntimeError("no quantizable modules found in model")
    if progress:
        console.log(
            f"TRIAD-PTQ: {len(layers)} quantizable layers, target={bits}-bit, "
            f"compute_dev={dev}, gram_dev={a_dev}"
        )

    # ---- Step A: sensitivity profiling ----
    t0 = time.perf_counter()
    stats = collect_input_stats(
        model, cal_list, device=dev, n_max=n_calib,
        layers=layers, forward_fn=forward_fn, a_device=a_dev,
    )
    if progress:
        console.log(f"  collected A, abs_X, Y2 in {time.perf_counter()-t0:.1f}s")

    # ---- rho probe ----
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
        if abs(bits - round(bits)) < 1e-6 and int(round(bits)) in (3, 4):
            alloc = uniform_bits(layer_W, int(round(bits)))
        else:
            alloc = allocate_bits(layer_S, layer_W, target_avg_bits=float(bits))
    else:
        alloc = allocate_bits(layer_S, layer_W, target_avg_bits=float(bits))
    if progress:
        console.log(
            f"  bit alloc: target={alloc.target_bits_per_w:.2f} "
            f"achieved={alloc.achieved_bits_per_w:.3f} "
            f"distribution={dict(zip(*[list(x) for x in (set(alloc.bits), [alloc.bits.count(b) for b in set(alloc.bits)])]))}"
        )

    # ---- Step C (pass 1): streaming top-K kappa for global super-weight pick ----
    sw_global: dict[str, SuperWeightSet] = {}
    if super_weight_frac and super_weight_frac > 0:
        t2 = time.perf_counter()
        total_weights = sum(layer_W)
        target_count = max(1, int(round(total_weights * float(super_weight_frac))))
        candidates: list[dict] = []
        for (name, mod), b in zip(layers, alloc.bits):
            s = stats[name]
            W_full = _module_weight2d(mod).to(dev).float()
            layer_count = W_full.numel()
            expected_share = max(1, int(round(layer_count * float(super_weight_frac))))
            cap = min(layer_count, max(256, expected_share * 4))
            rows, cols, vals = compute_kappa_topk(
                W_full, s.abs_X.to(dev), s.rho, cap
            )
            candidates.append({
                "name": name,
                "rows": rows.cpu(),
                "cols": cols.cpu(),
                "vals": vals.cpu().float(),
                "shape": tuple(W_full.shape),
            })
            del W_full, rows, cols, vals
            _empty_cache(dev)

        all_vals = torch.cat([c["vals"] for c in candidates])
        k_global = min(target_count, all_vals.numel())
        if k_global > 0:
            threshold = (
                torch.topk(all_vals, k_global, largest=True).values.min().item()
            )
        else:
            threshold = float("inf")
        del all_vals

        achieved_count = 0
        for c in candidates:
            mask = c["vals"] >= threshold
            r = c["rows"][mask].long()
            cc = c["cols"][mask].long()
            sw_global[c["name"]] = SuperWeightSet(
                layer_name=c["name"],
                rows=r,
                cols=cc,
                values=torch.empty(0),
                shape=c["shape"],
            )
            achieved_count += int(r.numel())
        del candidates
        if progress:
            achieved_tau = achieved_count / max(total_weights, 1)
            console.log(
                f"  super-weights tau={super_weight_frac:.2e} "
                f"(achieved {achieved_tau:.2e}) in {time.perf_counter()-t2:.1f}s"
            )
    else:
        for name, _ in layers:
            sw_global[name] = SuperWeightSet(
                layer_name=name,
                rows=torch.empty(0, dtype=torch.long),
                cols=torch.empty(0, dtype=torch.long),
                values=torch.empty(0),
                shape=tuple(_module_weight2d(_).shape) if False else (0, 0),
            )

    # ---- Step D+E+F (pass 2): per-layer streaming GPTQ + module replace ----
    repl_count = 0
    layer_grid_meta: dict[str, dict] = {}
    asym_meta: dict[str, dict] = {}
    t3 = time.perf_counter()
    for (name, mod), b in zip(layers, alloc.bits):
        s = stats[name]
        W = _module_weight2d(mod).to(dev).float()

        # ---- Phase-2 asymmetric transfer (GPTAQ) -------------------------
        # Before TRIAD's basis change, redirect the layer toward
        #   target output = X̃ Wᵀ        (FP16 reference output)
        # while accepting input X = post-quant cascade. The continuous
        # optimum is W_aug = W · Cᵀ · H⁻¹ where C = X̃ᵀ X, H = XᵀX. This
        # commutes with TRIAD's W' = W·U·Λ^β so the rest of the pipeline
        # is unchanged. Convs are flattened to 2-D as elsewhere; for now
        # we only run the transfer on Linears (Conv2d coverage is a
        # follow-up — see ADR-010).
        if asymmetric_calib and isinstance(mod, nn.Linear):
            # Re-collect the per-layer Grams on the *current* model state
            # (rolling-quant for layers 0..l-1, FP16 for the rest, plus
            # the FP16 reference for X̃). One forward-pass through each
            # of two models per layer; cost ~2 × n_calib batches.
            ga = collect_layer_grams(
                model_quant=model,
                model_fp16=model_fp16_ref,
                layer_name=name,
                batches=cal_list,
                device=dev,
                forward_fn=forward_fn or _default_forward,
                a_device=a_dev,
            )
            asym_meta[name] = asymmetry_strength(ga, H_pre=s.A)
            asym_meta[name]["bits"] = int(b)

            W = asymmetric_transfer(W, ga, percdamp=0.01)
            del ga
            _empty_cache(dev)

        # ---- D: grid (analytic eigh + closed-form beta) ------------------
        if cov_grid == "analytic":
            # Bring this one layer's A to compute device. eigh is done on CPU
            # via safe_eigh inside compute_grid (it pulls A.cpu()).
            A_layer = s.A.to(dev)
            grid = compute_grid(W, A_layer)
            del A_layer
            if grid.beta_star <= 1e-4:
                U = None
                Lam_b = None
                W_prime = W
                eig = grid.eig
                grid_info = {"beta": 0.0, "eig": eig.detach().cpu()}
            else:
                U = grid.U
                Lam_b = grid.Lam_pow_beta
                W_prime = W @ U * Lam_b.unsqueeze(0)  # (m, n)
                eig = grid.eig
                grid_info = {"beta": float(grid.beta_star), "eig": eig.detach().cpu()}
            del grid
        else:
            U = None
            Lam_b = None
            W_prime = W
            eig = None
            grid_info = {"beta": 0.0, "eig": None}

        # ---- Build H' ----------------------------------------------------
        if U is not None:
            beta = grid_info["beta"]
            Hp_diag = eig.to(W_prime.device).clamp_min(1e-12).pow(1.0 - 2.0 * beta)
            H_prime = torch.diag(Hp_diag)
            del Hp_diag
        else:
            # use original A (pulled to compute device just for this layer)
            H_prime = s.A.to(dev).float()

        # Free the per-layer A on a_device once we've consumed it. Setting
        # .A to a zero-byte tensor breaks any later access (intentional --
        # it should only be consumed once per layer).
        stats[name].A = torch.empty(0)

        # ---- Super-weight rows/cols (rebase to transformed basis if needed)
        sw_set = sw_global.get(name)
        sw_rows = sw_cols = sw_vals = None
        if sw_set is not None and sw_set.rows.numel() > 0:
            r = sw_set.rows.to(W_prime.device)
            c = sw_set.cols.to(W_prime.device)
            if U is not None:
                # rebuild kappa in transformed basis (paper §2.3, transformed)
                est_absXp = (Lam_b.pow(-1) * (U.t() @ s.abs_X.to(dev))).abs()
                kp = compute_kappa(W_prime, est_absXp, s.rho)
                k_count = sw_set.rows.numel()
                if k_count > 0:
                    flat = kp.flatten()
                    top_idx = torch.topk(flat, min(k_count, flat.numel())).indices
                    # Defensive clamp: on MPS the integer-divide on a large
                    # final-FC layer (lm_head, m=32000) was observed to
                    # produce exactly `m` (one past the last valid row),
                    # crashing the later gather at
                    # W_prime[sw_rows, sw_cols] with
                    # 'index 32000 is out of bounds: 0, range 0 to 32000'.
                    m_p, n_p = W_prime.size(0), W_prime.size(1)
                    r = ((top_idx // kp.size(1)).clamp_(0, m_p - 1)
                         .to(W_prime.device))
                    c = ((top_idx % kp.size(1)).clamp_(0, n_p - 1)
                         .to(W_prime.device))
                    del flat, top_idx
                del kp, est_absXp
            sw_rows = r.cpu()
            sw_cols = c.cpu()

        # ---- E: GPTQ on (W_prime, H_prime) -------------------------------
        qweight = gptq_quantize_layer(
            W_prime, H_prime,
            bits=int(b),
            group_size=group_size,
            actorder=False,
            clip_search=clip_search,
        )
        del H_prime

        # Compute SW residuals in W_prime basis
        if sw_rows is not None and sw_rows.numel() > 0:
            W_dq = _dequantize(qweight.q, qweight.scales, qweight.zeros, qweight.group_size)
            sw_vals = (
                W_prime[sw_rows.to(W_prime.device), sw_cols.to(W_prime.device)]
                - W_dq[sw_rows.to(W_prime.device), sw_cols.to(W_prime.device)]
            ).cpu().float()
            del W_dq

        # ---- F: build replacement module ---------------------------------
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
        new_mod.to(dev)
        _set_module(model, name, new_mod)
        repl_count += 1

        layer_grid_meta[name] = grid_info

        # Free intermediates. U/Lam_b survive only inside the new module
        # buffers; the local references go out of scope here.
        del W, W_prime, qweight, U, Lam_b
        if eig is not None:
            del eig
        _empty_cache(dev)

    if progress:
        console.log(
            f"  replaced {repl_count} layers in "
            f"{time.perf_counter()-t3:.1f}s (total {time.perf_counter()-t0:.1f}s)"
        )

    # Free the FP16 reference now that the streaming loop is done.
    if model_fp16_ref is not None:
        del model_fp16_ref
        _empty_cache(dev)

    if return_meta:
        return model, {
            "alloc": alloc,
            "n_layers": len(layers),
            "grid": layer_grid_meta,
            "super_weights": {n: int(s.rows.numel()) for n, s in sw_global.items()},
            "asymmetric_calib": asymmetric_calib,
            "asymmetry_per_layer": asym_meta,
        }
    return model
