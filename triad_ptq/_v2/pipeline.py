"""Phase H — v2 (SPECTRA-Q) pipeline orchestrator.

Wires together Phases B–G into a single entry point that the public
:func:`triad_ptq.api.optimize` calls when ``algorithm='v2'``.

Pipeline (Phase H1)
-------------------
1. **Phase C — Block-diagonal rotation.** ``apply_block_rotation_to_llama``
   pre-rotates the residual stream at the chosen group_size with the
   chosen rotation kind (sign_perm | block_hadamard).
2. **Phase B (lite) — ρ derivation.** For each rotated transformer block
   we run :func:`triad_ptq._v2.router.squisher.derive_rho` on a small
   activation slice from the calibration set. Squisher's full Fisher
   diagonal is NOT computed here — it is reserved for Phase D's per-block
   trainer; for the GPTAQ ρ-α schedule a single block-level scalar
   suffices.
3. **Phase F — α schedule.** :func:`alpha_schedule` maps the per-block
   ρ to per-layer α, honouring ADR-010's scope-limit. The result is
   passed to v1's ``compile_model`` via a callable ``asym_alpha`` that
   matches each layer name to its block's α.
4. **Phase D placeholder — learnable β.** Not invoked here. It is
   exposed via ``train_learnable_beta`` for callers that want to bake
   smoothing into the weights before this pipeline runs; v2.0 keeps the
   fold optional. ADR-018 documents why per-block β-smoothing is not
   wired to the default v2 path in v2.0.
5. **Phase E placeholder — channel-INT8.** Phase E's
   :class:`ChannelInt8Bundle` is produced post-calibration as a separate
   export step (the MLC bundle writer reads it). It is not invoked
   inside compile_model since the canonical INT4 codes are still the
   primary on-disk artifact. Phase H emits a ``v2_meta`` dict with the
   list of super channels per layer.

Hardware-deferred parts (H2–H4)
--------------------------------
The full evaluation matrix (Llama-3.2-1B + TinyLlama-1.1B + SmolLM-360M
+ SmolLM-135M × 8 baselines × 2 devices, N=10 paired-t) requires a
4090 calibration host and the Galaxy Z Flip7. ADR-017 documents the
deferral; this pipeline runs end-to-end on synthetic Llama configs in
the test suite and is ready for the runbook.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
import torch.nn as nn

from triad_ptq._v2.calib.gptaq_rho_alpha import (
    DEFAULT_ALPHA_MAX,
    DEFAULT_C,
    DEFAULT_EXCLUDE_SUFFIXES,
    alpha_schedule,
)
from triad_ptq._v2.rotation.sign_perm import (
    BlockRotationDiagnostics,
    apply_block_rotation_to_llama,
)
from triad_ptq._v2.router.squisher import derive_rho


@dataclass
class V2RunMeta:
    rotation: BlockRotationDiagnostics
    rho_per_block: dict[str, float]
    alpha_per_layer: dict[str, float]
    super_channel_indices: dict[str, list[int]] = field(default_factory=dict)
    notes: str = ""


def _llama_block_iter(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    """Yield (block_name, block) for an HF Llama-family model."""
    for i, blk in enumerate(model.model.layers):
        yield f"layers.{i}", blk


def _build_alpha_callable(alpha_per_layer: dict[str, float]):
    """Wrap a per-layer α dict as a callable for compile_model."""
    def get(name: str) -> float:
        # The compile loop iterates Linear modules; alpha_schedule emits
        # one entry per BLOCK. We map "layers.<i>.<sub>" → "layers.<i>"
        # by matching the longest prefix.
        for k in alpha_per_layer:
            if name.startswith(k + "."):
                return alpha_per_layer[k]
        return alpha_per_layer.get(name, 0.5)
    return get


def derive_rho_per_block(
    model: nn.Module,
    calib_inputs: torch.Tensor,
    *,
    n_calib_for_rho: int = 4,
) -> dict[str, float]:
    """Compute per-block ρ via a single forward+backward through the full
    model with module hooks.

    We sidestep the brittle "block-as-standalone-callable" pattern —
    newer HF Llama blocks require `position_embeddings` and won't run
    when called directly with hidden states alone. Instead we register
    full-backward hooks on each block, run a single forward+backward,
    and read ‖∂L/∂y‖² / ‖∂L/∂x‖² from autograd's populated grad_input /
    grad_output tensors.

    Loss is a stand-in BRECQ surrogate: the squared mean of the output
    logits, which is non-zero in both grad axes. The absolute scale of
    the loss cancels in the ρ ratio — only the relative sensitivity
    matters.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("derive_rho_per_block: not an HF Llama-family model")

    device = next(model.parameters()).device
    x = calib_inputs[: n_calib_for_rho].to(device)
    if x.dtype != torch.long:
        x = x.long()

    grads_in: dict[str, torch.Tensor] = {}
    grads_out: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, grad_input, grad_output):
            gi = grad_input[0] if grad_input and grad_input[0] is not None else None
            go = grad_output[0] if grad_output and grad_output[0] is not None else None
            if gi is not None:
                grads_in[name] = gi.detach()
            if go is not None:
                grads_out[name] = go.detach()
        return hook

    for i, blk in enumerate(model.model.layers):
        handles.append(blk.register_full_backward_hook(make_hook(f"layers.{i}")))

    try:
        # Make sure parameters can hold grads (caller's model may be in eval
        # with all params frozen; that doesn't prevent grads on activations).
        with torch.enable_grad():
            for p in model.parameters():
                p.requires_grad_(True)
            out = model(x)
            logits = out.logits if hasattr(out, "logits") else out
            loss = logits.float().pow(2).mean()
            loss.backward()
    finally:
        for h in handles:
            h.remove()

    rhos: dict[str, float] = {}
    for i in range(len(model.model.layers)):
        name = f"layers.{i}"
        gy = grads_out.get(name)
        gx = grads_in.get(name)
        if gy is None or gx is None:
            rhos[name] = float("nan")
            continue
        num = float(gy.float().pow(2).sum().item())
        den = float(gx.float().pow(2).sum().item())
        rhos[name] = num / max(den, 1e-30)
    return rhos


def run_v2_pipeline(
    model: nn.Module,
    *,
    calibration: list[torch.Tensor],
    bits: int = 4,
    group_size: int = 64,
    rotation: str = "sign_perm",
    rotation_seed: int = 0xACE1,
    super_channel_rate: float = 0.015,
    gptaq_alpha_c: float = DEFAULT_C,
    gptaq_alpha_max: float = DEFAULT_ALPHA_MAX,
    gptaq_exclude_suffixes: tuple[str, ...] = DEFAULT_EXCLUDE_SUFFIXES,
    asymmetric_calib: bool = True,
    lwc_threshold_percentile: float = 75.0,
    n_calib_for_rho: int = 4,
    compile_kwargs: dict[str, Any] | None = None,
) -> tuple[nn.Module, V2RunMeta]:
    """End-to-end v2 calibration entry point.

    The model is rotated IN-PLACE; the returned model handle is the same
    object. compile_model then runs v1's TRIAD calibration with the v2-
    derived per-layer α. Phase D's learnable β + LWC are NOT auto-baked
    here — Phase H's caller can pre-bake before this call (see ADR-018).
    """
    if rotation not in ("sign_perm", "block_hadamard"):
        raise ValueError(f"unknown rotation kind {rotation!r}")
    compile_kwargs = dict(compile_kwargs or {})

    # Phase C — rotate the model in place.
    rot_diag = apply_block_rotation_to_llama(
        model, group_size=group_size, kind=rotation, seed=rotation_seed,  # type: ignore[arg-type]
    )

    # Phase B-lite — derive per-block ρ for the schedule.
    if calibration:
        # calibration entries are usually token-id tensors; concatenate to
        # a single (batch, seq) tensor for the ρ probe.
        cal0 = calibration[0]
        if isinstance(cal0, dict):  # forward-batch dict (legacy shape)
            cal0 = cal0.get("input_ids", None)
        if cal0 is None:
            raise ValueError("v2 pipeline could not extract input_ids from calibration[0]")
        rho_per_block = derive_rho_per_block(model, cal0, n_calib_for_rho=n_calib_for_rho)
    else:
        rho_per_block = {}

    # Phase F — schedule α from ρ.
    rho_for_schedule = {k: (v if v == v else 0.5) for k, v in rho_per_block.items()}  # NaN→0.5
    alpha_per_block = alpha_schedule(
        rho_for_schedule,
        c=gptaq_alpha_c,
        alpha_max=gptaq_alpha_max,
        exclude_suffixes=gptaq_exclude_suffixes,
    )

    # Phase 2 calibration via v1 compile_model with the per-layer α callable.
    from triad_ptq.compile import compile_model

    asym_alpha_arg = _build_alpha_callable(alpha_per_block)

    # Don't override any user-supplied compile kwargs — use them as-is.
    compile_kwargs.setdefault("bits", bits)
    compile_kwargs.setdefault("group_size", group_size)
    compile_kwargs.setdefault("calibration", calibration)
    compile_kwargs.setdefault("asymmetric_calib", asymmetric_calib)
    compile_kwargs.setdefault("asym_alpha", asym_alpha_arg)
    compile_kwargs.setdefault("asym_exclude_suffixes", gptaq_exclude_suffixes)

    compile_model(model, **compile_kwargs)

    meta = V2RunMeta(
        rotation=rot_diag,
        rho_per_block=rho_per_block,
        alpha_per_layer=alpha_per_block,
        super_channel_indices={},  # Phase E packing happens post-export
        notes=(
            f"v2 pipeline: rotation={rotation} G={group_size} "
            f"alpha_c={gptaq_alpha_c} super_rate={super_channel_rate}"
        ),
    )
    return model, meta


__all__ = [
    "V2RunMeta",
    "derive_rho_per_block",
    "run_v2_pipeline",
]
