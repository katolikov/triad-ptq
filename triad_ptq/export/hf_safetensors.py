"""Materialise a TRIAD-quantized model as Hugging Face safetensors.

Produces a directory that *looks like* the original HF TinyLlama (or any
Llama-derivative) snapshot, but with `model.safetensors` rebuilt from
TRIAD's int4 codes folded back to dense fp16 (with U/Lam and sparse
super-weights folded in). This is the input to `mlc_llm convert_weight`.

See ADR-004 for why we go via HF safetensors instead of writing MLC's
canonical layout directly.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ..core.modules import TriadConv2d, TriadLinear
from ..core.quantize import _dequantize


def _materialise_layer(mod: TriadLinear) -> torch.Tensor:
    """Reconstruct the deployment-side dense weight (m, n) for a TriadLinear.

    Order matches `triad_ptq.export.mlc._materialise_layer_fp32`:
      1. dequantize: W_dq = (q - zero) * scale
      2. add stored super-weight residuals (in W_prime basis)
      3. fold U / Lam_pow_beta into W -- runtime executes plain x @ W^T
    """
    W_dq = _dequantize(
        mod.q, mod.scales.to(torch.float32), mod.zeros, mod.group_size
    )
    if mod.sw_rows is not None and mod.sw_rows.numel() > 0:
        r = mod.sw_rows.to(W_dq.device).long()
        c = mod.sw_cols.to(W_dq.device).long()
        v = mod.sw_vals.to(W_dq.device).to(W_dq.dtype)
        W_dq = W_dq.clone()
        W_dq[r, c] = W_dq[r, c] + v
    if mod.U is not None and mod.Lam_pow_beta is not None:
        Linv = mod.Lam_pow_beta.to(W_dq.dtype).reciprocal()  # (n,)
        W_dq = (W_dq @ mod.U.to(W_dq.dtype).t()) * Linv.unsqueeze(0)
    return W_dq.contiguous()


def export_triad_to_hf_safetensors(
    model: nn.Module,
    output_dir: str | Path,
    *,
    hf_snapshot_dir: str | Path,
    dtype: torch.dtype = torch.float16,
) -> dict:
    """Write `output_dir/model.safetensors` (TRIAD-folded) plus the
    other files an HF snapshot needs (`config.json`, tokenizer files,
    `generation_config.json`).

    `hf_snapshot_dir` is the source HF snapshot — we copy `config.json`,
    tokenizer files, and `generation_config.json` from there verbatim.
    Only `model.safetensors` is rebuilt.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    snap = Path(hf_snapshot_dir)

    # Copy the unchanged metadata + tokenizer files.
    files_to_copy = [
        "config.json", "generation_config.json",
        "tokenizer.json", "tokenizer.model",
        "tokenizer_config.json", "special_tokens_map.json",
    ]
    for fname in files_to_copy:
        src = snap / fname
        if src.exists():
            shutil.copy2(src, out / fname)

    # Walk the model graph. Build a state_dict whose Linear weights are
    # the TRIAD-folded fp16 matrices, and whose other tensors (embed,
    # norms, lm_head if non-quantized) come straight from the model.
    triad_replaced: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for name, mod in model.named_modules():
        if isinstance(mod, TriadConv2d):
            skipped.append(name)
            continue
        if not isinstance(mod, TriadLinear):
            continue
        with torch.no_grad():
            W = _materialise_layer(mod).to(dtype).cpu()
        triad_replaced[f"{name}.weight"] = W
        if getattr(mod, "bias", None) is not None:
            triad_replaced[f"{name}.bias"] = (
                mod.bias.detach().to(dtype).cpu().contiguous()
            )

    sd_full: dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        # Skip the TriadLinear-internal buffers; they are folded into
        # `triad_replaced` already.
        if any(k.startswith(prefix + ".") and k.split(".")[-1] in
               {"q", "scales", "zeros", "U", "Lam_pow_beta",
                "sw_rows", "sw_cols", "sw_vals"}
               for prefix in (n.rsplit(".", 1)[0] for n in triad_replaced)):
            continue
        # Drop TriadLinear suffix tensors regardless of prefix
        suffix = k.rsplit(".", 1)[-1]
        if suffix in {"q", "scales", "zeros", "U", "Lam_pow_beta",
                      "sw_rows", "sw_cols", "sw_vals", "bits",
                      "group_size", "_Wcache"}:
            continue
        if not isinstance(v, torch.Tensor):
            continue
        sd_full[k] = v.detach().to(dtype).cpu().contiguous()

    sd_full.update(triad_replaced)

    # Write safetensors. We use the `safetensors` package which the
    # transformers stack already pulls in.
    from safetensors.torch import save_file
    save_file(sd_full, str(out / "model.safetensors"), metadata={"format": "pt"})

    return {
        "output_dir": str(out),
        "n_triad_replaced": len(triad_replaced),
        "n_total_tensors": len(sd_full),
        "n_skipped_conv2d": len(skipped),
        "skipped_conv2d": skipped,
        "model_safetensors_size_mb": (out / "model.safetensors").stat().st_size / 1e6,
        "dtype": str(dtype).split(".")[-1],
    }
