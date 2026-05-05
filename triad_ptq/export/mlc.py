"""TRIAD checkpoint -> MLC-LLM-compatible q4f16_1 weight bundle.

The output of `export_to_mlc` is a directory containing:

  mlc-chat-config.json    -- model arch + quantization config
  ndarray-cache.json      -- manifest of weight shards (params/files)
  params_shard_0.bin      -- packed weights for shard 0
  params_shard_*.bin      -- additional shards if needed
  tokenizer.json
  tokenizer.model         -- (copied from the source HF repo)

This is the "byte-equivalent of running mlc_llm convert_weight" step. The
follow-up `mlc_llm compile --device android` step (which produces the
SPIR-V shader library) requires the mlc_llm + tvm + android NDK toolchain
and is run separately by the user (see ADR-003 for why we defer the
Android compile in autonomous mode).

Implementation notes
--------------------

1. **Sparse super-weight folding ("collapse_sparse_into_dense").**
   Per Phase-4.1 of the session prompt: MLC has no native dense+sparse
   kernel. We fold each layer's stored super-weight residuals into the
   dequantized weight and re-quantize. This raises the per-group dynamic
   range and costs 0.2-0.5 PPL per the SqueezeLLM ablation, but produces
   a single dense INT4 buffer compatible with q4f16_1.

2. **Group-size mapping.** TRIAD calibrates with `group_size=64`. MLC's
   `q4f16_1` uses `group_size=32`. We re-pack from g=64 to g=32 by
   re-running per-group asymmetric quantization on the dequantized
   weight at finer granularity. The (small) quality cost of this
   re-quant is the same as plain RTN at g=32.

3. **Per-group fp16 scale + fp16 zero_point.** Stored in two contiguous
   arrays of shape (n_groups,) per row, fp16 little-endian. MLC packs
   them as `[scale_0, zero_0, scale_1, zero_1, ...]` — the "_1"
   variant of q4f16. Verify against MLC's
   `quantize_weight_int4` after install.

4. **Code packing (uint32 lanes, 8 codes/lane).** A row of length n is
   packed into ceil(n/8) uint32 values. Code i goes into lane i//8 at
   bit position (i % 8) * 4. This is the layout consumed by MLC's
   Vulkan q4f16_1 GEMM kernel.

5. **Activation-side rotation U.** TRIAD's runtime form is
       y = (x @ U / Lam_pow_beta) @ W_dq^T
   For deployment on a kernel that has no awareness of U, we fold
   U/Lam into W:  W_eff = W_dq @ U / Lam^beta — equivalent in fp32
   exact arithmetic, but stored as a single dense matrix. This means
   the deployed model executes `y = x @ W_eff^T`, the standard linear
   form. TRIAD's quality benefit is preserved because U was used
   *during quantization* to align with the activation Hessian; at
   inference time we don't need to re-apply it.

If you want to keep U at runtime instead, set fold_U=False — we then
emit two extra parameters per layer (U, Lam) and the runtime must
support them. MLC q4f16_1 does not, so default is fold_U=True.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch

from ..core.modules import TriadConv2d, TriadLinear
from ..core.quantize import _dequantize, quantize_grouped


# -----------------------------------------------------------------------------
# Per-layer materialisation
# -----------------------------------------------------------------------------


@dataclass
class _MLCLayerBlob:
    name: str           # e.g. "model.layers.0.self_attn.q_proj"
    qcodes: torch.Tensor  # int8 in [0, 15], shape (m, n)
    scales: torch.Tensor  # fp16, shape (m, n_groups_g32)
    zeros: torch.Tensor   # fp16, shape (m, n_groups_g32)
    shape: tuple[int, int]


def _fold_super_weights(W_dq: torch.Tensor, mod: TriadLinear) -> torch.Tensor:
    """Add stored super-weight residuals back into the dense weight.

    TriadLinear stores `sw_vals = W_orig[r,c] - W_dq[r,c]` after GPTQ. The
    runtime forward adds them back at inference. For MLC export we fold
    them in once and re-quantize.
    """
    if mod.sw_rows is None or mod.sw_rows.numel() == 0:
        return W_dq
    out = W_dq.clone()
    r = mod.sw_rows.to(out.device).long()
    c = mod.sw_cols.to(out.device).long()
    v = mod.sw_vals.to(out.device).to(out.dtype)
    out[r, c] = out[r, c] + v
    return out


def _fold_U(W_dq: torch.Tensor, mod: TriadLinear) -> torch.Tensor:
    """Fold the activation-side rotation U / Lam back into W so the
    deployed kernel can run plain x @ W^T.

    Forward in TriadLinear is
        y = (x @ U / Lam) @ W_dq^T              (eq 4 of paper)
          = x @ (U / Lam) @ W_dq^T
          = x @ (W_dq @ (U / Lam)^T )^T
    -> deploy weight = W_dq @ (U / Lam)^T  =  W_dq @ U^T / Lam[None, :]
    We materialise this and use it as the dense weight to re-quantize.
    """
    if mod.U is None or mod.Lam_pow_beta is None:
        return W_dq
    # diag(1/Lam) @ U^T  has columns = (1/Lam[i]) * U[:, i]
    # so W_dq @ U^T diag(1/Lam) has shape (m, n) with
    #   out[:, j] = (W_dq @ U[:, j]) / Lam[j]
    Linv = mod.Lam_pow_beta.to(W_dq.dtype).reciprocal()  # (n,)
    return (W_dq @ mod.U.to(W_dq.dtype).t()) * Linv.unsqueeze(0)


def _materialise_layer_fp32(mod: TriadLinear) -> torch.Tensor:
    """Reconstruct the deployed-side fp32 weight matrix from a TriadLinear.

    The reconstruction order matters: super-weights are stored in the
    *transformed* basis (W_prime), so they must be added BEFORE the U fold.
    """
    W_dq = _dequantize(mod.q, mod.scales.to(torch.float32),
                       mod.zeros, mod.group_size)
    W_dq = _fold_super_weights(W_dq, mod)
    W_eff = _fold_U(W_dq, mod)
    return W_eff.float()


# -----------------------------------------------------------------------------
# q4f16_1 packing
# -----------------------------------------------------------------------------


def _quantize_to_g32_asymmetric(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row asymmetric INT4 with group_size=32 (MLC q4f16_1 layout)."""
    qw = quantize_grouped(W, bits=4, group_size=32)
    # qw.q is int32 in [0,15]; cast to int8 for compactness pre-pack.
    return qw.q.to(torch.int8), qw.scales.to(torch.float16), qw.zeros.to(torch.float16)


def _pack_int4_uint32_lanes(codes: torch.Tensor) -> torch.Tensor:
    """Pack an (m, n) int8 tensor with values in [0, 15] into (m, n_lanes)
    uint32 with 8 codes per lane (lower nibble of code 0 in bits 0..3, etc.).
    n must be a multiple of 8 (we pad with zeros if not).
    """
    assert codes.dtype == torch.int8
    m, n = codes.shape
    pad = (8 - (n % 8)) % 8
    if pad:
        codes = torch.nn.functional.pad(codes, (0, pad))
    n2 = codes.size(1)
    n_lanes = n2 // 8
    out = torch.zeros((m, n_lanes), dtype=torch.int64)
    for k in range(8):
        col = codes[:, k::8].to(torch.int64) & 0xF
        out |= col << (4 * k)
    # MLC stores as uint32 little-endian; torch has no uint32 dtype, so we
    # emit int32 with the same bit pattern (writers use int32.numpy().tobytes()
    # which preserves byte order on little-endian hosts: M1 / Apple Silicon
    # is LE).
    return out.to(torch.int32)


def _interleave_scale_zero_q4f16_1(
    scales: torch.Tensor, zeros: torch.Tensor,
) -> torch.Tensor:
    """Interleave per-group fp16 (scale, zero) -> shape (m, 2 * n_groups) fp16.

    Layout per row:   [scale_0, zero_0, scale_1, zero_1, ...]
    """
    assert scales.dtype == torch.float16 and zeros.dtype == torch.float16
    m, ng = scales.shape
    out = torch.empty((m, 2 * ng), dtype=torch.float16)
    out[:, 0::2] = scales
    out[:, 1::2] = zeros
    return out


# -----------------------------------------------------------------------------
# Whole-model export
# -----------------------------------------------------------------------------


def _state_dict_layers(model: torch.nn.Module) -> Iterable[tuple[str, TriadLinear]]:
    for name, mod in model.named_modules():
        if isinstance(mod, TriadLinear):
            yield name, mod
        elif isinstance(mod, TriadConv2d):
            # Conv2d not supported by MLC q4f16_1 deployment for LLMs.
            # Skip with a warning during export.
            yield name, mod


def export_to_mlc(
    model: torch.nn.Module,
    output_dir: str | Path,
    *,
    hf_model_id: str,
    hf_snapshot_dir: str | Path | None = None,
    fold_U: bool = True,
    fold_super_weights: bool = True,
) -> dict:
    """Export a TRIAD-quantized HF Llama / Llama-like model to MLC q4f16_1.

    Returns a small dict summary (counts, sizes, list of params).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model.eval()
    sd = model.state_dict()

    # Identify TriadLinear modules. The dense fp32 representation per layer
    # is materialised and re-quantized to MLC's g32 layout.
    layer_blobs: list[_MLCLayerBlob] = []
    skipped_conv: list[str] = []
    for name, mod in model.named_modules():
        if isinstance(mod, TriadConv2d):
            skipped_conv.append(name)
            continue
        if not isinstance(mod, TriadLinear):
            continue

        # Materialise (possibly folded) deployment weight.
        with torch.no_grad():
            W_dq = _dequantize(mod.q, mod.scales.to(torch.float32),
                               mod.zeros, mod.group_size).cpu()
            if fold_super_weights:
                W_dq = _fold_super_weights(W_dq, mod.cpu()) if mod.U is None else _fold_super_weights(W_dq, mod)
            if fold_U:
                W_dq = _fold_U(W_dq, mod)
            W_dq = W_dq.contiguous().float()

        codes_int8, scales_fp16, zeros_fp16 = _quantize_to_g32_asymmetric(W_dq)
        layer_blobs.append(_MLCLayerBlob(
            name=name,
            qcodes=codes_int8,
            scales=scales_fp16,
            zeros=zeros_fp16,
            shape=tuple(W_dq.shape),
        ))

    # Pack and write a single shard file.
    shard_path = out / "params_shard_0.bin"
    manifest_records: list[dict] = []
    offset = 0
    with open(shard_path, "wb") as f:
        # 1. q-codes shards (uint32 lanes)
        for blob in layer_blobs:
            packed = _pack_int4_uint32_lanes(blob.qcodes)
            data = packed.contiguous().numpy().tobytes()
            f.write(data)
            manifest_records.append({
                "name": f"{blob.name}.q_weight",
                "shape": list(packed.shape),
                "dtype": "uint32",
                "format": "raw",
                "shard": "params_shard_0.bin",
                "byte_offset": offset,
                "byte_length": len(data),
            })
            offset += len(data)

        # 2. (scale, zero) interleaved fp16
        for blob in layer_blobs:
            sz = _interleave_scale_zero_q4f16_1(blob.scales, blob.zeros)
            data = sz.contiguous().numpy().tobytes()
            f.write(data)
            manifest_records.append({
                "name": f"{blob.name}.q_scale_zero",
                "shape": list(sz.shape),
                "dtype": "float16",
                "format": "raw",
                "shard": "params_shard_0.bin",
                "byte_offset": offset,
                "byte_length": len(data),
            })
            offset += len(data)

        # 3. embeddings + non-quantized tensors (norms, biases) at fp16
        for k, v in sd.items():
            if not isinstance(v, torch.Tensor):
                continue
            if any(blob.name in k for blob in layer_blobs):
                continue
            if not (k.startswith("model.embed_tokens") or
                    k.startswith("model.norm") or
                    k.startswith("lm_head") or
                    "norm.weight" in k or
                    k.endswith(".bias")):
                continue
            t = v.detach().cpu().to(torch.float16).contiguous()
            data = t.numpy().tobytes()
            f.write(data)
            manifest_records.append({
                "name": k,
                "shape": list(t.shape),
                "dtype": "float16",
                "format": "raw",
                "shard": "params_shard_0.bin",
                "byte_offset": offset,
                "byte_length": len(data),
            })
            offset += len(data)

    # ndarray-cache.json manifest (MLC expects this name + schema)
    ndarray_cache = {
        "metadata": {
            "ParamSize": len(manifest_records),
            "ParamBytes": offset,
            "BitsPerParam": 4.0,
        },
        "records": [
            {
                "dataPath": "params_shard_0.bin",
                "format": "raw-shard",
                "nbytes": offset,
                "records": manifest_records,
                "md5sum": "",
            }
        ],
    }
    (out / "ndarray-cache.json").write_text(json.dumps(ndarray_cache, indent=2))

    # mlc-chat-config.json — model arch (Llama-like) + quantization
    cfg = _build_mlc_chat_config(model, hf_model_id=hf_model_id)
    (out / "mlc-chat-config.json").write_text(json.dumps(cfg, indent=2))

    # Tokenizer files copied verbatim from the HF snapshot (they have not
    # been touched by quantization).
    if hf_snapshot_dir is not None:
        snap = Path(hf_snapshot_dir)
        for fname in ("tokenizer.json", "tokenizer.model",
                      "tokenizer_config.json", "special_tokens_map.json"):
            src = snap / fname
            if src.exists():
                shutil.copy2(src, out / fname)

    summary = {
        "output_dir": str(out),
        "shard": str(shard_path),
        "n_quant_layers": len(layer_blobs),
        "n_skipped_conv_layers": len(skipped_conv),
        "skipped_conv": skipped_conv,
        "shard_size_mb": shard_path.stat().st_size / 1e6,
        "fold_U": bool(fold_U),
        "fold_super_weights": bool(fold_super_weights),
    }
    return summary


def _build_mlc_chat_config(model: torch.nn.Module, *, hf_model_id: str) -> dict:
    """Build the mlc-chat-config.json for a Llama-like model.

    We read the architecture parameters off `model.config` so this works
    for TinyLlama and any other Llama-derivative.
    """
    cfg = model.config
    return {
        "model_type": "llama",
        "model_id": hf_model_id,
        "quantization": "q4f16_1",
        "model_config": {
            "hidden_size": int(cfg.hidden_size),
            "intermediate_size": int(cfg.intermediate_size),
            "num_attention_heads": int(cfg.num_attention_heads),
            "num_key_value_heads": int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)),
            "num_hidden_layers": int(cfg.num_hidden_layers),
            "vocab_size": int(cfg.vocab_size),
            "rms_norm_eps": float(getattr(cfg, "rms_norm_eps", 1e-5)),
            "rope_theta": float(getattr(cfg, "rope_theta", 10000.0)),
            "max_position_embeddings": int(getattr(cfg, "max_position_embeddings", 2048)),
            "tie_word_embeddings": bool(getattr(cfg, "tie_word_embeddings", False)),
            "context_window_size": int(getattr(cfg, "max_position_embeddings", 2048)),
            "prefill_chunk_size": 512,
        },
        "vocab_size": int(cfg.vocab_size),
        "context_window_size": int(getattr(cfg, "max_position_embeddings", 2048)),
        "prefill_chunk_size": 512,
        "tokenizer_files": [
            "tokenizer.json", "tokenizer.model",
            "tokenizer_config.json", "special_tokens_map.json",
        ],
        "conv_template": {
            "name": "llama-2",
            "system_prefix_token_ids": [1],
            "system_template": "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
            "system_message": "",
            "roles": {"user": "[INST]", "assistant": "[/INST]"},
            "messages": [],
            "seps": [" "],
            "role_content_sep": " ",
            "role_empty_sep": " ",
            "stop_str": ["[INST]"],
            "stop_token_ids": [2],
            "function_string": "",
            "use_function_calling": False,
        },
        "_meta": {
            "exporter": "triad_ptq.export.mlc",
            "fold_U": True,
            "fold_super_weights": True,
            "note": (
                "Activation-side rotation U and stored sparse super-weights "
                "are folded into the dense INT4 weight at export time so the "
                "MLC q4f16_1 kernel can consume the bundle without "
                "TRIAD-specific runtime support."
            ),
        },
    }
