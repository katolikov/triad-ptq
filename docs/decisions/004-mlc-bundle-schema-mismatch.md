# ADR-004: MLC bundle schema — switch to "TRIAD-folded HF safetensors → mlc_llm convert_weight"

Status: **Awaiting-review** (autonomous mode, user is awake but stepping through Phase 1)
Date:   2026-05-05
Branch: `feat/exynos-mlc-compile`

## Context

Phase-4 of the prior session wrote `triad_ptq/export/mlc.py` to emit
an MLC q4f16_1 bundle directly from the TRIAD-quantized model. The
bundle that script produces:

- Single `params_shard_0.bin` (778 MB)
- `ndarray-cache.json` manifest
- Per-layer records named `model.layers.N.self_attn.q_proj.q_weight`
  etc., with **interleaved (scale, zero)** in `q_scale_zero`

When inspected against a canonical MLC bundle produced by
`mlc_llm convert_weight` on the reference HF TinyLlama, four
mismatches surface:

| Field | TRIAD exporter | MLC canonical |
|---|---|---|
| Manifest filename | `ndarray-cache.json` | `tensor-cache.json` |
| Shard count | 1 (778 MB) | 24 (~24 MB each, 576 MB total) |
| Record names (attn) | separate `q_proj`, `k_proj`, `v_proj` | **fused `qkv_proj`** |
| Record names (ffn) | separate `gate_proj`, `up_proj`, `down_proj` | **fused `gate_up_proj`** + `down_proj` |
| Scale layout | interleaved `q_scale_zero` (2*ng fp16 per row) | separate `q_scale` (ng fp16) per group, no explicit zero (q4f16_1 derives zero from min) |
| Bits-per-param | ~5.6 (because of fp16 scale + fp16 zero) | 4.501 |

The MLC q4f16_1 layout name is misleading: the `_1` suffix denotes
**asymmetric** quant where the zero point is derived FROM the storage
encoding (`zero = q_max / 2`), not stored separately. Our exporter
stored an explicit fp16 zero per group, which is structurally
different.

The MLC compile step succeeds against either layout (it only reads
the model architecture from `mlc-chat-config.json`), but at **load
time** the runtime tries to look up parameters by their canonical
names (`model.layers.0.self_attn.qkv_proj.q_weight`, etc.) and
fails when our manifest has separate `q_proj` / `k_proj` / `v_proj`
records.

## Decision

Replace `triad_ptq/export/mlc.py` with a two-stage path that hands
off to MLC's own canonical conversion:

1. **`triad_ptq/export/hf_safetensors.py`** — new. Reads a TRIAD
   checkpoint, materialises the deployment-side fp32 weight per
   layer (folding `U / Lam_pow_beta` and the stored sparse super-
   weight residuals just as before), and writes the result as a
   regular Hugging Face `model.safetensors` accompanied by the
   original `config.json`. The bundle is *the original HF TinyLlama
   directory*, byte-for-byte, with `model.safetensors` replaced by
   the TRIAD-improved weights.

2. **`mlc_llm convert_weight` + `mlc_llm gen_config` + `mlc_llm
   compile`** — invoked as subprocesses by
   `experiments/14_export_mlc.py`. These are the canonical steps
   MLC ships and tests; running them on the TRIAD-folded
   safetensors produces a bundle that uses MLC's exact record
   names, fused QKV / gate-up shapes, q4f16_1 layout, and 24-shard
   manifest. No layout work in our codebase.

The TRIAD-specific quality benefits flow through this pipeline
intact because they live in *the values* of the dense weight,
not in the storage layout. MLC's q4f16_1 quantization at
`group_size=32` is applied on top of TRIAD-improved weights;
the result is "TRIAD then RTN" — the same quality envelope
the previous direct-export delivered, but with a runtime-loadable
bundle.

`triad_ptq/export/mlc.py` from Phase-4 is retained for reference
(its `_pack_int4_uint32_lanes` and the test suite are useful
documentation of the MLC INT4 layout), but no longer invoked.

## Rejected alternatives

- **Patch the direct exporter to match MLC schema exactly.**
  Requires implementing fused-QKV packing, fused-gate-up packing,
  the (currently-undocumented) `f32-to-bf16` storage format that
  MLC writes for fp16 records, and the multi-shard splitting
  logic — ~400 lines of code that duplicate MLC's own
  `convert_weight` and would silently drift on every MLC version
  bump. Rejected.

- **Modify MLC to accept TRIAD's record naming.**  Out of scope.

## Consequences

- The intermediate "TRIAD-folded HF safetensors" file is ~2.2 GB on
  disk (1.1 B params * fp16 = 2.2 GB; we go fp16 instead of fp32
  to halve disk). It is a temporary artefact and can be deleted
  after the MLC bundle is built.
- `mlc_llm convert_weight` runs in ~10 s on M1 (host CPU); not a
  bottleneck.
- The previously-emitted bundle at `/tmp/triad-tinyllama-int4-mlc/`
  is now stale and re-generated under the new path.
