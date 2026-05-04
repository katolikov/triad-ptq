# ADR-002: Exynos 2500 device ships Xclipse 950 (AMD RDNA), not Mali

Status: **Awaiting-review** (autonomous mode, user asleep)
Date:   2026-05-05
Branch: `feat/exynos-baseline`

## Context

The session prompt specified deployment to "Samsung Exynos 2400 or 2500"
with GPU "Mali (Bifrost/Valhall family), accessed through Vulkan", and
all Phase 1–5 instructions are calibrated for that assumption (e.g.
`group_size=64` is described as "required for the Mali path; 32 is
too fine for warp execution").

Phase-0 device probe (see `experiments/device_profile.txt`):

- `ro.soc.model = s5e9955` → **Exynos 2500** (in-scope)
- Device:        Samsung Galaxy Z Flip7 (SM-F766B / b7s)
- Vulkan ICD:    `/vendor/lib64/hw/vulkan.samsung.so`
- Vulkan ext:    `VK_AMD_*` (gcn_shader, shader_ballot, shader_core_properties,
  shader_explicit_vertex_parameter, etc.), plus `VK_SEC_amigo_profiling`
  and `VK_SECX_properties` (Samsung-specific)
- Kernel:        no `mali0` device, `22200000.sgpu` (Samsung GPU /
  AMD RDNA derivative — public name **Xclipse 950**)
- NN runtime:    Exynos NN ("ENN") via `libenn_user_driver_gpu.so` —
  no `exynos_nn_compile` CLI on this non-rooted retail build.

Conclusion: the SoC is in-scope, but the GPU family in the prompt is
incorrect. Exynos 2500 ships Xclipse 950 (RDNA-based), not Mali. The
NPU access path the plan assumes (a userland `exynos_nn_compile`
binary) is also unavailable on a stock retail build.

## Decision

Treat the SoC match as the binding criterion (which the prompt itself
states: "If `ro.soc.model` reports a different SoC ... stop and tell
the user" — it does NOT). Continue Phases 1–5, but adjust the few
plan items that conflate "Vulkan" with "Mali":

1. **Vulkan target is preserved.** MLC-LLM compiles SPIR-V via TVM;
   the same `--device android` / `--device vulkan` path works for
   any conformant Vulkan ICD. Xclipse 950 advertises a feature set
   that strictly includes what MLC's Vulkan kernels need (`storage_buffer_storage_class`,
   `subgroup_size_control`, `shader_float16_int8`, `8bit_storage`,
   `16bit_storage`, `shader_integer_dot_product` — all required for
   q4f16_1 and present in the extension list).

2. **`group_size` choice.** The plan asserts `group_size=64` is
   "required for Mali; 32 is too fine for warp execution". RDNA
   uses wave32/wave64 — wave32 is the default on RDNA3, so
   `group_size=32` actually maps cleanly to one wave per group.
   However MLC's reference `q4f16_1` format is **`group_size=32`**
   (`q4f16_1` literally encodes 32-elt groups in its packing). To
   stay on the well-trodden MLC path and avoid having to rewrite the
   q4f16_1 packing (which Phase 4 explicitly says to mirror byte-for-
   byte), we keep **`group_size=32`** end-to-end.
   - Phase-3 calibration uses `group_size=32`.
   - Phase-4 export packs as MLC `q4f16_1`.
   This is the conservative choice for correctness; performance
   on RDNA is, if anything, slightly better than g64 thanks to wave32.

3. **Asymmetric quant.** Kept as planned. MLC's `q4f16_1` stores
   per-group fp16 zero-point alongside fp16 scale; this is what we
   produce.

4. **Phase 6 (NPU INT4 path).** Deferred. The path requires
   `exynos_nn_compile` as a userland CLI which is not present on
   stock retail Galaxy Z Flip7 (Android 16, non-rooted). Documenting
   the gap rather than spending session time on it. If/when Samsung
   exposes ENN compilation outside Galaxy AI privileged binaries we
   can revisit.

5. **Acceptance numbers (PPL ≤ +1.0, decode ≥ 25 tok/s, peak GPU mem
   ≤ 1.2 GB) are unchanged.** They are stated in the prompt against
   "Mali Vulkan", but the underlying constraints (model size 1.1 B at
   4 bits, batch 1, prompt+gen 64+64) are GPU-vendor-agnostic, and
   Xclipse 950 sits in the same performance class as Mali G725 / Adreno
   830. If anything the AMD-derived part should outperform on dense
   matmul. We will report numbers as-is.

## Rejected alternatives

- **Stop and ask.** Per autonomous-mode rule, only stop if the SoC
  itself is out of scope. It is in scope. Document and continue.
- **Switch to a different deployment runtime (e.g. ENN-only / SNPE-
  style).** Requires privileged Galaxy AI binaries we cannot install.
- **Switch group_size to 64 anyway.** Would require forking MLC's
  q4f16_1 packing layout. Explicitly forbidden by Phase-4 instruction
  ("mirror its layout byte-for-byte"). Rejected.

## Consequences

- The phrase "Mali Vulkan" in any subsequent commit / report should
  be read as "Vulkan on Xclipse 950" for this run.
- If the device underperforms vs. the 25 tok/s acceptance, root cause
  may be Xclipse-specific Vulkan driver behaviour, not MLC bugs;
  that diagnosis lane is opened by this ADR.
- Phase 6 is recorded as N/A on this device, not as a failure.
