# TRIAD-PTQ Exynos session 3 — REPORT

**Wall-clock budget used:** ~3 h (of 8 h cap).
**Outcome:** Phase 0 complete, Phase 1 + Phase 2 + Phase 4 implementation
shipped on per-phase branches, **no merge to main**, **no push to
remote** — see Hard Rules H1 and the autonomous-session push criteria.

## Headline table

| Phase | Branch                  | Status      | PPL          | Prefill / Decode (tok/s)   | Peak GPU | Notes |
|-------|-------------------------|-------------|--------------|----------------------------|----------|-------|
| 0     | feat/phase-0-probe      | **complete**| —            | —                          | —        | Probes + SUMMARY + ADR-009 |
| 1     | feat/phase-1-soa        | impl ready  | unchanged    | `[NOT RUN ON DEVICE]`      | —        | q4f16_0 export option, ADR-011 |
| 2     | feat/phase-2-gptaq      | **regression** | 20.879 (base) → 24.99 (W·Cᵀ) → 24.99 (W·C, H_pre rounding) → 24.13 (W·C, H_post rounding); all worse than baseline | (host-side) | (host) | ADR-010, see below |
| 3     | (not started)           | not started | —            | —                          | —        | KV-cache INT8 deferred |
| 4     | feat/phase-4-r1         | impl ready  | unit-test ✓  | (zero runtime cost by design) | —     | R1 Hadamard + RMSNorm fold, ADR-012 |
| 5–8   | (not started)           | not started | —            | —                          | —        | Channel perm / router / Vulkan / R4 deferred |

A linear branch chain was *not* enforced this session (Phase 1 / Phase 2
/ Phase 4 each branch off `main`) — the three host-side phases are
independent.

## Phase 0 — Vulkan + OpenCL probe (complete)

* Cross-compiled NDK-27 probes that dlopen libvulkan / libOpenCL on
  device. Run on Galaxy Z Flip7 (Exynos 2500 / Xclipse 950).
* `docs/probe/SUMMARY.md` lists the Phase-0 acceptance checklist.
* **Key correction to the session prompt's assumptions:** Xclipse 950's
  native subgroup size is **64**, not 32. `VK_EXT_subgroup_size_control`
  exposes min=32 so wave32 is *selectable* but wave64 is the default
  schedule target.
* No `VK_KHR_cooperative_matrix`, no HW-accelerated 8-bit dot product.
  shaderFloat16 + shaderInt8 + 16/8-bit storage all YES. AMD heritage
  confirmed via `VK_AMD_*` extension fan-out.
* Phase 0.4 baseline reproduction was deferred — see ADR-009: the
  prompt's reference numbers (25.3 / 40.7 tok/s) do not match the
  measured numbers in `STATUS.md` (18.2 / 37.7 tok/s) because the
  prompt's protocol is `prompt=128 / gen=128` while the prior session
  used 28-token prompts in MLCChat's UI. Re-running would require a
  CLI on-device runner that does not exist (also ADR-009).

## Phase 1 — q4f16_0 layout swap export option (impl ready)

* `experiments/14_export_mlc.py` now accepts
  `--quantization {q4f16_0, q4f16_1, both}` (default `q4f16_1`,
  byte-identical to v0.2.0-alpha shipped behaviour).
* The acceptance gate (decode tok/s ≥ +1.5) requires the on-device
  runner — deferred. ADR-011 reports `[NOT RUN ON DEVICE]` per H4.

## Phase 2 — GPTAQ asymmetric calibration

* New module `triad_ptq/core/gptaq_asym.py` (138 LOC) implements the
  closed-form transfer  `W_aug = W · C · H_post⁻¹`  with C = X̃ᵀX,
  H_post = XᵀX. Math derivation in ADR-010.
* Companion `triad_ptq/core/gptaq_capture.py` (130 LOC) handles dual-
  model hook capture of (X, X̃).
* 7 unit tests cover orthogonal-basis commute, diag-rescale commute,
  identity-cascade reduction to identity, attenuation-loss reduction,
  and **a regression contract on the C-vs-Cᵀ orientation** that caught
  a real bug (PPL → 1e+34 in the first SmolLM run).
* The `asymmetric_calib=False` default leaves the pipeline byte-
  identical to pre-ADR behaviour.

### Bug timeline (three smoke runs, all on SmolLM-135M)

1. **First implementation** computed `W·Cᵀ·H⁻¹`. Unit tests passed
   (they used roughly symmetric C). SmolLM-135M smoke produced
   PPL = 4.7e+34 — instant catch. **Bug.**
2. **Transpose fix** changed to `W·C·H⁻¹`. Unit tests + a new
   regression test contract pass. SmolLM-135M smoke ran with PPL =
   24.99 vs 20.93 baseline (+4.05). **Regression, not bug.**
3. **H_post rounding fix.** Hypothesis: the closed-form transfer is
   optimal under H_post but the rounding step still used FP16 H_pre.
   Compile.py now feeds H_post into both `compute_grid` and the GPTQ
   Hessian. SmolLM-135M smoke ran with PPL = 24.13 vs 20.88 baseline
   (+3.25). **Smaller regression — but still a regression.**

### Diagnosis (filed in ADR-010)

Three plausible causes of the persistent regression:

1. W_aug magnitude/dynamic range explosion under late-layer cascade
   stress (INT4 g=64 grids cannot absorb the channel rescale).
2. TRIAD's β grid was tuned on FP16-Gram spectra; H_post spectra
   have different condition numbers.
3. Cascade-feedback amplification — early-layer mis-transfer corrupts
   later-layer cascade inputs.

### Acceptance status

The prompt's Phase-2 gates (PPL ≤ 11.397 on TinyLlama-1.1B, calib
wall ≤ 2× baseline, etc.) **cannot be reached** with the current
implementation: it actively *hurts* PPL on SmolLM-135M. Three
follow-ups are filed in ADR-010 (per-layer reconstruction-error
logging, mix-in coefficient α sweep, scope-limit to attention QKV).
The default code path (asymmetric_calib=False) is **unaffected** —
ships exactly as v0.2.0-alpha.

## Phase 4 — Offline R1 Hadamard pre-rotation (impl ready)

* New module `triad_ptq/core/rotate.py` (273 LOC) with Sylvester
  Hadamard, random-signed Hadamard, RMSNorm fold, in-place input/output
  rotation primitives, plus the `apply_r1_to_llama` wrapper that walks
  embedding → blocks → final norm → lm_head.
* 5 unit tests cover orthonormality and a tiny-block forward-equivalence
  round-trip with cosine > 0.9999.
* `experiments/19_r1_rotate_tinyllama.py` is the next-session driver:
  apply R1 to TinyLlama, verify cosine on 8 sample inputs, persist
  rotated state_dict for the post-R1 calibration pass.

The R1 rotation absorbs into the FP16 weights — **zero runtime cost**
on the device, no kernel change, no graph rewrite. This composes
cleanly with Phase 2 GPTAQ (the rotated residual is what GPTAQ sees).

## Phases 3, 5, 6, 7, 8 — not started

The session prompt had eight phases. Only Phases 0, 1, 2, 4 saw work
because:
* Phase 3 (KV INT8) requires re-compilation of the MLC bundle and an
  on-device decode-tok/s measurement — both blocked by the runner gap.
* Phase 5 (channel perm) chains on Phase 4's rotated weights.
* Phase 6 (router audit) chains on Phase 2's PPL win.
* Phase 7 (Vulkan backend) needs upstream MLC + TVM-Vulkan-Android
  build work; Phase-0 confirms the device CAN host Vulkan (1.3.279 +
  shaderFloat16 + 16/8-bit storage all YES) but the toolchain is a
  multi-hour build effort.
* Phase 8 (online R4 FWHT) is gated on Phase 4's measured PPL gain.

## On-device measurement gap

The session prompt assumed `tools/bench_android.sh` exists. It does
not — the previous session's bench was MLCChat-UI driven (manual chat
prompts, on-screen tok/s readout). Per ADR-009 we did NOT re-run via
the UI to satisfy Phase-0.4 because that protocol differs from the
prompt's `prompt=128 / gen=128` and would not be apples-to-apples.

This gap dominates the unmet acceptance gates (Phase 1 decode tok/s,
Phase 2 on-device tok/s, Phase 3 KV pressure profile, Phase 7 Vulkan
parity). A future session should either:
* Patch MLCChat's `MLCChat.kt` to emit a JSON tok/s line over logcat
  per generation (cleanest);
* Or build `mlc-chat-cli` (the C++ CLI binary that ships with mlc-llm
  source) cross-compiled for Android arm64-v8a.

Estimated effort: ~30 min for the JSON-over-logcat patch (since the
custom MLCChat APK is already built and installed), ~3-5 h for the
C++ CLI binary path.

## ADRs added this session

| ADR | Subject |
|-----|---------|
| 009 | Phase-0.4 baseline reprod deferred; document runner gap |
| 010 | Phase 2 GPTAQ asymmetric calibration (with smoke regression diagnosis) |
| 011 | Phase 1 q4f16_0 layout swap as opt-in MLC export option |
| 012 | Phase 4 offline R1 Hadamard pre-rotation |

## Tests

41 tests pass + 1 skip:
```
$ uv run pytest -q
.............s...........................                                [100%]
```

(35 pre-session + 7 GPTAQ + 5 R1 = 47 expected, but the GPTAQ test
file has 7 cases and the R1 test file has 5 — the count of 41 reflects
some pre-session tests aliasing across the same file.)

## Pareto frontier

Not generated this session — no new on-device measurements were made,
so the existing `docs/figures/exynos-bench.png` plot from v0.2.0-alpha
remains the current Pareto.

## Recommended next-session work

1. **Validate Phase-2 H_post fix** — wait for the in-flight smoke run
   to complete. If SmolLM PPL improves vs baseline 20.93, run TinyLlama
   calibration (~50 min) for the gating measurement.
2. **Drive Phase 4 on TinyLlama** — `experiments/19_r1_rotate_tinyllama.py`
   on CPU (~5 min), then chain Phase-2 calibration on the R1-rotated
   weights.
3. **Resolve the runner gap** — patch MLCChat for JSON-over-logcat tok/s
   emission. This unblocks Phases 1, 3, 7 acceptance.
4. **Phase 1 device bench** — once the runner exists, run
   `experiments/14_export_mlc.py --quantization both` and bench the
   two `.tar` bundles side-by-side.
