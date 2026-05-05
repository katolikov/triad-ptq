# SPECTRA-Q (TRIAD-v2) design

This document summarises the v2 architecture as implemented on the
`v2-spectra` branch. It is the canonical companion to the v2.0.0-alpha
release. Per-phase ADRs under `docs/decisions/` (014, 015, 016, 017) own
the specific deviations from the original v2 plan.

## Goals

1. Honest measurement protocol (N=10 + paired-t).
2. Byte-compatible MLC q4f16_1 layout — zero kernel changes.
3. Calibration ≤ 30 min on a single RTX 4090 for any ≤ 1.1B-param model.
4. No backward passes through the full model. Block-wise SGD only.
5. No sparse FP16 path at inference.
6. No dependency on `torch.linalg.eigh` on MPS.

## Pipeline summary

```
input model (HF Llama-family)
   │
   ▼
┌──────────────────────────────────────────────────────┐
│ Phase C: block-diagonal sign+permutation rotation    │  group-aligned at G;
│           folded into RMSNorm γ                      │  preserves per-group max
└──────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────┐
│ Phase B: Squisher Fisher diagonal + ρ-per-block      │  via full backward hooks
│           (replaces v1 noise-injection probe)        │
└──────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────┐
│ Phase F: ρ-weighted α schedule for GPTAQ             │  α = min(0.8, σ(c·log ρ))
│           (replaces fixed α=0.5)                     │  scope-limit retained
└──────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────┐
│ Phase D: learnable per-block β + selective LWC       │  100 Adam steps BRECQ
│           (replaces closed-form β*)                  │  jointly trains α_g
└──────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────┐
│ v1 GPTQ Cholesky update with H_post                  │  per-block α via callable
│  (asym_alpha now accepts dict | callable | float)    │
└──────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────┐
│ Phase E: channel-INT8 mixed precision                │  top-1.5% by κ → INT8;
│           pack into one bundle + 1-bit indicator     │  rest INT4
└──────────────────────────────────────────────────────┘
   │
   ▼
MLC q4f16_1 export (G ∈ {32, 64, 128} sweep — Phase G)
```

## Phase summary

### Phase A — plumbing
- Branch `v2-spectra` cut from `main` (v0.3.0-session3).
- `triad_ptq/_v2/` skeleton + `optimize(algorithm='v1'|'v2')` flag.
- `tools/bench_android.sh` default N=3 → N=10 + paired-t (ADR-014).
- Six baseline runners under `experiments/baselines/`.
- `safe_cholesky_inverse` with CPU/fp64 fallback (resolves v1 TinyLlama OOM).

### Phase B — Squisher Fisher diagonal (router)
- `SquisherAccumulator` EMA of g²; `squisher_fisher_diagonal(block, X, Y)`.
- Hutchinson sanity (rademacher probe, double-backward).
- Correlation ≥ 0.7 acceptance gate. **Measured: mean 0.813, min 0.751
  across 5 seeds on `ToyMLP(12→24→12, GELU)`** (`results/v2/phase_b_squisher_correlation.json`).

### Phase C — block-diagonal rotation
- Two builders: `block_signed_permutation` (default), `block_hadamard_rotation`.
- Block-diagonal at G; **per-group max invariant** (proved by unit test).
- `apply_block_rotation_to_llama(model, group_size, kind, seed)` walker.
- **Measured: forward cosine 1.0000005 .. 1.0000021 over 3 seeds × 2
  kinds on tiny LlamaConfig(hidden=64, num_layers=2)** (`results/v2/phase_c_rotation_forward_equivalence.json`).
- `tools/verify_kernel_identity.sh` smoke-tests md5 invariance of MLC
  device-code objects (gated on a real `mlc_llm compile` run that the
  Phase H runbook owns).

### Phase D — learnable β + selective LWC
- AWQ-style smoothing s_j = (E|X_j|)^β / (max|W_ij|)^(1−β).
- 100 Adam steps with BRECQ block-output reconstruction loss.
- INT4 fake-quant with STE backward.
- Selective LWC: top-25 % most sensitive blocks get learnable α_g ∈ [0.5, 1.0].
- **Measured: BRECQ loss reduction mean 6.28 % across 5 seeds** on
  `_MiniBlock(d=64, hidden=128)` (`results/v2/phase_d_learnable_beta.json`).
- ADR-016 documents why D3's "v1 closed-form β\* as init" is deferred:
  v1's β operates in a different basis.

### Phase E — channel-INT8 super-weights
- κ_j = max_i (|W_ij^rot| · E|X_j|) per output channel.
- Top-1.5 % go INT8; rest stay INT4 — same packed bundle, 1-bit indicator.
- v2.0 deployment Option A: super-channels become a small dense FP16 GEMV
  at runtime (no kernel change, no sparse format).
- E2: `detect_true_super_weights` for the rare PPL-crash weights (Yu et al. arXiv:2411.07191).
- **Measured: MSE(INT4) / MSE(INT8) ≈ 290× across 4 fixtures** at
  G=64 (`results/v2/phase_e_channel_int8.json`).

### Phase F — GPTAQ ρ-weighted α
- `alpha_from_rho(rho, c, alpha_max=0.8) = min(α_max, σ(c · log ρ))`.
- α(ρ=1) = 0.5 exactly — matches v1 default.
- Scope-limit (exclude `o_proj`, `down_proj`) preserved from ADR-010.
- Per-block α JSON log via `write_alpha_log` (`v2_gptaq_alpha/1` schema).
- v1 `compile.py::asym_alpha` now accepts `float | dict | callable`,
  enabling per-layer dispatch from the schedule.

### Phase G — group-size sweep
- `estimate_disk_mb` static byte counter for the v2 packed bundle.
- `run_group_size_sweep(model_id, calibrate_at_g)` harness.
- `decide_default_group_size` strictly requires measured `decode_tps`
  on the target device for both G=32 and G=64.
- `tools/bench_android.sh` annotates `BENCH_GROUP_SIZE` into the JSON.
- ADR-015 holds the default-G choice **provisional** pending Mali measurement.
- **Measured: static disk-MB ratio G=64 / G=32 ≈ 0.945 across 3
  fixtures** (`results/v2/phase_g_groupsize_sweep_static.json`).

### Phase H — integration
- `triad_ptq/_v2/pipeline.py::run_v2_pipeline` orchestrates C → B-lite → F → v1.
- `triad_ptq.api.optimize(algorithm='v2', ...)` dispatches to it.
- `derive_rho_per_block` uses `register_full_backward_hook` on each block
  during one forward+backward pass (sidesteps standalone-block call
  pattern that newer HF Llama can't accept).
- ADR-017 documents the H2–H4 hardware-deferred eval contract.

### Phase I — documentation
- README updated with v2 status banner, "what changed in v2" section,
  "what TRIAD-v2 doesn't claim" section, retired-trace-router note,
  and superseded "+2.7 % decode" claim.
- This document.
- BibTeX extended with a `katolikov2026spectraq` entry for v2.

## Original-vs-port matrix

| Component                   | Origin                                                         | Implementation                  |
|-----------------------------|----------------------------------------------------------------|---------------------------------|
| Block-diag sign+perm rot.   | arXiv:2511.04214 (port; G-aligned)                             | `_v2/rotation/sign_perm.py`     |
| GPTAQ asym. transfer        | arXiv:2504.02692 (v1 port; v2 reuses)                          | `triad_ptq/core/gptaq_asym.py` |
| AWQ-style smoothing         | arXiv:2306.00978 (port)                                        | `_v2/transform/learnable_beta.py` |
| BRECQ block reconstruction  | arXiv:2102.05426 (port)                                        | `_v2/transform/learnable_beta.py` |
| OmniQuant LWC               | arXiv:2308.13137 (port; LWC-only)                              | `_v2/lwc/selective.py`          |
| Squisher Fisher diagonal    | arXiv:2507.18807 (port)                                        | `_v2/router/squisher.py`        |
| FP16 super-weight insight   | Yu et al. arXiv:2411.07191 (port; channel-INT8 reframing)      | `_v2/superweight/channel_int8.py` |
| **ρ-weighted α scheduling** | **Original**                                                   | `_v2/calib/gptaq_rho_alpha.py`  |
| **Channel-INT8 packing fmt**| **Original**                                                   | `_v2/superweight/channel_int8.py` |

## Acceptance summary (v2.0.0-alpha)

* **155 pass + 5 skip** unit tests on M1 (5 skipped tests are heavy
  TinyLlama / SmolLM gates behind env vars + the full compile_model
  smoke deferred to Phase H runbook).
* **No v1 regressions** — every test that passed in v0.3.0-session3
  still passes.
* **Synthetic-fixture measurements** archived in
  `results/v2/phase_{b,c,d,e,f,g}_*.json`.
* **No model-and-device PPL or decode-tps claim** — Phase H runbook
  produces those on a 4090 + Galaxy Z Flip7 (ADR-017).

## Roadmap to rc1

* RTX 4090 calibration on Llama-3.2-1B + TinyLlama-1.1B + SmolLM-360M
  + SmolLM-135M.
* On-device sweep at G ∈ {32, 64, 128} on Galaxy Z Flip7 (Exynos 2500)
  and Galaxy S25+ (Snapdragon 8 Gen 4).
* Apply the H4 falsification gate.
* Update ADR-015 from "Provisional" to "Accepted: G=…".
* Update README with measured v2 numbers.
* Promote tag from `v2.0.0-alpha` to `v2.0.0-rc1`.
