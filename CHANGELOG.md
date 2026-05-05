# Changelog

## v0.2.0-alpha — 2026-05-05

First end-to-end deployment of TRIAD-PTQ on real edge hardware
(Galaxy Z Flip7 / Exynos 2500 / Xclipse 950 GPU). All three Phase 5
acceptance criteria met:

- WikiText-2 PPL: 11.477 vs FP16 10.882 (gap +0.595, budget +1.0).
- On-device decode: 40.7 ± 0.6 tok/s (target ≥ 25), N=3 averaged.
- Peak GPU memory on device: 789 MB Graphics (target ≤ 1200 MB).

### Added

- `triad_ptq/core/calibration.py` — streaming per-layer calibration;
  per-layer A / U / W_prime / kappa no longer retained for all layers
  simultaneously. Unblocks TinyLlama-1.1B on M1 with 16 GB UM
  (`a_device='cpu'` keeps Gram matrices on host RAM).
- `triad_ptq/core/gptq_solver.py::clip_search` — per-group activation-
  weighted clip-ratio search. **Default OFF.** On TinyLlama-1.1B it
  lowers eval-window PPL by 0.13 but degenerates autoregressive
  generation; treat as research-only (see ADR-007).
- `triad_ptq/export/mlc.py` — initial direct MLC q4f16_1 exporter
  (kept for reference; superseded by the canonical path below).
- `triad_ptq/export/hf_safetensors.py` — TRIAD-folded HF safetensors
  exporter (the input to `mlc_llm convert_weight`).
- `experiments/14_export_mlc.py`, `experiments/17_export_mlc_v2.py` —
  one-shot pipelines: TRIAD checkpoint → HF safetensors →
  `mlc_llm gen_config` + `convert_weight` + `compile`.
- `experiments/13_tinyllama_phase3.py`,
  `experiments/16_tinyllama_phase3_v2.py` — TinyLlama-1.1B
  calibration entry-points (v1: n_calib=8 ships; v2: n_calib=64 +
  clip_search measured but not deployed).
- `experiments/profile/` — Stream A diagnostics: tooling probe,
  scale-distribution analysis, replicated-bench harness, plot
  generator.
- `tests/test_clip_search.py`, `tests/test_super_weight_index_bounds.py`,
  `tests/test_generation_smoke.py`, `tests/test_mlc_export*.py`,
  `tests/test_memory_streaming.py` — 18 new tests on top of the
  v0.1 baseline (33 total; smoke skips by default).
- `docs/decisions/00{1,2,3,4,5,6,7,8}.md` — eight ADRs documenting
  every divergence from the original spec.
- `results/plots/exynos_device_bench.png`,
  `results/exynos_comparison.md` — final device numbers (N=3 mean ±
  std).

### Fixed

- Defensive clamp on super-weight (row, col) indices in
  `triad_ptq/compile.py` for large-vocab final FC layers
  (`lm_head`, m=32000 on TinyLlama). MPS produced exactly `m` once
  for `top_idx // kp.size(1)` after the `n_calib` bump in v2;
  pre-existing bug, just hadn't been hit at v1's smaller calibration
  set.
- GPTQ Cholesky no longer retains `O(layers)` fp32 working tensors
  simultaneously (Phase 2).

### Methodology

- Bench results MUST be N≥3 averaged (ADR-006). The original
  Phase-5 single-run numbers showed apparent gaps of 12 % decode and
  28 % prefill that disappeared at N=3 averaging — the gap was within
  the reference model's own 1-σ band.

### Deferred / known-issue

- True GPTAQ asymmetric-target calibration (Stream B / B.3) deferred
  — full multi-pass calibration didn't fit the session budget.
- `clip_search` ratio band needs tightening (`(1.0, 0.99, …, 0.95)`
  cap) and `Wq` pre-clamp removed before re-enabling. ADR-007
  outlines the follow-up.
- NPU INT4 path on Exynos 2500 unreachable on stock Galaxy Z Flip7
  (Samsung's `exynos_nn_compile` toolchain is privileged).

### Rollback

Every branch tip was tagged `archive/<branch>-pre-rebase` before
this release's rebase. To revert: `git reset --hard <pre-FF
commit>` and check out the archive tags.
