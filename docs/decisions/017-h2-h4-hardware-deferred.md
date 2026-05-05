# ADR-017 — Phase H2–H4 (full eval matrix + falsification gate) deferred to runbook

Status: **Accepted** (in scope: v2-spectra Phase H)
Date: 2026-05-06
Branch: `v2-spectra`

## Context

Phase H of the v2 plan calls for:

* **H2**: full evaluation matrix on Llama-3.2-1B, TinyLlama-1.1B,
  SmolLM-360M, SmolLM-135M against 8 baselines, on Galaxy Z Flip7
  (Exynos 2500) AND Galaxy S25+ (Snapdragon 8 Gen 4), with WT2 PPL +
  HellaSwag + ARC-c/e + Winogrande + PIQA + BoolQ.
* **H3**: emit `results/v2_full_sweep.json` and `results/plots/v2_*.png`.
* **H4**: enforce the falsification gate (Llama-3.2-1B WT2 ≤ FP16+0.5;
  zero-shot avg gap ≤ 2 abs %; TinyLlama quantises end-to-end; decode
  tok/s ≥ baseline; disk MB ≤ 0.92 × baseline at G=64; calibration
  ≤ 30 min on RTX 4090; Squisher–Hutchinson r ≥ 0.7).

All of these require:
* An **RTX 4090** (or comparable CUDA host) for calibration.
* The **Galaxy Z Flip7 / Exynos 2500** physical device for primary
  decode tok/s and Mali decode_tps.
* The **Galaxy S25+ / Snapdragon 8 Gen 4** for cross-platform.

The current development host (M1 Pro, 16 GB) can:
* Calibrate SmolLM-135M and SmolLM-360M end-to-end (the v1 pipeline
  is M1-native, and v2 reuses it through `compile_model`).
* Calibrate TinyLlama-1.1B with the v2 pipeline IFF the Cholesky
  fallback from Phase A6 is wired up (it is, in
  `triad_ptq/utils/device.py::safe_cholesky_inverse`). However a 4090
  run completes in ≤ 30 min vs an M1 run that exceeds the budget.
* **Cannot** run the on-device sweep — no Z Flip7 attached to this
  development environment.

## Decision

The v2-spectra branch ships:

1. **The pipeline code** (`triad_ptq/_v2/pipeline.py::run_v2_pipeline`)
   wired through `triad_ptq.api.optimize(algorithm='v2', ...)`.
2. **Unit + integration tests** that exercise the pipeline end-to-end
   on a synthetic Llama config and verify each phase's contracts in
   isolation. Test suite is **155 pass + 5 skip** as of the Phase H
   commit; the 5 skips are gated on `TRIAD_RUN_TINYLLAMA_TESTS=1` /
   `TRIAD_RUN_SMOLLM_TESTS=1` / the full compile_model end-to-end.
3. **Pre-staged** baseline runners under `experiments/baselines/` (Phase
   A5) that emit `skipped:no_cuda` JSONs on the dev host and produce
   real numbers when invoked on the 4090.
4. **The on-device runner** (`tools/bench_android.sh`) tightened to N=10
   + paired-t (ADR-014) and group-size annotated (Phase G3).

**The actual measured numbers for H2–H4 are produced by a separate
runbook, not committed to this branch.** The runbook's job:

1. On RTX 4090: `algorithm='v2'` + `experiments/baselines/run_*.py` for
   all 4 models × 8 baselines.
2. Move the q4f16_1 bundles to the Z Flip7.
3. Run `tools/bench_android.sh BENCH_GROUP_SIZE=$G ...` at N=10 for each
   bundle pair.
4. Aggregate via `decide_default_group_size` (ADR-015) and the H4 gates.
5. Update `README.md` per Phase I if and only if the gates pass.

## Why this isn't pretending to ship measured numbers

The v2 plan's working rules are explicit:

> "Every numerical claim in commits, docs, or PRs must cite the
>  experiment file (`results/.../*.json`) it came from."

We honour that by **not making any v2 PPL or decode-tps claim** that
isn't backed by a `results/v2/*.json` file written from a real run on
the dev host. Phase A–G commit messages cite only synthetic-fixture
measurements (rotation cosine on tiny LlamaConfig, BRECQ loss on
ToyMLP, INT4/INT8 MSE on randomly-init weights). The MODEL-AND-DEVICE
claims are blank in `results/v2/` and the README rewrite (Phase I)
does not pretend they exist — see Phase I section "What v2 doesn't
claim yet."

## What v2.0.0-alpha tags ship

* All v2 code paths, tests, ADRs, and synthetic measurements.
* The `algorithm='v2'` API entry point (working on Llama-family models).
* Empty `results/v2/v2_full_sweep.json` placeholder reserved for the
  runbook.
* A note in the README that the v2 path is **alpha** — it runs
  end-to-end on synthetic fixtures and is ready for the runbook, but
  the published quality / speed claims are from v1's session-3 numbers
  until the runbook produces v2 numbers.

## v2.0.0-rc1

The promotion from alpha → rc1 is gated on:
* All H4 falsification criteria pass.
* ADR-015 status flips from "Provisional" to "Accepted: G=…".
* Phase I's README rewrite cites real `results/v2/*.json` files for
  every PPL / decode claim it makes.

## References

- v2 plan, Phase H4 falsification gate.
- ADR-014 (N=10 paired-t protocol).
- ADR-015 (group-size default decision).
- `experiments/baselines/` (Phase A5).
- `tools/bench_android.sh` (Phase G3 group-size annotation).
