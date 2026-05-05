# TRIAD-PTQ Exynos session 3 — STATUS

Wall-clock used: ~3 h of an 8-h budget.
Outcome: **mixed** — Phase 0 fully shipped, Phase 1/4 implementation +
tests shipped, Phase 2 implementation regressed twice and a third fix
is in flight at session close.

## Branches

| Branch                     | Last commit (local)                              | Pushed? |
|----------------------------|--------------------------------------------------|---------|
| `feat/phase-0-probe`       | `ff44873` ADR-009: defer 0.4 baseline; runner gap| no      |
| `feat/phase-1-soa`         | `57c6cc1` Phase 1: --quantization {q4f16_0,_1,both} | no   |
| `feat/phase-2-gptaq`       | (in flight — H_post Hessian-swap fix; smoke re-running) | no   |
| `feat/phase-4-r1`          | `<commit pending>` Phase 4 R1 Hadamard          | no      |

Per H1 + the autonomous-session prompt: **no merge to main**, **no push
to remote**. The user reviews and merges by hand, then explicitly
authorises a push.

## Headline numbers

|                                    | value                | acceptance       | status |
|------------------------------------|----------------------|------------------|--------|
| Phase-0 probe artefacts on disk    | 6 835 + 1 572 bytes  | required         | **PASS** |
| All Phase-0 capability Y/N answered | 6/6                 | required         | **PASS** |
| TRIAD-INT4 baseline reproduced     | not measured         | within ±5%       | **DEFERRED — see ADR-009** |
| Phase-2 SmolLM-135M smoke          | 20.88 base / 24.13 asym (H_post fix) | < baseline | **FAIL** — best variant still +3.25 regression |
| Phase-2 TinyLlama calibration      | not run              | PPL ≤ 11.397     | **DEFERRED** |
| Phase-4 forward cosine on TinyLlama | not run              | ≥ 0.9999         | **DEFERRED** (passes on tiny-block unit test) |
| Phase-1 on-device decode tok/s     | not measured         | ≥ +1.5 vs q4f16_1 | **DEFERRED — runner gap** |

## What shipped this session

* **Phase 0** — `tools/{vk,cl}_probe`, NDK-27 cross-compiled, run on
  device. `docs/probe/SUMMARY.md` answers the Phase-0 acceptance
  checklist. Committed.
* **Phase 1** — `experiments/14_export_mlc.py` accepts
  `--quantization {q4f16_0, q4f16_1, both}` with per-format bundle
  dirs. ADR-011. Committed.
* **Phase 2 (regression — research-only)** — `triad_ptq/core/gptaq_asym.py`
  (closed-form transfer + diagnostics, 138 LOC), `gptaq_capture.py`
  (dual-model hook capture, 130 LOC), wiring in `compile.py` behind a
  default-off `asymmetric_calib` flag. 7 unit tests green. Three SmolLM
  smoke runs:
    1. PPL=1e+34 — transpose bug, fixed.
    2. PPL=24.99 vs 20.93 baseline — H_pre rounding mismatch, fixed.
    3. PPL=24.13 vs 20.88 baseline — **still a regression** despite
       H_post Hessian-swap landing.
  See ADR-010 for the diagnosis (W_aug magnitude, β-grid mistune,
  cascade-feedback) and the three filed follow-ups (per-layer logging,
  α mix-in, scope-limit to QKV). The default `asymmetric_calib=False`
  path is byte-identical to v0.2.0-alpha and ships unchanged.
* **Phase 4** — `triad_ptq/core/rotate.py` (Sylvester Hadamard, random
  signed Hadamard, RMSNorm fold, per-block + per-Llama R1 application;
  273 LOC). 5 unit tests green including a tiny-block forward
  equivalence round-trip with cos > 0.9999. ADR-012. Committed.

## What did NOT ship

* Phase 0.4 baseline reproduction (deferred, ADR-009).
* All on-device measurements (deferred, runner gap, ADR-009).
* Phase 2 TinyLlama gating measurement (deferred, ~50 min calib + 5 min
  eval after the H_post fix smokes clean on SmolLM).
* Phase 3, 5, 6, 7, 8 — not started.
* The Pareto plot in `docs/figures/pareto-2026-05-05.png` was not
  refreshed because no new on-device numbers exist.

## ADRs added (continuing from prior 008)

| ADR | Subject |
|-----|---------|
| 009 | Phase-0.4 deferred; runner gap |
| 010 | Phase 2 GPTAQ asymmetric (impl + smoke regression + H_post fix proposal) |
| 011 | Phase 1 q4f16_0 export option |
| 012 | Phase 4 R1 Hadamard pre-rotation |

## Tests

```
$ uv run pytest -q
.............s...........................                                [100%]
41 passed, 1 skipped
```

## Next 3 actions for the next session

1. **Diagnose Phase-2 regression with per-layer logging.** Add a
   per-layer `||W − W_aug|| / ||W||` and per-layer reconstruction-MSE
   diagnostic in `compile_model` (gated by `asymmetric_calib=True`)
   and re-run the SmolLM smoke. The hypothesis to falsify is "the
   transfer over-corrects on cascade-stressed late layers". Filed in
   ADR-010 as follow-up #1.
2. **Run Phase 4 on TinyLlama** — `experiments/19_r1_rotate_tinyllama.py`
   on CPU (~5 min). Expect cos > 0.9999. The R1 rotation is *
   independent* of the Phase-2 regression (orthogonal weight transform
   absorbed into FP16 weights, no calibration involvement). Once
   verified, the rotated state_dict is the input to the next
   calibration pass.
3. **Build the device bench harness** — patch the custom MLCChat APK
   to emit a JSON `{"prefill": …, "decode": …}` line over logcat per
   generation. Then drive it from `tools/bench_android.sh` with
   `adb shell input` + `logcat -s triad_bench:I`. This unblocks Phases
   1, 3, 7 acceptance gates and the prompt's `prompt=128 / gen=128 /
   N=5 / cooldown=60s` protocol.
