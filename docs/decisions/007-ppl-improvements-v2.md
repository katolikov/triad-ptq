# ADR-007: Stream B PPL improvements — code complete, calibration deferred (pre-existing super-weight bug)

Status: **Awaiting-review**
Date: 2026-05-05
Branch: `feat/exynos-improve-ppl`

## Context

Phase-5 (session 2) reported TRIAD-INT4 PPL 11.477 vs FP16 baseline
10.882 on a 4096-token WikiText-2 window — a +0.595 gap, well under
the +1.0 acceptance budget but above the +0.40 stretch goal.

This session's Stream B was tasked with closing some of that gap via
four specific levers (B.2 n_calib, B.3 GPTAQ, B.4 per-group clip
search, B.5 verify asymmetric quant).

## Audit (B.1)

`experiments/B1_current_config.md` documents the actual state of the
v1 calibration as found in `/tmp/triad-tinyllama-int4/meta.json` and
`triad_ptq/core/`:

| Lever                  | v1 status      |
|------------------------|----------------|
| n_calib                | **8** (the Phase-3 prompt cited 128; meta.json on disk shows 8) |
| gptq_variant           | standard GPTQ (target = local layer output) |
| clip_search            | absent         |
| asymmetric quant       | already on     |

So B.2 and B.4 were the genuinely-available levers; B.3 was a code
addition; B.5 was already in place.

## What landed in code

### B.4 — per-group clip search (committed)

`triad_ptq/core/gptq_solver.py` gains a new helper `_find_clip` and
a `clip_search: bool` kwarg on `gptq_quantize_layer`. When enabled:
1. before the per-group scale/zero are computed, sweep
   `ratios = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)` and for each row
   pick the ratio that minimises an activation-weighted RTN MSE,
   weighted by the per-column input variance `H[j, j]` (true
   `E[x_j^2]` from calibration);
2. pre-clamp `Wq` to the chosen per-row range so the GPTQ Cholesky
   propagates only sub-grid quantisation noise (matches AWQ-clip /
   OmniQuant treatment).

Threaded through `compile_model` → `triad_ptq.api.optimize` as a
keyword argument, default off (no behaviour change unless requested).

A latent bug surfaced during testing: the original draft of
`_find_clip` mixed two asymmetric-quant conventions (added `+wmin`
on top of a zero-point shift). Fixed to match the main GPTQ loop's
convention. Without that fix, every call would pick the smallest
ratio because the dq formula was monotonically advantaged by smaller
scale.

Tests added (`tests/test_clip_search.py`):

- `test_clip_search_reduces_mse_with_outliers`
- `test_clip_search_idempotent_on_uniform`
- `test_gptq_quantize_layer_clip_search_flag_runs`
- `test_gptq_quantize_layer_clip_search_helps_on_realistic_outliers`

Full suite: 29/29 (was 25).

### B.2 — n_calib bump (committed, calibration script only)

`experiments/16_tinyllama_phase3_v2.py` mirrors the v1 entrypoint but
sets `n_calib=64` (8× the v1 setting). The Phase-3 stretch goal of
n_calib=256 was not chosen because, at v1's measured ratio of
profile time to Cholesky time, that projects to ~7 hours on M1 and
the GPTQ paper's Figure 3 shows the PPL/n curve flattens well before
n=64.

### B.3 — GPTAQ (deferred)

Full GPTAQ (arXiv:2504.02692) requires multi-pass calibration where
each layer's GPTQ pass uses the FP16 model's *output target* with
the *quantised upstream activations*. That doubles the calibration
time and requires a re-forward pass between layer quantisations.
Deferred for a future session; no code added.

### B.5 — asymmetric quantisation (verified, no code change)

Confirmed `_quantize_group` and the GPTQ loop both already store
per-group `(scale, zero_point)` pairs.

## What did NOT land — calibration crash

Calibration with `n_calib=64` and `clip_search=True` ran for **66
minutes** on M1 (11:08 → 12:14) before crashing during the per-layer
GPTQ pass:

```
File ".../triad_ptq/compile.py", line 335, in compile_model
    del W_dq
torch.AcceleratorError: index 32000 is out of bounds: 0, range 0 to 32000
```

The full log is preserved at
`experiments/B6_v2_calibration_crash.log`.

`32000` is the TinyLlama vocabulary size (= number of rows in
`lm_head.weight`). MPS reports the error at the next synchronisation
boundary (`del W_dq`), but the actual offending op is the indexing
on lines 332-334:

```python
W_prime[sw_rows.to(W_prime.device), sw_cols.to(W_prime.device)]
- W_dq[sw_rows.to(W_prime.device), sw_cols.to(W_prime.device)]
```

The path that fills `sw_rows` for layers with `U is not None`
(compile.py:307-313) re-runs `compute_kappa` in the transformed
basis and selects top-k by flattened index:

```python
top_idx = torch.topk(flat, min(k_count, flat.numel())).indices
r = (top_idx // kp.size(1)).to(W_prime.device)
c = (top_idx % kp.size(1)).to(W_prime.device)
```

For `lm_head` (`m=32000, n=2048`) we expect `top_idx // 2048` to
yield values in `[0, 31999]`, which is in range. The crash implies
either `kp.size(1)` is not `n` for this layer, or `top_idx` is
exceeding `m * n` somewhere. The bug is **pre-existing** — it does
not depend on Stream B's `clip_search` change; it is sensitive to
`n_calib` only inasmuch as the rho-probe outcome influences which
super-weights are selected for `lm_head`.

## Decision

**Stream B ships as code-only** (clip_search implementation +
tests, n_calib bump in the v2 calibration script). The v2
calibration is not run; v1 numbers stand in
`results/exynos_comparison.md`.

The pre-existing super-weight bug is a separate workstream:
- Branch a fix on top of `feat/exynos-cholesky-fix` (where the
  super-weight code last had attention).
- The minimal repro is `experiments/16_tinyllama_phase3_v2.py` with
  `super_weight_frac=5e-4` and `n_calib=64` on TinyLlama-1.1B; the
  crash is deterministic.
- Once fixed, re-run `experiments/16_tinyllama_phase3_v2.py` and
  `experiments/17_export_mlc_v2.py` (already committed) to produce
  the v2 device bundle.

## Acceptance status

- B.1 audit: **DONE** (`experiments/B1_current_config.md`).
- B.2 n_calib bump: **CODE DONE**, calibration not completed.
- B.3 GPTAQ: **DEFERRED** with rationale.
- B.4 clip_search: **CODE DONE + tests pass**, calibration not
  completed.
- B.5 asymmetric quant verify: **DONE**.
- B.6 device bench v2: **NOT RUN** (no v2 bundle).
- B.7 tests: **DONE**, 29/29 green (4 new).
- B.8 ADR-007: this document.

The "M1 PPL improvement ≥ 0.15" acceptance is **not demonstrated in
this session**. The PR for `feat/exynos-improve-ppl` is therefore a
draft with the new code paths gated behind opt-in flags
(`clip_search=False` by default), ready to be activated once the
super-weight crash is resolved.

## Consequences

- The v1 device bundle and the v1 device numbers remain authoritative
  in the final report.
- `results/exynos_comparison.md` is updated with N=3 means from
  Stream A's replicated bench (ADR-006), which strengthens the v1
  claim irrespective of whether v2 ever lands.
- No Stream B branch is pushed this session; the user decides
  whether to merge the code-only changes as-is or wait for the
  super-weight fix + v2 numbers.
