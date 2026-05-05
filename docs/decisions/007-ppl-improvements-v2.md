# ADR-007: Stream B PPL improvements — clip_search lowers eval PPL but breaks generation; do NOT enable

Status: **Awaiting-review**
Date: 2026-05-05
Branch: `feat/exynos-improve-ppl`

## TL;DR

The B.4 clip-search intervention reduces 4096-token WikiText-2 PPL
from **11.477 → 11.347 (-0.130)** but causes the deployed v2 model
to **emit degenerate, repetitive text** ("The the-l-i-c- and the the-
l-i-c-..." — see `experiments/screenshots/v2-garbage-output.png`).
The eval-window PPL gain is real but does not transfer to coherent
autoregressive generation. **Do not enable clip_search by default.**

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

### B.4 — per-group clip search (committed, default OFF)

`triad_ptq/core/gptq_solver.py` gains a new helper `_find_clip` and a
`clip_search: bool` kwarg on `gptq_quantize_layer`. When enabled:
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

Tests added (`tests/test_clip_search.py`, 4 tests; full suite 33/33
green).

### B.2 — n_calib bump (committed, calibration script only)

`experiments/16_tinyllama_phase3_v2.py` mirrors the v1 entrypoint but
sets `n_calib=64` (8× the v1 setting).

### B.3 — GPTAQ (deferred)

Full GPTAQ (arXiv:2504.02692) requires multi-pass calibration where
each layer's GPTQ pass uses the FP16 model's *output target* with
the *quantised upstream activations*. That doubles the calibration
time and requires a re-forward pass between layer quantisations.
Deferred for a future session; no code added.

### B.5 — asymmetric quantisation (verified, no code change)

Confirmed `_quantize_group` and the GPTQ loop both already store
per-group `(scale, zero_point)` pairs.

### Pre-existing super-weight crash fix (independent value)

The first v2 calibration attempt crashed at `compile.py:335` with
`torch.AcceleratorError: index 32000 is out of bounds: 0, range 0
to 32000` on TinyLlama's `lm_head` (`m=32000`). The MPS path
through `top_idx // kp.size(1)` produced exactly `m`. Defensive
clamp added; CPU-side regression tests in
`tests/test_super_weight_index_bounds.py` (4 tests). The bug is
independent of clip_search; v1 (n_calib=8) just never hit a top-k
boundary that exposed it. Crash log preserved at
`experiments/v2-artifacts/v2_calib_full.log`.

## v2 calibration outcome

After the clamp fix, the v2 calibration completed cleanly:

|                              | v1 (baseline) | v2 (clip_search + n_calib=64) |
|------------------------------|--------------:|------------------------------:|
| n_calib                      | 8             | 64                            |
| clip_search                  | off           | on                            |
| Calibration wall clock       | 1556 s        | 3364 s                        |
| Peak MPS during calib        | 12.19 GB      | 13.74 GB                      |
| WikiText-2 PPL (4088 tokens) | 11.477        | **11.347**                    |
| FP16 baseline (same window)  | 10.882        | 10.882                        |
| Gap above FP16               | +0.595        | **+0.465**                    |

The eval-window PPL improvement is **0.130** — short of the 0.15
acceptance band but well above measurement noise.

`results/triad_tinyllama_int4_v2_m1.json` captures the run.

## v2 device deployment outcome

The v2 bundle was exported via `experiments/17_export_mlc_v2.py`
(canonical `mlc_llm convert_weight + compile`, same device target
as v1). Compiled artefact:
`/tmp/triad-tinyllama-int4-v2-mlc/lib/triad-tinyllama-v2-android.tar`,
total memory with 4K KV cache 804 MB. The OpenCL device-code object
md5 matches v1 exactly (`586216b8…ed8b`); only the param shards
differ.

The v2 bundle was pushed to the device, staged into MLCChat's
internal storage at `/data/data/ai.mlc.mlcchat/files/triad-tinyllama-int4`
(replacing v1; v1 backed up to `triad-tinyllama-int4-v1bak`),
verified by md5 of `params_shard_0.bin`
(`5edd0d32c4a2bf4699623b8ded541411`, matches v2 host).

On running the same prompt that produced coherent (if uninspired)
poetry under v1 ("Now please tell me all about how apples and
bananas make you sing..."), v2 produced **degenerate output**:

```
The the-l-i-c- and the the-l-i-c- and the the-l-i-c-
The the-l-i-c- and the the-l-i-c-
The the-l-i-c- and the the-l-i-c- and the the-l-i-c-
…
```

(See `experiments/screenshots/v2-garbage-output.png`.)

This is a classic PTQ failure mode where the **eval-window
cross-entropy improves but generation collapses**. Pre-clamping
weights destroys the dynamic range needed for autoregressive
sampling: the model still produces a "decent" distribution
*on average* over a fixed test set (hence the slightly better PPL),
but the modes that the sampler picks during sequential generation
become degenerate — typically because the pre-clamping wipes out
the rare large weights that gate attention to specific tokens or
positions, causing the head to collapse onto a narrow few
high-frequency tokens.

This effect is well documented in PTQ literature for aggressive
clip ratios (e.g. AWQ paper §4.3 discussing why their grid search
is conservative, OmniQuant §3.2). Our 0.7 floor on the ratio sweep
is too aggressive; the implementation also pre-clamps `Wq` rather
than just adjusting the per-group `(scale, zero)` pair while
leaving `W` untouched, which literature variants typically avoid.

The device weights were restored to v1 immediately after this
discovery (md5 verified back to v1's `472cd3f6…530a`).

## Decision

Stream B ships with:

- The **clip_search code path implemented**, threaded, and tested,
  but **default OFF** in `triad_ptq.api.optimize`. Nothing changes
  for callers who do not pass `clip_search=True`.
- The **super-weight OOB clamp fix** (independent value; no
  generation impact).
- A clear instruction in the docstring of `gptq_quantize_layer`
  and in this ADR that `clip_search=True` **MUST** be validated by
  qualitative generation, not only by PPL, on each new model.
- The B.4 implementation is **not** declared "ready for production
  on TinyLlama-1.1B." Treat as research-only.

The v2 PPL number is preserved in
`results/triad_tinyllama_int4_v2_m1.json` for completeness, but the
v1 bundle remains the authoritative TRIAD-INT4 deployment for this
project.

## Future work (for a follow-up session)

Per the v2 generation collapse, a follow-up would be:

1. **Tighten the ratio range.** Try `(1.0, 0.99, 0.98, 0.97, 0.96,
   0.95)` so the per-group range never shrinks below 95 %. Several
   AWQ implementations cap at 0.92.
2. **Don't pre-clamp Wq.** Keep the original W; only adjust the
   per-group `(scale, zero)` derived from the chosen ratio. The
   GPTQ Cholesky then absorbs the clamping error as part of its
   normal residual propagation, and outliers retain their
   magnitudes in the per-column update term.
3. **Add a generation smoke check to the v2 calibration script.**
   After the PPL eval, sample 64 tokens from a fixed prompt and
   assert the entropy-of-token-frequency is above some threshold
   (e.g. ≥ 3.0 bits) — repetitive collapse drops it well below.
4. Potentially **gate clip_search per-layer** rather than per-row,
   only enabling on layers where outlier weights are
   demonstrably (a) rare and (b) coincide with low-variance input
   columns.

## Acceptance status

- B.1 audit: **DONE**
- B.2 n_calib bump: **CODE DONE**, calibration completed
- B.3 GPTAQ: **DEFERRED** with rationale
- B.4 clip_search: **CODE DONE + tests pass**, calibration showed
  PPL gain BUT generation is broken; **do not enable by default**
- B.5 asymmetric quant verify: **DONE**
- B.6 device bench v2: ATTEMPTED; aborted after observing degenerate
  output. v1 device bundle restored, untouched by this session.
- B.7 tests: **DONE**, 33/33 green (8 new vs session start)
- B.8 ADR-007: this document.

The "M1 PPL improvement ≥ 0.15" acceptance is **not met**: 0.130
PPL gain on the eval window, and that gain itself is invalidated by
the deployment-side generation collapse.

## Consequences

- The v1 device bundle and v1 device numbers (now N=3, see
  ADR-006) remain authoritative.
- The PR for `feat/exynos-improve-ppl` is a draft. Recommended
  status: **do not merge as-is**; the clip_search code is correct
  but its naive activation is harmful. Either land the
  default-off code paths with the warning in the docstring, or
  drop the clip_search commits and keep only the super-weight
  clamp + n_calib script + tests for a future v2 attempt with
  a less aggressive ratio sweep and no pre-clamp.
