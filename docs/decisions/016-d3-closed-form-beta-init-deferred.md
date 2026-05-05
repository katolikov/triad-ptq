# ADR-016 — D3 closed-form β* init deferred to Phase H caller

Status: **Accepted** (in scope: v2-spectra Phase D)
Date: 2026-05-06
Branch: `v2-spectra`

## Context

The v2 plan (Phase D step D3) calls for using v1's closed-form β* (paper
eq. 5) as the initialiser for the v2 learnable per-block β instead of
the naive 0.5 default. The plan describes this as a "free improvement"
noted in caveat #4 of the SPECTRA-Q design doc.

In v1, `triad_ptq.core.grid.closed_form_beta(eig, s_spec)` returns the
optimal exponent β* ∈ [0, 0.5] for the transform

    W' = W · U · Λ^β

where U, Λ come from the eigendecomposition of the per-layer activation
Gram. v2's transform is the AWQ / SmoothQuant migration

    s_j = (E|X_j|)^β / (max_i |W_ij|)^{1−β}
    W'  = W · diag(s)

These two transforms operate in **different bases**: v1's β is the
exponent of an eigenbasis dilation, v2's β is the AWQ smoothing
exponent. Numerically they live on the same [0, 1] interval, but they
are not the same scalar — there is no closed-form mapping from the
eigenbasis β* to the AWQ β.

## Decision

`triad_ptq._v2.transform.learnable_beta.train_learnable_beta` exposes
`beta_init: float = 0.5` as a caller-supplied parameter. The default
remains 0.5 (the AWQ paper's grid centre). v1's closed-form β* is
**deliberately not** used as the v2 initialiser inside Phase D's
standalone implementation.

Phase H's integration runbook is responsible for any pre-computation of
a smarter init (e.g. a one-shot AWQ grid search on the rotated calibration
batches, or v1's eigenbasis β* converted via a known-monotone rescaling
if such a rescaling is empirically validated). Phase D's contract ends
at "exposes a `beta_init` knob" — it does not wire v1's eigenbasis β*
into the v2 path.

## Why not just hard-wire it?

1. **Wrong basis.** v1's β minimises a quadratic surrogate over the
   eigenbasis Gram; v2's β minimises a per-group MSE over the AWQ
   migration. Plugging v1's number directly is a category error — the
   loss landscapes do not share their minimisers in general.
2. **No M1 measurement budget for the alternative.** Evaluating whether
   "v1 β converted via heuristic X" beats `β=0.5` on TinyLlama-1.1B
   requires a 4090 host and the full Phase D integration; that
   measurement is the runbook's job.
3. **The trainer converges anyway.** Phase D's measured BRECQ loss
   reductions (`results/v2/phase_d_learnable_beta.json`) show that with
   `beta_init=0.5` the loss decreases over 100 steps even when the
   optimum lies at the [0.05, 0.95] boundary — the saturation detector
   flags this and the per-input-channel-group fallback is wired ready
   for Phase H.

## Consequences

- The Phase D `beta_init` default stays at 0.5 in v2.0.
- If Phase H measurements show that a smarter init reduces calibration
  time meaningfully (≥ 20 % wall-clock at the same final loss), an ADR
  in the v2.1 milestone proposes the specific mapping. Until then the
  caller is free to pass any value through `beta_init=...`.
- The original Phase D plan caveat #4 ("free improvement, drop closed
  form into the init") is not implemented. This ADR documents that the
  caveat was investigated and intentionally not adopted in v2.0.

## References

- v1's eigenbasis closed-form β: `triad_ptq/core/grid.py::closed_form_beta`.
- v2's AWQ-style β: `triad_ptq/_v2/transform/learnable_beta.py`.
- Phase D measurements: `results/v2/phase_d_learnable_beta.json`.
