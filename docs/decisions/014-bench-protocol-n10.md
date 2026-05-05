# ADR-014 — Tighten device-bench protocol from N=3 to N=10 + paired-t test

Status: **Accepted** (in scope: v2-spectra Phase A, this branch)
Date: 2026-05-05
Branch: `v2-spectra`
Supersedes: ADR-006 (only for the `iters` default and the comparison method).

## Context

ADR-006 (2026-05-05) found that the Phase-5 12 % decode gap was an artefact
of the single-prompt (N=1) protocol — under N=3 the gap collapsed to 2.2 %,
inside the 1 σ band of the reference run's own variance (σ ≈ 6 % of decode
mean). ADR-006's mitigation was the H5 rule "iters >= 3" baked into
`tools/bench_android.sh`.

The session-3 README headline ("+2.7 % decode") cites N=3 numbers
(`results/device_bench/2026-05-05_session-3_clean60s.json`). With σ ≈ 6 %
and a delta of ~2.7 %, **N=3 cannot rule out chance**: at α=0.05 a
two-sided paired-t test on N=3 has the rejection region only at
|t| > 4.30, requiring a delta of ~14 % to register significant. Our
honest read is that session-3's "+2.7 %" claim is statistically
indistinguishable from zero.

V2 (SPECTRA-Q) will, by design, make smaller per-component deltas
(channel-INT8 < 0.5 % decode overhead; G=64 vs G=32 maybe ±5 %; rotation
fold is exactly 0 % in expectation). To resolve effects of that size we
need both more iterations and a paired-comparison test that controls for
session-level drift (thermal, OS scheduler).

## Decision

1. **Default `iters` raised from 5 (script default) and 3 (H5 floor) to 10.**
   The hard floor stays at 3 (CI on cold devices), with a script-level
   warning when `iters < 10` that the paired-t output is suppressed.

2. **Paired-t test added to the JSON output**, gated on a `BENCH_BASELINE_JSON`
   environment variable pointing at a previous run's JSON. The pairing is
   iter-k vs iter-k under the assumption that the runs were collected
   back-to-back on the same physical device session (we expect the caller
   to enforce this). Test: per-iter difference d_k = TRIAD_k − ref_k,
   t = mean(d) / (std(d) / sqrt(n)), df = n − 1, p two-sided via the
   Abramowitz–Stegun erf approximation (sufficient for N ≥ 10; the small
   error vs the exact t distribution is < 0.005 in p for N=10).

3. **Reporting contract.** Any speed claim in `README.md`, `STATUS.md`,
   `REPORT.md`, ADRs, or PR descriptions made on the basis of an N < 10
   run is forbidden after this ADR. Existing claims from session-3 stay in
   the changelog as historical record but the README is rewritten in
   Phase I to reflect v2's N=10 numbers.

4. **Cooldown stays at 60 s.** This was already correct in the v0.3.0
   harness; we observed thermal recovery within 30–45 s on Exynos 2500
   under typical room temperature, so 60 s gives margin without bloating
   the runtime budget.

## What did NOT change

- Bundle layout, kernel binaries, MLC compile flags: all unchanged. ADR-014
  is purely a measurement-protocol change.
- The H5 rule from ADR-006 (`iters ≥ 3`) is unchanged as a floor.
- Single-prompt content stays the 14-token deterministic story prompt
  baked into `tools/bench_android.sh`.

## Forward-looking

- Phase G (group-size sweep) and Phase H (full v2 sweep) will be the
  first runs that use N=10 by default.
- The paired-t implementation here is a stop-gap. If we hit a phase
  where p-values cluster near 0.05 we will switch to bootstrap CI on the
  median, which is more robust to the heavy-tailed thermal-throttle
  distribution we see on the Exynos 2500.

## References

- ADR-006 (`docs/decisions/006-decode-gap-root-cause.md`).
- `tools/bench_android.sh` (this branch).
- session-3 device bench JSON
  (`results/device_bench/2026-05-05_session-3_clean60s.json`).
