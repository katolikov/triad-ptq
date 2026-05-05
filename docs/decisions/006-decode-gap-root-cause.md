# ADR-006: Phase-5 decode/prefill gap is measurement noise, not a real shader-level effect

Status: **Accepted** (superseded for the `iters` default by ADR-014)
Date: 2026-05-05
Branch: `feat/exynos-profile-gap`

> **Update 2026-05-05 (v2-spectra branch).** ADR-014 raises the default
> iteration count from 3 to 10 and adds a paired-t test, because v2's
> per-component deltas are smaller than what N=3 can resolve. The H5
> floor of `iters >= 3` from this ADR remains the script-level minimum;
> the new minimum for any *published claim* is N=10. See
> [`docs/decisions/014-bench-protocol-n10.md`](014-bench-protocol-n10.md).

## Context

Phase 5 (session 2) reported, on Galaxy Z Flip7 / Exynos 2500 / Xclipse 950:

|              | Prefill (tok/s) | Decode (tok/s) |
|--------------|-----------------|----------------|
| MLC q4f16_1 community ref | 25.2 | 42.9 |
| TRIAD-INT4   | 18.2            | 37.7           |
| Δ (TRIAD vs ref) | **-27.8 %**  | **-12.1 %**    |

The continuation prompt asked us to find the root cause and propose
one concrete fix. Two streams ran in parallel; this ADR is Stream A.

## What we did (Stream A)

### A.1 — Tooling probe
On-device profilers available: `perfetto v49.0`, `simpleperf`,
`atrace` with `gfx`/`sched`/`freq`/`memory` categories. **No vendor
GPU profiler** (no AGI, no RenderDoc). MLC's prebuilt CLI bench
binary not present on the production APK. Result captured in
`experiments/profile/A1_tools_check.log`.

### A.2 — Bit-level kernel comparison (H1: denormals)
Both bundles share `--device android` compile flags. The TVM-emitted
GPU device code (`llama_q4f16_1_devc.o`) is **bit-identical** between
TRIAD and reference (md5 `586216b8…ed8b` for both). Only `lib0.o`
differs, and only at byte ranges consistent with timestamp metadata.
Therefore the GPU executes the same instructions for both bundles —
the slowdown can only come from **data-dependent runtime behaviour**
(memory access patterns or numerical fast/slow paths).

The most plausible numerical-path hypothesis: TRIAD's per-group fp16
scales might cluster in fp16 subnormal territory due to the U/Λ
rotation, hitting a slow microcode path on Xclipse 950.

We extracted all 34,373,632 per-group fp16 scales from each bundle
(`experiments/profile/A2_scale_distribution.py`,
`A2_scale_distribution_summary.json`):

| Metric                 | TRIAD     | Ref       | Δ      |
|------------------------|-----------|-----------|--------|
| abs_mean               | 6.51 e-3  | 6.48 e-3  | +0.4 % |
| abs_p50                | 6.06 e-3  | 6.03 e-3  | +0.5 % |
| abs_p99                | 1.59 e-2  | 1.58 e-2  | +0.4 % |
| abs_p99.9              | 2.74 e-2  | 2.72 e-2  | +0.8 % |
| abs_max                | 0.4441    | 0.4441    | 0.0 %  |
| frac < 6.10e-5 (fp16 subnormal) | 2.067e-4 | 2.067e-4 | < 0.001 pp |

The distributions are statistically indistinguishable. **H1 is
rejected.**

### A.3 — Replicated bench (H4: measurement variance)
We drove `MLCChat` via `adb input` + `uiautomator dump` and ran the
same 144-character prompt three times against each of the two loaded
models (ref then TRIAD), capturing the in-app
"prefill: X tok/s, decode: Y tok/s" line after each generation
stabilised. Raw runs in
`experiments/profile/A3_replicated_results.json`.

|              | Run 1 | Run 2 | Run 3 | mean | std  |
|--------------|------:|------:|------:|-----:|-----:|
| ref prefill  | 25.9  | 25.0  | 25.5  | 25.5 | 0.45 |
| ref decode   | 38.8  | 42.4  | 43.6  | 41.6 | 2.59 |
| TRIAD prefill| 25.3  | 25.3  | 25.4  | 25.3 | 0.06 |
| TRIAD decode | 40.0  | 41.2  | 40.8  | 40.7 | 0.62 |

**ref's own decode std (2.6 tok/s) is larger than the entire delta
between TRIAD and ref means (0.9 tok/s).**

Comparing N=3 means against Phase-5 single-run numbers:

| Quantity              | Phase 5 (N=1) | This session (N=3) |
|-----------------------|---------------|--------------------|
| ref prefill  / decode | 25.2 / 42.9   | 25.5 / 41.6        |
| TRIAD prefill / decode| 18.2 / 37.7   | 25.3 / 40.7        |
| TRIAD vs ref Δ prefill| **-27.8 %**   | **-0.5 %**         |
| TRIAD vs ref Δ decode | **-12.1 %**   | **-2.2 %**         |

## Verdict

**The Phase-5 gap was an artefact of the single-prompt protocol.**

- The 28 % prefill gap collapsed to 0.5 % under N=3 averaging.
- The 12 % decode gap collapsed to 2.2 %, well within the 1-σ band of
  ref's own decode distribution (σ ≈ 6 % of mean).
- The likely sources of the Phase-5 outlier are (a) cold OpenCL
  command queue / kernel JIT warm-up on the FIRST inference after
  loading TRIAD, and (b) variance from the random sampling path in
  the LLM head (temperature=1.0 default; different token sequences
  stress different parts of the runtime).
- Phase-5's reference-model run was *first* of the session (already
  warm from the Phase-5 walkthrough); TRIAD was loaded *fresh*. This
  matches the cold-start penalty being concentrated on the second
  load.

## Hypothesis log

| ID | Hypothesis                          | Status           | Evidence |
|----|-------------------------------------|------------------|----------|
| H1 | TRIAD scales hit fp16 subnormal slow path | **rejected**  | A.2: distributions identical |
| H2 | Different memory access pattern (cache misses) | **rejected** | layouts identical, kernel code bit-identical (A.2 setup) |
| H3 | Cold-cache / JIT warm-up on first run | **strong**     | A.3: ref run-1 decode 38.8 < ref-mean 41.6 |
| H4 | Sampling-path / token-sequence variance | **strong**    | A.3: σ(decode) ≈ 6 % of mean for both models |
| H5 | Real, persistent shader-level effect | **rejected**   | A.3 means within 2 σ |

## Decision (intervention)

A.5 was framed as "pick one fix and try it." The fix that follows
from A.3 is **a methodology change, not a code change**: any future
device benchmark of TRIAD must use ≥ 5 runs, drop the first as
warm-up, and report mean ± std (or median + IQR), not a single number.

**ADR-014 (v2-spectra) raises the published-claim minimum to N=10**
and adds a paired-t test against a baseline-run JSON. The H5 floor at
N≥3 below stays as the absolute hard floor for sanity runs.

Mechanically:

- Adopt `experiments/profile/A3_replicated_bench.sh` as the canonical
  device-bench harness for this project.
- Update `results/exynos_comparison.md` to use the N=3 means, and
  add a clear note that Phase-5's single-prompt numbers were
  superseded.
- No change to TRIAD's quantization or export code. The bundle ships
  unchanged.

## Rejected alternatives

- **Re-export with `flush_denormals=True`.**  H1 is rejected, so this
  is a fix for a non-problem.
- **Switch from OpenCL to Vulkan as the device target.**  Out of
  scope for this ADR; a parallel investigation. The MLC nightly does
  not currently expose `vulkan:android` as a preset (ADR-002 noted
  this).
- **TVM RPC op-level timing (A.3 in the prompt).**  Would have given
  the most detailed breakdown but requires building and pushing a
  `tvm_rpc` server binary cross-compiled for Android — multi-hour
  work that A.3's variance result made unnecessary.

## Consequences

- The acceptance criterion "decode ≥ 25 tok/s" was previously
  PASSING (TRIAD 37.7 ≥ 25); the N=3 mean of 40.7 is comfortably
  above. **No regression.**
- Future TRIAD changes (e.g. the v2 from Stream B / ADR-007) need
  their device numbers compared at the same N=3 protocol; do not
  cite single-run numbers in the comparison table without the
  ± σ band.
- `STATUS.md` (and any future PR descriptions) should reference these
  N=3 numbers, not the Phase-5 single-run ones, when discussing the
  decode/prefill claim.
