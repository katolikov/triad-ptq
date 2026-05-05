# ADR-015 — Recommended group size default for the Xclipse 950 deployment

Status: **Provisional — Mali measurement pending** (in scope: v2-spectra Phase G)
Date: 2026-05-06
Branch: `v2-spectra`

## Context

v1 ships with the MLC-canonical group size G = 32. v2 adds a hardware-
aware sweep over G ∈ {32, 64, 128} (Phase G) on the basis that:

1. The Xclipse 950 native subgroup size is **64** (wave64), confirmed by
   the Phase-0 Vulkan probe in `docs/probe/SUMMARY.md`.
2. G=64 halves the FP16 scale traffic per matmul vs G=32 (one fp16 per
   64 weights instead of per 32).
3. v2's Phase-C block-diagonal rotation is itself block-aligned at G, so
   moving G also moves the rotation block size — the per-group MSE under
   sign+perm is preserved exactly (proved in `tests/v2/
   test_sign_perm_rotation.py::test_per_group_max_invariance_under_sign_perm`).

The v2 plan (Phase G2) calls for setting **G=64 as the new recommended
default for the Xclipse 950 deployment path**, gated on measured Mali
decode tok/s being ≥ G=32.

## Decision

The recommended default value of `group_size` for `optimize(...,
algorithm='v2', ...)` is **set by `decide_default_group_size(sweep_result)`
in `triad_ptq/_v2/groupsize/sweep.py`**, which strictly requires both
G=32 and G=64 to have been measured ON THE TARGET DEVICE (decode_tps
populated). Until that measurement runs:

- The v2 path uses `group_size=64` as a CANDIDATE default
  (`RECOMMENDED_DEFAULT_G`) only for static-MB falsification checks via
  `estimate_disk_mb`, not for any runtime claim.
- No README, STATUS, or REPORT text claims G=64 is the default.
- Any G < 32 is rejected by the v2 entry point (the q4f16_1 MLC layout
  requires G ≥ 32 for valid scale alignment).

## Why "candidate" until measured

We have observed in v1 (ADR-006) that small reasoning chains about Mali
performance are unreliable; the Xclipse 950 OpenCL driver has known
scheduling pathologies in the 2 KiB–8 KiB workgroup region. A G=64
bundle is a modestly different workgroup distribution (per-group scale
fetch frequency halved); whether the Xclipse 950 scheduler handles that
well is an empirical question, not a derived one.

The measurement protocol:
1. Calibrate v2 at G ∈ {32, 64, 128} on TinyLlama-1.1B (Phase H runbook).
2. Export each to MLC q4f16_1 with the appropriate `group_size` flag.
3. Install three APKs on Galaxy Z Flip7 (one per G).
4. Run `tools/bench_android.sh` with `BENCH_GROUP_SIZE=$G` for each,
   N=10, 60 s cooldown (ADR-014).
5. Feed the resulting JSONs into `decide_default_group_size`.
6. If decode(G=64) ≥ decode(G=32) within paired-t p < 0.05 → set
   `RECOMMENDED_DEFAULT_G = 64`. Otherwise revert to 32 and update this
   ADR's status to "Accepted: G=32".

## Consequences

- The unit-test suite verifies the harness logic and disk-MB falsification
  but does NOT make a default-G claim. `tests/v2/test_group_size_sweep.py`
  asserts `RECOMMENDED_DEFAULT_G == 64` only as a CANDIDATE constant.
- The v2 design doc's tentative claim "G=64 halves FP16 scale traffic"
  is correct (it's a static structural fact) and stays in the README.
  The runtime-speed implication of that claim is gated.
- If the Mali measurement comes back showing G=64 slower, this ADR is
  updated rather than removed; the v2 plan's caveat #5 specifically
  asks for reverting with measured data.

## References

- v2 plan, Phase G.
- ADR-014 (N=10 paired-t protocol).
- `tools/bench_android.sh` (Phase G annotation `BENCH_GROUP_SIZE`).
- `triad_ptq/_v2/groupsize/sweep.py` (harness + estimator + decider).
- Phase-0 Vulkan probe: `docs/probe/SUMMARY.md`.
