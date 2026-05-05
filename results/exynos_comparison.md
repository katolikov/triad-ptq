# TRIAD-PTQ on Exynos 2500 — final comparison

Acceptance criteria (top of session prompt):

- WikiText-2 PPL TRIAD-INT4 vs FP16: **≤ +1.0**
- Decode throughput on device (batch=1): **≥ 25 tok/s**
- Peak GPU memory during decode: **≤ 1.2 GB**

## M1-side checkpoint summary

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Calibration time (M1, fp32): 1555.6 s
- Peak MPS allocation during calib: 12.19 GB
- Simulated INT4 PPL on M1: 11.477 (on 4088 tokens)
- Checkpoint: `/tmp/triad-tinyllama-int4/model.pt` (4540.1 MB)

## On-device comparison

| Method | Bits | WikiText-2 PPL | HellaSwag | Tok/s decode | Peak GPU MB | Disk MB |
|---|---|---|---|---|---|---|
| FP16 (reference) | 16 | 10.882 | — | — | — | — |
| MLC q4f16_1 (community baseline) | 4 | — | — | — | — | — |
| **TRIAD-INT4 (this work, M1 quality only)** | 4 | 11.477 | — | — | — | — |
|  |  |  |  |  |  |  |
| _PPL delta_ (10.882 → 11.477) |  | +0.595 | _acc ≤ +1.0_ | PASS |  |  |

## Notes

- MLC q4f16_1 community baseline row is empty: Phase 1 deferred per ADR-003.
- TRIAD-INT4 device row is empty: Phase 5 device bench requires the manual MLC runtime install (see STATUS.md).
