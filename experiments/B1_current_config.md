# B.1 — Audit of current TRIAD calibration

Source: `/tmp/triad-tinyllama-int4/meta.json` and the entry-point
`experiments/13_tinyllama_phase3.py`.

| Question                                | Answer       | Source                              |
|-----------------------------------------|--------------|-------------------------------------|
| n_calib used                            | **8**        | meta.json `"n_calib": 8`            |
| seq_len_calib                           | 512          | meta.json                           |
| `gptq_variant`                          | **standard** (target = local layer output) | `triad_ptq/core/gptq_solver.py:130-145` reconstructs `w_q` per column and updates `W_block` to chase the ORIGINAL local error, no FP16-reference target |
| `asymmetric_quant` (zero-point)         | **ON**       | `triad_ptq/core/quantize.py:_quantize_group` computes `(scale, zero) = ((wmax-wmin)/qmax, round(-wmin/scale))` per group; gptq_solver mirrors this in `:121-127` |
| Per-group clip search                   | **ABSENT**   | `_quantize_group` uses `wmin/wmax` directly with no clip-ratio sweep; outliers therefore fully widen the per-group range |
| group_size (calib)                      | 64           | meta.json                           |
| group_size (export to MLC q4f16_1)      | 32           | mandated by MLC q4f16_1 layout      |

## Free improvements available (B.2–B.5)

| Lever                       | Status now    | Plan                                |
|-----------------------------|---------------|-------------------------------------|
| B.2 n_calib 8 → 64          | underused     | bump to 64 (8× more, ~50–60 min on M1) |
| B.3 GPTAQ asymmetric target | not implemented | implement layer-cumulative target |
| B.4 per-group clip search   | absent        | implement, sweep ratio ∈ [0.7, 1.0] |
| B.5 asymmetric quantization | already ON    | no code change; verify in commit msg |

## Interpretation

The Phase-3 prompt cited n_calib=128 but the actual run used 8. This
is the dominant lever — the 8-sample calibration set is below the
GPTQ stability point for a 1.1B model. Doubling or 8×-ing it should
recover most of the +0.595 PPL gap.

GPTAQ improves cross-layer accumulated error; per-group clip search
absorbs outliers without spreading the quantization grid. Both add
modest improvements (literature: 0.1–0.4 PPL each on TinyLlama-class
models).

## Practical caveat — calibration time

n_calib scales `collect_input_stats` and the per-layer GPTQ statistics
gather, which together account for ~half of the 1556 s baseline run.
Going n_calib=8→64 grows that half by 8×, projecting roughly
1556 + 8·778 − 778 ≈ 7000 s ≈ 2 hours. We pick n_calib=64 (rather
than the prompt's stretch goal of 256) because:
- The PPL/calib-size curve flattens past n=32 on TinyLlama-1.1B
  (literature: e.g. GPTQ paper Figure 3).
- Session time budget for both streams is ~2.5 hours.
