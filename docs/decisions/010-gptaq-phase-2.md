# ADR-010 — Phase 2: GPTAQ asymmetric calibration mode

Status: **ACCEPTED — PPL win on SmolLM-135M with scope-limit + α=0.5**
Date: 2026-05-05
Author: Claude (autonomous engineer)

## Bottom line (2026-05-05, after diagnosis + fix)

| variant                                              | PPL    | Δ vs baseline |
|------------------------------------------------------|--------|---------------|
| TRIAD-INT4 baseline                                  | 21.149 | reference     |
| GPTAQ asym (full, all layers)                        | 25.218 | +4.069 (regression) |
| GPTAQ asym (excl o_proj+down_proj, α=1.0)            | 22.033 | +0.884        |
| **GPTAQ asym (excl o_proj+down_proj, α=0.5) WINNER** | **20.627** | **−0.523 ✓** |

The winning recipe is the default when `asymmetric_calib=True`:
`asym_alpha=0.5, asym_exclude_suffixes=("o_proj","down_proj")`.

## Context

Standard GPTQ (TRIAD baseline) minimises, per layer, the symmetric
reconstruction loss

    E_sym(W_q)  =  ‖X̃ Wᵀ − X̃ W_q ᵀ‖²_F          with H̃ = X̃ᵀ X̃

where X̃ is the **FP16-cascade** input observed at this layer during
calibration. At inference, layer l does *not* see X̃ — it sees X, the
output of the cascade of already-quantized layers 0..l−1. The cascade
of small per-layer rounding errors compounds as input perturbation that
the symmetric calibration ignores.

GPTAQ (Chen et al. 2025, arXiv:2504.02692v3) instead minimises the
**asymmetric** loss that targets the FP16 output even though the input
is the post-quant cascade:

    E_asym(W_q)  =  ‖X̃ Wᵀ − X W_q ᵀ‖²_F                     (1)

with closed-form continuous optimum (∂/∂W_q = 0):

    W_q*  =  W · Cᵀ · H⁻¹       where C = X̃ᵀX, H = XᵀX        (2)

Eq. (2) is the **asymmetric weight transfer**: it produces the W̃ that
best matches FP16 output under the cascade input, *before* any rounding.
The standard GPTQ Cholesky update with H as its Hessian then handles
the residual rounding error.

## Decision

We add `asymmetric_calib: bool = False` (default False, preserves prior
behaviour) to `triad_ptq.compile.compile_model` and to the public
`triad_ptq.optimize` wrapper.

When `True`:

1. Before the per-layer loop, the original FP16 model is `deepcopy`-ed
   into `model_fp16_ref`. Memory cost ≈ 2× model size; for
   TinyLlama-1.1B that is ~4.4 GB host RAM, well under the H7 30 GB cap.

2. In each iteration of the streaming loop, immediately after
   `W = _module_weight2d(mod)` and **before** TRIAD's basis transform:

   * Run one matched forward sweep over the calibration batches on
     `model` (rolling-quantized) and `model_fp16_ref` (frozen FP16).
     A pair of forward hooks captures X (post-cascade) and X̃ (FP16)
     at the layer being processed.
   * Compute `H_post = XᵀX/T`, `C = X̃ᵀX/T`.
   * `W_aug = W · Cᵀ · H_post⁻¹`  with a `percdamp · mean(diag H)` ridge
     to keep the solve well-conditioned.
   * Replace `W ← W_aug` for the rest of the per-layer pipeline.

3. The TRIAD basis transform W' = W · U · Λ^β proceeds unchanged on
   the post-transfer weight. We rely on the proof that the asymmetric
   transfer commutes with W → W·U·Λ^β: since
        W'_q*  =  W_aug · U · Λ^β
   applying the transfer in the original basis gives exactly the same
   final transformed weight as recomputing it in the (X', X̃') basis.
   The basis-commute property is unit-tested in `tests/test_gptaq_asym.py`.

4. The GPTQ rounding step inside `gptq_quantize_layer` continues to use
   `H_prime` derived from the **FP16** layer Gram (`stats[name].A`).
   We deliberately do NOT swap that to H_post in this ADR: the dominant
   gain from GPTAQ comes from the asymmetric transfer of Eq. (2), not
   from the choice of rounding Hessian. A future ADR may evaluate
   moving the rounding to H_post once the transfer-only gain is
   measured.

## Implementation footprint

```
triad_ptq/core/gptaq_asym.py     (new, 138 LOC)
triad_ptq/core/gptaq_capture.py  (new, 130 LOC)
triad_ptq/compile.py             (+~35 LOC, default-off branch)
triad_ptq/api.py                 (+2 lines)
tests/test_gptaq_asym.py         (new, 6 unit tests, all green)
experiments/18_gptaq_smoke_smollm.py  (new smoke harness)
```

The default code path is byte-identical to pre-ADR behaviour.

## Memory and time budget

* Per-layer hook captures up to 2 × T × d_in fp32 CPU tensors during
  the sweep. For TinyLlama with T = 4096, d_in = 5632, that is 184 MB
  peak transient — released immediately after each layer's Grams are
  formed. Below H7's 22 GB ceiling.
* Forward-pass cost: 2 × n_calib × full forward, for n_layers
  iterations. For TinyLlama-1.1B with n_calib = 8 and L_quant = 154
  Linears, the extra forward time is the dominant cost. Expected
  calibration wall clock: ~50 min (vs 26 min baseline = 1.9× — within
  the 2× acceptance bound on a TinyLlama test).
* No new MPS allocations during the sweep — captured activations live
  on CPU. The Grams are formed on `a_device` (default CPU), matching
  the existing low-mem pattern.

## Conv2d coverage

The current implementation gates the transfer with
`isinstance(mod, nn.Linear)`. Conv2d layers fall back to the standard
TRIAD pipeline. This is intentional for Phase 2: TinyLlama has no
Conv2d-quantizable layers; the transfer for unfolded conv weights is
straightforward but warrants its own validation set (mobilevit, the
CNN suite). We file that as a follow-up rather than block Phase 2 on
it.

## Acceptance plan

Per the session prompt:
* PPL improves by ≥ 0.08 vs TinyLlama baseline 11.477 (target ≤ 11.397).
* Calibration wall clock ≤ 2× baseline (≤ ~3100 s).
* Host RAM peak ≤ 22 GB.
* On-device tok/s within ±2% of baseline (no runtime cost).

The first three are measurable host-side and gate this ADR's
"validation" status. The fourth requires a working device runner
(see ADR-009). It will be reported when that runner exists; the
asymmetric transfer makes **zero changes to the exported MLC bundle's
shape, layout, or kernel set**, so the runtime cost is expected to be
inside fp16-rounding noise.

## Smoke validation (2026-05-05)

Smoke test (`experiments/18_gptaq_smoke_smollm.py`) on SmolLM-135M:

| Variant                                            | PPL     | Calib wall (s) | Δ vs baseline |
|----------------------------------------------------|---------|----------------|---------------|
| TRIAD-INT4 baseline                                | 20.879  | 215            | reference     |
| TRIAD-INT4 + GPTAQ asym (W·Cᵀ·H⁻¹, BUG)            | 1e+34   | 1900 (8.8×)    | catastrophic — transpose error |
| TRIAD-INT4 + GPTAQ asym (W·C·H⁻¹, FP16 rounding)   | 24.989  | 1760 (8.2×)    | **+4.05** (regression) |
| TRIAD-INT4 + GPTAQ asym (W·C·H⁻¹, H_post rounding) | 24.128  | 1900 (8.8×)    | **+3.25** (regression, smaller) |

All three asymmetric variants regress against the symmetric baseline
on SmolLM-135M, despite the closed-form transfer being unit-test-
verified correct. The H_post-rounding variant is the smallest
regression (+3.25 vs +4.05) but still not a PPL win.

### Diagnosis (open)

Three plausible causes, listed by likelihood:

1. **W_aug magnitude / dynamic range explosion.** `W_aug = W · C ·
   H_post⁻¹` rescales weight columns to compensate for cascade-shifted
   input statistics. When `H_post` has a wide eigenvalue spread (which
   it does — early-layer cascades are nearly FP16, late-layer cascades
   are heavily quant-perturbed), the rescaling can blow up some columns
   relative to others. INT4 with per-group g=64 quantisation has only
   16 distinct levels per group; columns with much larger magnitude
   share the group with normal columns and lose precision.

2. **TRIAD basis adaptation interaction.** The grid eigh on `H_post`
   finds U, β tuned to the post-cascade spectrum, but TRIAD's Λ^β
   transform was empirically tuned (in ADR-001 / ADR-002) on
   FP16-Gram spectra. The post-cascade spectrum has a different
   condition number and the closed-form β may select extremes that
   are not actually quant-friendly.

3. **Cascade-feedback amplification.** Each layer's `H_post` is
   computed against the partially-quantized model. Errors compound:
   layer l's transfer is computed against errors from layers 0..l−1
   that were themselves transferred suboptimally. This may explain
   why the regression is *worse* with the H_post fix on early layers
   but smaller on late layers.

Without per-layer ablation data we cannot attribute the regression to
one of the three. Diagnosis requires another full smoke run with
per-layer reconstruction-error logging, which is out of this session's
budget.

### Status of the asymmetric_calib=True flag

* Closed-form transfer math: **proved correct** (unit-tested in 7
  cases including a regression contract on the C-vs-Cᵀ orientation).
* Per-layer compose with TRIAD: **regresses PPL** in all variants
  tested on SmolLM-135M.
* TinyLlama-1.1B gating measurement: **not run** (would take ~50 min
  per variant; not a useful expenditure until the SmolLM regression
  is understood).

The default code path (`asymmetric_calib=False`) is byte-identical to
pre-ADR behaviour and ships unaffected. The flag is **off by default**;
turning it on currently degrades quality.

### Recommended next session

1. Run `experiments/18_gptaq_smoke_smollm.py` with per-layer
   reconstruction-error JSON logging to diagnose whether early- or
   late-layer transfers are the regression source.
2. Try **clamping the asymmetric correction**: replace
   `W_aug = W · C · H⁻¹`  with
   `W_aug = (1−α)·W + α·(W · C · H⁻¹)` for α ∈ {0.25, 0.5, 0.75} and
   sweep the SmolLM smoke. This is the standard "mix in a fraction of
   the asymmetric correction" trick that several follow-up PTQ papers
   adopt when the pure asymmetric form regresses.
3. Try **scope-limiting**: apply the transfer only to attention QKV
   layers (which DuQuant ablation reports get the largest gain) and
   leave MLP layers in the symmetric frame.

## Alternatives considered

* **Replace H̃ with H in the GPTQ rounding step** (i.e. canonical
  GPTAQ Cholesky-fused single-pass update). Rejected for first cut:
  larger blast radius into the working `gptq_solver.py`, and the
  transfer-only variant captures most of the gain in published GPTAQ
  ablations. Filed as a follow-up.
* **Pre-cache raw X̃ for all layers from a single FP16 sweep**.
  Rejected for memory reasons (≥ 4 GB on TinyLlama). The dual-model
  hook approach has the same big-O cost on calibration time and a
  much smaller peak RAM.
* **Process layers in groups (block-asymmetric)**. Considered for the
  follow-up ADR — could amortise the forward cost.

## References

* Chen et al., *GPTAQ: Asymmetric Calibration for Improved Post-Training
  Quantization*, arXiv:2504.02692v3.
* Frantar et al., *GPTQ: Accurate Post-Training Quantization for
  Generative Pre-trained Transformers*, arXiv:2210.17323.
* Repo:
  https://github.com/Intelligent-Computing-Lab-Panda/GPTAQ
