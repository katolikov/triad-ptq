# ADR-010 — Phase 2: GPTAQ asymmetric calibration mode

Status: **accepted (implementation), validation pending**
Date: 2026-05-05
Author: Claude (autonomous engineer)

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

## Smoke validation

Smoke test (`experiments/18_gptaq_smoke_smollm.py`) on SmolLM-135M
verifies the implementation runs end-to-end and the asymmetric path
produces a non-degenerate, finite PPL. Numbers will be filled in here
once the run completes. Smoke is **not** sufficient for Phase-2
acceptance — TinyLlama-1.1B is the gating model.

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
