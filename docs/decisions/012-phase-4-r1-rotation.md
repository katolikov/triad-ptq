# ADR-012 — Phase 4: offline R1 Hadamard pre-rotation

Status: **proposed (implementation + unit tests committed; TinyLlama
forward-equivalence run + calibration deferred)**
Date: 2026-05-05
Author: Claude (autonomous engineer)

## Context

QuaRot (arXiv:2404.00456 §3.1) defines a family of orthogonal rotations
that sit at four points in a Llama-style transformer:

* **R1** — applied to the residual stream. Folds into the dense weights
  of attention QKV/O and MLP gate/up/down, plus embedding and lm_head.
  Zero runtime cost.
* **R2** — applied to the attention head dimension.
* **R3** — applied online inside the attention softmax block.
* **R4** — applied online inside the down_proj input.

R1 alone is the only rotation that absorbs into the FP16 weights with
no runtime cost. SpinQuant (arXiv:2405.16406) shows that at the 1.1B
scale a random Hadamard already comes within ~0.05 PPL of a Cayley-
optimised R1, so we use the random signed Hadamard variant — no
gradient-based optimisation, no per-model fine-tuning step.

## Decision

Add `triad_ptq/core/rotate.py` (273 LOC) with:

* `hadamard_matrix(d)` — Sylvester construction for d power-of-two.
* `random_signed_hadamard(d, seed)` — Q = H · diag(±1) with seeded
  signs (default seed `0xACE1`).
* `fold_rmsnorm_into_next(norm, [linears])` — multiply each linear's
  input axis by γ, then set γ ← 1.
* `rotate_linear_input` / `rotate_linear_output` /
  `rotate_embedding_output` — orthogonal-rotation wrappers.
* `R1Spec` + `apply_r1_to_block` — apply the per-block R1 sequence.
* `apply_r1_to_llama(model, seed)` — convenience wrapper for HF
  Llama-family models that walks all blocks + handles embedding,
  final norm, lm_head.

The function operates **in place** on the model's `nn.Linear` /
`nn.Embedding` weights. After return, the model produces (by the
QuaRot computational-invariance argument) outputs that are
mathematically identical to the original model up to fp32 rounding,
*even though* the dense weights have changed. The only externally
visible behaviour difference is that the residual-stream activations
(internal to the model) have a near-Gaussian distribution, which
shrinks per-group quant ranges for the downstream TRIAD calibration.

## Implementation footprint

```
triad_ptq/core/rotate.py                   (new, 273 LOC)
tests/test_r1_rotation.py                  (new, 5 unit tests, all green)
experiments/19_r1_rotate_tinyllama.py      (new — driver, not run in this session)
docs/decisions/012-phase-4-r1-rotation.md  (this ADR)
```

The compile.py per-layer pipeline is **not** modified. R1 runs as a
preprocessing step before `optimize(...)` is invoked.

## Acceptance plan (from session prompt)

* **Forward-pass cosine ≥ 0.9999** on 8 sample inputs against the
  unrotated FP32 baseline. Implemented in
  `experiments/19_r1_rotate_tinyllama.py`. Test runs end-to-end on the
  unit-test side already (`test_r1_invariance_on_tiny_block`).
* **PPL improves by ≥ 0.10 vs Phase 2 + Phase 3 output**. Requires
  TinyLlama calibration with R1-rotated weights as input. Deferred —
  the GPTAQ Phase-2 smoke run is currently consuming the M1 GPU; the
  Phase-4 calibration should chain on top of Phase 2's branch and run
  in the next session.
* **Tok/s within ±1% of previous best**. Requires device runner
  (ADR-009).

## Why we are confident in correctness

The unit test `test_r1_invariance_on_tiny_block` builds a minimal
Llama-block (RMSNorm → q/k/v + identity-attn → o → residual → RMSNorm
→ SwiGLU MLP → residual) on random fp32 weights and compares the
block's output before/after R1 application on identically-sized random
input. The cosine similarity is **> 0.9999** and the relative L2 error
is below 1e-3, both within the gate from the session prompt. The full
Llama wrapper `apply_r1_to_llama` is a straightforward composition of
the per-block primitive that the unit test covers, so the only
remaining failure mode is a Llama-specific structural mismatch —
which the TinyLlama experiment script will catch.

## Composition with Phase 2 (GPTAQ)

R1 is applied **before** TRIAD / GPTAQ calibration. The rotated
residual stream defines the new X̃ that GPTAQ asymmetric calibration
sees; the cross-Gram C = X̃ᵀX is well-defined in the rotated basis.
Because the rotation is exactly orthogonal, both the GPTQ Cholesky
update (via H_pre = X̃ᵀX̃) and the GPTAQ asymmetric transfer (via
H_post and C) remain numerically well-conditioned.

## Notes on R2/R3/R4

Out of scope for Phase 4 by explicit prompt instruction. R4 (online
FWHT for down_proj) is deferred to Phase 8, gated on Phase 4's PPL
result. R2 has measurable on-device cost on Vulkan even with
subgroupShuffleXor available (Xclipse 950 Phase-0 probe shows wave64
native; an FWHT in wave64 is exactly 6 butterfly stages).

## References

* Ashkboos et al., *QuaRot: Outlier-Free 4-bit Inference in Rotated
  LLMs*, arXiv:2404.00456.
* Liu et al., *SpinQuant: LLM Quantization with Learned Rotations*,
  arXiv:2405.16406.
* AMD Quark documentation on QuaRot R1+R2 integration.
