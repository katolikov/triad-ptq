"""Phase B — Squisher Fisher diagonal sensitivity router.

Reference: arXiv:2507.18807. Replaces v1's empirical-Fisher KFAC + the
noise-injection ρ probe (which the v1 trace router never actually used
because it collapsed to {3, 8} watershed and shipped uniform-only).

Design summary
--------------
During the calibration forward pass, accumulate

    v_t = γ · v_{t-1} + (1 − γ) · g_t²

per parameter, where g_t is the gradient of the BRECQ block-output
reconstruction loss (we have no labels, so we cannot use LM loss).

For 100 Adam steps per block we use γ = 0.9 (the paper's γ = 0.999
requires longer accumulation). Hutchinson sanity check
(triad_ptq/_v2/router/hutchinson_check.py) must show pearson ≥ 0.7
between Squisher and the true diagonal Fisher.

Then ρ^(ℓ) per block is derived as
    ρ^(ℓ) = ‖∇_output L_block‖² / ‖∇_input L_block‖²
which feeds Phase F's GPTAQ ρ-weighted α scheduling.

Status
------
NOT IMPLEMENTED — Phase B placeholder.
"""
from __future__ import annotations

IMPLEMENTED = False


def squisher_fisher_diagonal(*args, **kwargs):
    raise NotImplementedError(
        "Phase B (Squisher Fisher diagonal) is not yet implemented; "
        "v2 currently delegates to v1's noise-injection ρ probe."
    )
