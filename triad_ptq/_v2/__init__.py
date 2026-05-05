"""TRIAD-PTQ v2 (codename: SPECTRA-Q).

This package is intentionally a *skeleton* during Phase A of the v2 migration.
Each subpackage will be populated in its own phase (B–G); the v1 code in the
parent package remains untouched until Phase H wires everything together.

Layout
------
- rotation.sign_perm         Phase C — block-diagonal random sign+permutation
                             (replaces v1 R1 Hadamard).
- transform.learnable_beta   Phase D — single learnable scalar β per block,
                             trained via BRECQ-style block reconstruction
                             (replaces closed-form β* of v1 eq. 5).
- router.squisher            Phase B — Squisher Fisher diagonal (replaces
                             empirical-Fisher KFAC + noise-injection ρ probe).
- superweight.channel_int8   Phase E — channel-grained mixed precision
                             (replaces FP16 sparse super-weights).
- lwc.selective              Phase D — OmniQuant-style selective LWC.
- groupsize.sweep            Phase G — hardware-aware G ∈ {32, 64, 128}.

Status
------
All modules below currently expose `IMPLEMENTED = False` and raise
`NotImplementedError` if invoked. The v2 path through `triad_ptq.optimize`
short-circuits to v1 unless the relevant phase has landed.
"""

__version__ = "2.0.0-alpha0"
ALGORITHM_NAME = "SPECTRA-Q"
