# ADR-001: truncated_eigh in TRIAD grid causes rank deficiency

## Status
Rejected. v0.1.0-alpha, 2026-05-04.

## Context
The roadmap proposed replacing safe_eigh with truncated_eigh (top-k
randomized SVD) in core/grid.py, motivated by the "5-10 s per d=2048
layer" eigh bottleneck reported in the v0.1.0-alpha README.

## Implementation
Branch feat/truncated-eigh added truncated_eigh to utils/device.py
and switched compute_grid to use it via kwarg eigh_topk=128.

## Tests
- test_truncated_eigh_matches_full: PASS (eigvals rel-err <1e-4)
- test_truncated_eigh_recovers_beta_star: PASS (|Δβ*| < 0.005)
- test_grid_unchanged_under_truncation: PASS (|Δβ*| < 0.02)
- SmolLM-135M end-to-end PPL: 21.521 → 81.531 (+60.01, FAIL)

## Root cause
The transformation W' = W·U·Λ^β is applied to W. Truncating U to
shape (d, k=128) on a d=576 layer means TriadLinear.forward implicitly
multiplies the orthogonal complement (448 dimensions of x) by zero.
Eq. (5)'s "(s_k)² weighting" justifies top-k recovery of β*, not of
the full rotation. Property tests verified the wrong invariant:
β* convergence, not rank preservation of W'.

## Alternatives considered
- (A) Use truncated_eigh for β* only, full eigh for U: 8 lines of
  code with no calibration speedup (eigh is still called).
- (B) Full-rank padded approximation V = U_k Λ^β U_k^T + (I - U_k U_k^T):
  materializes a d×d matrix; defeats memory savings; still requires
  full eigh for verification.

## Decision
Rejected. Future work (C): replace eigh with a cheaper *full*
decomposition (LOBPCG / block-Lanczos with warm start across layers).
This preserves rank and targets the actual bottleneck on d≥960 layers.

## Lessons for the roadmap
- The next attempt must include a test_grid_full_rank_preserved
  property that fails on truncation.
- Speed benchmarks must be measured on d≥960 (SmolLM-360M FFN), not
  d=576 (SmolLM-135M); the latter doesn't exercise the bottleneck.
- Profile first, fix second.
