# ADR-008: Linearise the nine Exynos branches into `main` for v0.2.0-alpha

Status: **Accepted** (release tag `v0.2.0-alpha`)
Date: 2026-05-05

## Context

Across three sessions (Phases 0–5 + Stream A + Stream B), nine
feature branches were created in chain order:

```
feat/exynos-baseline           Phase 0 device inventory + ADR-002/003
feat/exynos-cholesky-fix       Phase 2 streaming calibration refactor
feat/exynos-calib-tinyllama    Phase 3 TinyLlama-1.1B INT4 calibration
feat/exynos-mlc-export         Phase 4 (initial direct exporter)
feat/exynos-bench              Phase 5 scaffold + FP16 baseline + comparison.md
feat/exynos-mlc-compile        Phase 1.1 + Phase 4 redo via mlc_llm convert_weight + ADR-004/005
feat/exynos-bench-device       Phase 5 device bench, custom APK, Phase-5 PASS
feat/exynos-profile-gap        Stream A: ADR-006 (gap was noise, N=3 protocol)
feat/exynos-improve-ppl        Stream B: clip_search + clamp + ADR-007
```

The first seven are already a strict ancestor chain because they were
created sequentially. The two Stream branches (A and B) both branched
off `feat/exynos-bench-device`, so B is *not* an ancestor of A.

## Decision

Rebase Stream B onto Stream A, fast-forward `main` to Stream B's tip,
no squash. Two bookkeeping commits added on top:

1. `tests/test_generation_smoke.py` + `triad_ptq/testing/` — 4-gram
   repetition detector against a SmolLM-135M INT4 fixture, opt-in
   build via `TRIAD_BUILD_FIXTURES=1`. Skipped by default to keep
   `make test` fast. Direct regression guard for the ADR-007 class
   of bug.
2. `CHANGELOG.md` for `v0.2.0-alpha`.

## Conflicts encountered

The rebase of Stream B onto Stream A applied cleanly with no
conflicts. The two streams touched disjoint files:

- Stream A: `docs/decisions/006-…`, `experiments/profile/`,
  `results/exynos_comparison.md`.
- Stream B: `docs/decisions/007-…`, `experiments/B1_…`,
  `experiments/16_…`, `experiments/17_…`, `experiments/v2-artifacts/`,
  `experiments/screenshots/v2-…`, `tests/test_clip_search.py`,
  `tests/test_super_weight_index_bounds.py`, `triad_ptq/api.py`,
  `triad_ptq/compile.py`, `triad_ptq/core/gptq_solver.py`,
  `triad_ptq/export/hf_safetensors.py`,
  `results/triad_tinyllama_int4_v2_m1.json`,
  `results/phase4_v2_export_summary.json`.

The only file both might have touched —
`results/exynos_comparison.md` — was modified only by Stream A in
this rebase window (the v1 → N=3 update). Stream B's note about v2
in the comparison table was deferred to a follow-up because v2 was
not deployed (per ADR-007).

## Rollback

Each branch tip was tagged `archive/<branch-name>-pre-rebase`
before rebasing. To revert v0.2.0-alpha:

```bash
git tag --list 'archive/feat/exynos-*-pre-rebase'
# Find the pre-rebase tip of the branch you want to restore
git checkout main
git reset --hard <commit-before-FF>
```

Archive tags are pushed to origin alongside the release tag.

## Consequences

- Linear `main` history from `508471e7…` (origin/main pre-session)
  through `v0.2.0-alpha`.
- The seven existing draft PRs (#1–7) are now subsumed by `main`;
  GitHub will mark them merged when their HEADs are detected on the
  default branch.
- Stream A and Stream B were never opened as draft PRs (kept local
  in their respective sessions); they ship implicitly via the
  fast-forward.
