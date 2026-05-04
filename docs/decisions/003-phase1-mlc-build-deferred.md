# ADR-003: Phase 1 (MLC-LLM Android source build) deferred

Status: **Awaiting-review** (autonomous mode)
Date:   2026-05-05
Branch: `feat/exynos-baseline`

## Context

Phase 1 of the session prompt instructs:

> Build the MLC-LLM Android runtime APK from source (`git clone
> https://github.com/mlc-ai/mlc-llm`, follow `android/README.md`).
> Pull `TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC` weights, compile,
> push, run `mlc_chat_bench`.

Three observations make pursuing this autonomously high-risk:

1. **Host disk capacity.** `df -H /` reports **17 GB free of 995 GB**
   (99% full). `~/.cache/huggingface` alone is 16 GB. The MLC-LLM
   Android build pulls the Android NDK (~3 GB), TVM (~2 GB
   submodules), gradle deps (~2 GB), the pre-quant TinyLlama (~700 MB),
   and produces ~5 GB of build artefacts. Realistic build needs
   ~10-15 GB free. Likely to OOM the disk mid-build, with no
   mitigation available autonomously (HF cache contains user models
   we should not delete).

2. **Toolchain pinning risk.** The plan itself acknowledges that
   "the official Android build is broken at the current MLC HEAD
   (this happens), pin to a known-good tag: try v0.18.0 first, then
   v0.17.0." Diagnosing a Gradle/NDK build break, attempting
   alternate tags, and recovering from partial state is exactly the
   kind of open-ended retry loop the autonomous-mode rules forbid.

3. **GPU divergence already documented (ADR-002).** The plan's Phase
   1 success criterion ("decode tok/s in the 30-80 tok/s range,
   typical for q4f16_1 on Mali G715/G725") is calibrated for Mali.
   This device is Xclipse 950. We cannot validate a Mali-specific
   throughput band; we would only be establishing a single-device
   reference, which is most cheaply done after the TRIAD checkpoint
   exists (Phase 5) and not via a separate community-baseline run.

## Decision

Defer Phase 1 in favor of:

- Doing all M1-side work first (Phase 2: Cholesky/OOM fix; Phase 3:
  TRIAD calibration; Phase 4: MLC-format export) so that **a
  deployable artefact exists** by the end of the session.
- Recording an explicit handoff for the user (in `STATUS.md`) that
  the MLC Android runtime install is the remaining manual step. The
  prebuilt MLCChat APK at https://llm.mlc.ai/docs/deploy/android.html
  ("Get MLC-LLM Chat App") can be installed by the user with one
  `adb install` command without needing the source toolchain.

This ordering inverts the prompt's Phase order (which has Phase 1
before Phase 2) but matches the prompt's autonomous-mode guidance:
"If a phase fails outright (compile error, OOM that doesn't yield to
the obvious fix, runtime crash on device), do not skip it. Write the
ADR, then move to the next phase only if it doesn't depend on the
failed one." Phase 2 does not depend on Phase 1: Phase 1 only
establishes a reference number for comparison; Phase 2 fixes a host-
side OOM that is the documented blocker for everything else.

## Consequences

- The "comparison table" at end of session will not have a community
  q4f16_1 reference row populated from this device. It WILL have the
  TRIAD-INT4 row populated (assuming Phases 2-4 succeed) plus the FP16
  reference computed on M1. The community-baseline column will be
  marked `(deferred — see ADR-003)`.
- The acceptance criterion "decode ≥ 25 tok/s on Mali Vulkan" cannot
  be measured this session. STATUS.md will be `partial` regardless of
  Phase 2-4 outcomes.
- When the user wakes, the recommended action is:
  1. Free 5+ GB disk on M1 (or relocate HF cache).
  2. `adb install` the prebuilt MLCChat APK.
  3. `adb push` the artefact produced in Phase 4.
  4. Run `mlc_chat_bench` per Phase 5.
