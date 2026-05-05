# ADR-009 — Phase-0.4 baseline reproduction blocked by missing on-device runner

Status: **accepted (deferred)**
Date: 2026-05-05
Author: Claude (autonomous engineer)

## Context

The Phase-0 plan (this session's prompt) lists, under Phase 0.4:

> Re-run the BASELINE bench on TRIAD-INT4 to confirm 25.3/40.7 tok/s
> still reproduces (within ±5%). If it does NOT reproduce, STOP and
> write ADR-002 documenting what changed.

The same prompt also cites a reference baseline that disagrees with
last session's measured numbers in `STATUS.md`:

| Metric                 | Prompt's "baseline" | Measured 2026-05-05 (STATUS.md) |
|------------------------|---------------------|---------------------------------|
| TRIAD-INT4 prefill     | 25.3 tok/s          | **18.2** tok/s                  |
| TRIAD-INT4 decode      | 40.7 tok/s          | **37.7** tok/s                  |
| Ref q4f16_1 prefill    | 25.5 tok/s          | 25.2 tok/s                      |
| Ref q4f16_1 decode     | 41.6 tok/s          | 42.9 tok/s                      |
| TRIAD-INT4 PPL         | 11.477              | 11.477 (matches)                |
| Peak GPU memory        | 789 MB              | 789 MB (matches)                |

The reference q4f16_1 numbers are within ±5%, which is tolerance. The
**TRIAD-INT4 throughput** numbers are not — prefill is 39% higher in the
prompt and decode is 8% higher. PPL and peak GPU agree exactly.

The prompt's bench protocol calls for `prompt-len 128 / gen-len 128`.
The previous session (per STATUS.md) drove a 28-token prompt through the
in-app MLCChat UI and read its on-screen prefill / decode tok/s line.
**These are different measurement protocols**, which plausibly explains
the prefill delta (a 28-token prefill is dominated by single-token
launch overhead; a 128-token prefill amortises that across more tokens
and produces a higher prefill tok/s). The 8% decode delta is closer to
what we'd attribute to thermal / governor variance.

## The harness gap

The prompt's bench scaffold reads:

```
adb push build/android/triad-runner /data/local/tmp/triad/
adb shell "cd /data/local/tmp/triad && ./triad-runner ..."
```

and refers to `tools/bench_android.sh`. **Neither exists in this repo.**
There is no committed CLI device runner; the previous session's only
on-device measurement vector was the custom-built MLCChat APK
(`/tmp/MLCChat.apk`) driven manually through its chat UI. That harness
is not automatable from `adb shell` without either screen-OCR'ing the
on-screen tok/s readout or an MLCChat broadcast intent we have not
audited.

## Decision

1. **Phase-0.4 baseline reproduction is deferred** until a CLI on-device
   runner exists. We will NOT fabricate baseline numbers and we will NOT
   re-run via the manual UI path solely to satisfy 0.4 — that would not
   match the prompt's `prompt=128 / gen=128` protocol anyway.
2. The headline reference numbers used as "baseline" for subsequent
   acceptance gates in this session are the **measured** ones from
   `STATUS.md`, *not* the prompt's 25.3 / 40.7 figures. Specifically:
     - Decode-tok/s baseline = **37.7 tok/s** (Phase 1 target +1.5 ⇒ ≥ 39.2).
     - Prefill-tok/s baseline = **18.2 tok/s** (Phase 1 target +2 ⇒ ≥ 20.2).
     - PPL baseline = 11.477.
     - Peak GPU baseline = 789 MB.
   Where the prompt's targets reference the higher numbers, we treat
   them as stretch goals and report against both.
3. Building a CLI on-device runner is itself non-trivial (MLC-LLM's
   on-device runtime is normally exercised through the JNI of MLCChat).
   We **will attempt** a minimal runner using `mlc-chat-cli` (the C++
   binary that ships with mlc-llm builds) cross-compiled for Android,
   but mark this as best-effort — if it does not build cleanly we
   document the failure and continue with host-only phases (Phase 2,
   Phase 4, Phase 5) which produce ADR-able PPL deltas without device
   measurement.
4. The same numbers must be re-measured at the close of any phase whose
   acceptance gate depends on tok/s. Those re-measurements will use
   whatever harness exists at the time, will be N≥3 with 60 s cooldown
   per H5, and will be honest about which protocol they used (prompt
   length, gen length, app-driven vs. binary-driven).

## Consequences

* Phase 0 ships its primary deliverable (the probe + SUMMARY.md) and
  carries a known TODO for 0.4.
* Phase 1 (q4f16_0 layout swap) acceptance is conditional on the runner
  existing. If the runner is not built in this session, Phase 1 ships
  the host-side export change + dual-format compile artefacts only, and
  marks the on-device numbers as `[NOT RUN ON DEVICE — runner missing]`
  per H4.
* Phases 2 and 4 are pure host-side (calibration / weight transform) and
  ship full deltas in PPL with no device dependency.
* The autonomous-session push criteria (all 3 acceptances + tests +
  clean tree) cannot be met without a working device bench. **No push
  to remote will be attempted.**

## Alternatives considered

* **Drive MLCChat UI via `adb shell input` + screencap-OCR.** Rejected
  for this session: latency is significant, and OCR'ing the on-screen
  tok/s readout is brittle. If a future session needs it, the steps are
  (a) `am start -n ai.mlc.mlcchat/.MainActivity`, (b) tap into model,
  (c) `input text "..."`, (d) `input keyevent 66` (enter), (e) wait for
  generation, (f) `screencap -p` and OCR the prefill/decode line.
* **Patch MLCChat to emit JSON over logcat.** Cleaner than OCR but
  requires re-building the APK from source, which the previous session
  showed costs ~30 min wall-clock + JDK21 toolchain dance. Saved as a
  Phase-1.5 follow-up if the throughput-relevant phases (1, 3, 7) end
  up being the binding constraint.
