#!/usr/bin/env bash
# tools/bench_android.sh — placeholder on-device bench runner.
#
# STATUS (2026-05-05): NOT IMPLEMENTED. The TRIAD-PTQ session-3 prompt
# refers to this script as the harness driving N=5 prompt=128/gen=128
# benches with 60 s cooldown, but no such harness exists yet. The
# previous session benched via the custom MLCChat APK's chat UI by
# manually typing prompts and reading the on-screen tok/s line.
#
# This script outlines the two paths considered (see ADR-009):
#
# A. JSON-over-logcat patch (recommended, fastest path)
#    Patch ai/mlc/mlcchat/AppViewModel.kt to emit a log line of the form
#       Log.i("triad_bench", JSON.stringify({"prefill": …, "decode": …,
#                                             "prompt_tokens": …, "gen_tokens": …,
#                                             "model_id": …}))
#    after each generation completes. Then this script does:
#       am start -n ai.mlc.mlcchat/.MainActivity ...
#       input text "<prompt>"
#       input keyevent 66
#       logcat -s triad_bench:I -d -t 1
#       compute mean ± stdev across N runs.
#    Estimated effort: 30 min APK rebuild + 10 min driver.
#
# B. mlc-chat-cli native binary (heavier; canonical TVM workflow)
#    Cross-compile the C++ mlc-chat-cli tool against
#    android-aarch64-clang for the same TVM version that compiled the
#    .tar bundle. This is mlc-llm's intended deployment path on
#    devices without the Java/Kotlin shell. ~3-5 h work for a fresh
#    cross-compile + smoke + driver.
#
# Until one of A or B is in place, this script does the only thing it
# safely can: refuse to fabricate device numbers and explain why.

set -euo pipefail

cat <<'NOTE' >&2
[bench_android] NOT IMPLEMENTED.

Acceptance gates that depend on this runner are currently marked
[NOT RUN ON DEVICE] in the various Phase ADRs (see docs/decisions/
009-baseline-numbers-and-runner-gap.md for the full diagnosis).

The previous session's manual MLCChat-UI workflow is documented in
results/exynos_comparison.md. To reproduce that workflow:

    adb shell am start -n ai.mlc.mlcchat/.MainActivity
    # tap the model in the in-app list
    # type a 28-token prompt, hit send
    # read the prefill / decode tok/s line that mlc_chat prints above
    # the assistant message
    # capture dumpsys meminfo immediately after

For an autonomous driver, the recommended next-session plan is to
patch MLCChat to emit JSON tok/s over logcat (see comment block at
the top of this script).
NOTE
exit 64  # EX_USAGE: command line usage error
