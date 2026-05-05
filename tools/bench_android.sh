#!/usr/bin/env bash
# tools/bench_android.sh — autonomous on-device bench driver for TRIAD-PTQ.
#
# Drives the patched MLCChat APK on the connected Android device:
#   1. Switches the foreground activity to MLCChat.
#   2. Taps the chat icon for a given model (so the engine loads weights).
#   3. Types a fixed prompt + taps send.
#   4. Waits for completion, parses the JSON line emitted by the APK
#      patch on the `triad_bench` logcat tag.
#   5. Repeats for N iterations, sleeping between iterations to keep the
#      SoC out of thermal throttling (per session prompt H5).
#   6. Reports mean ± stdev for prefill_tps and decode_tps.
#
# Usage:
#     tools/bench_android.sh <model-list-row> [iters] [warmups] [cooldown_s]
#
# Where <model-list-row> is the 1-based row number in the MLCChat
# "Model List" view (e.g. 1 = first model, 2 = second). The script taps
# the chat-icon at the right end of that row.
#
# Defaults: iters=5, warmups=1, cooldown_s=60.
#
# Output: JSON on stdout, one final summary line of the form
#     {"model_id":"...","prefill_tps_mean":..., "prefill_tps_stdev":...,
#      "decode_tps_mean":...,"decode_tps_stdev":..., "n_iter":..., ...}
#
# Prerequisites:
#   - The patched MLCChat APK must be installed (AppViewModel.kt patch
#     landed; see ADR-013).
#   - Bundles for the target model staged in
#     /data/data/ai.mlc.mlcchat/files/.

set -euo pipefail

ROW="${1:?model-list-row required (1-indexed)}"
ITERS="${2:-5}"
WARMUPS="${3:-1}"
COOLDOWN="${4:-60}"

if (( ITERS < 3 )); then
    echo "[bench] iters must be >= 3 per H5" >&2
    exit 64
fi

# Screen size — used to derive tap coordinates that scale with device.
WH="$(adb shell wm size | awk -F': ' '/Physical/ {print $2}')"
W_NUM="${WH%x*}"
H_NUM="${WH#*x}"
W_NUM=${W_NUM:-1080}
H_NUM=${H_NUM:-2520}

# Approximate tap coords on a 1080×2520 phone, rescaled to actual device.
ICON_X=$(( W_NUM * 882 / 1080 ))
ICON_Y=$(( H_NUM * (460 + (ROW - 1) * 115) / 2520 ))

INPUT_X=$(( W_NUM * 480 / 1080 ))
INPUT_Y_KBDDOWN=$(( H_NUM * 2375 / 2520 ))
SEND_X=$(( W_NUM * 970 / 1080 ))
SEND_Y_KBDUP=$(( H_NUM * 1230 / 2520 ))
RESET_X=$(( W_NUM * 1010 / 1080 ))
RESET_Y=$(( H_NUM * 165 / 2520 ))

# A short, deterministic prompt with about 14 tokens that yields ~150–200
# completion tokens.
PROMPT="Tell%sme%sa%sshort%sstory%sabout%sa%scuriosity-driven%srobot%sexploring%san%sancient%scity."

launch_chat() {
    adb shell am start -W -n ai.mlc.mlcchat/.MainActivity > /dev/null 2>&1
    sleep 2
    adb shell input tap "${ICON_X}" "${ICON_Y}"
    sleep 8
}

run_one_chat() {
    adb logcat -c
    adb shell input tap "${INPUT_X}" "${INPUT_Y_KBDDOWN}"
    sleep 1
    adb shell input text "${PROMPT}"
    sleep 1
    adb shell input tap "${SEND_X}" "${SEND_Y_KBDUP}"
    local deadline=$(( $(date +%s) + 90 ))
    while (( $(date +%s) < deadline )); do
        local line
        line="$(adb logcat -d -s triad_bench:I 2>/dev/null \
                  | awk '/triad_bench:/ {sub(/.*triad_bench:[ ]*/,""); print}' \
                  | tail -1)"
        if [[ -n "${line}" ]]; then
            echo "${line}"
            return 0
        fi
        sleep 2
    done
    return 1
}

reset_chat() {
    # Clean re-launch is more reliable than tapping the in-app reset
    # button (which moves around when the keyboard is up). Between
    # iterations we force-stop the app and re-launch so each run starts
    # with an empty chat history and a fresh input box.
    adb shell am force-stop ai.mlc.mlcchat > /dev/null 2>&1
    sleep 2
    launch_chat
}

echo "[bench] device=$(adb get-serialno), display=${W_NUM}x${H_NUM}, model_row=${ROW}, iters=${ITERS}, warmups=${WARMUPS}, cooldown=${COOLDOWN}s" >&2

launch_chat

prefills=()
decodes=()
ptokens=()
ctokens=()
model_id=""

for (( k = 0; k < WARMUPS + ITERS; k++ )); do
    label="iter $((k+1-WARMUPS))/${ITERS}"
    if (( k < WARMUPS )); then label="warmup $((k+1))/${WARMUPS}"; fi
    echo "[bench] ${label} ..." >&2

    if line="$(run_one_chat)"; then
        echo "[bench] ${line}" >&2
        if (( k >= WARMUPS )); then
            read -r p d pt ct mid <<<"$(python3 -c '
import json, sys
d = json.loads(sys.argv[1])
def f(x): return x if x is not None else "nan"
def i(x): return x if x is not None else 0
print(f(d.get("prefill_tps")), f(d.get("decode_tps")),
      i(d.get("prompt_tokens")), i(d.get("completion_tokens")),
      d.get("model_id") or "")
' "${line}")"
            prefills+=("${p}")
            decodes+=("${d}")
            ptokens+=("${pt}")
            ctokens+=("${ct}")
            model_id="${mid}"
        fi
    else
        echo "[bench] ${label} FAILED — no triad_bench line within timeout" >&2
    fi

    if (( k + 1 < WARMUPS + ITERS )); then
        echo "[bench] cooldown ${COOLDOWN}s..." >&2
        sleep "${COOLDOWN}"
        reset_chat
    fi
done

python3 - "${model_id}" "${prefills[@]:-}" -- "${decodes[@]:-}" -- "${ptokens[@]:-}" -- "${ctokens[@]:-}" -- "${WARMUPS}" "${COOLDOWN}" <<'PY'
import json, sys, statistics as s
args = sys.argv[1:]
model_id = args.pop(0)
def take():
    out = []
    while args and args[0] != "--":
        out.append(args.pop(0))
    if args: args.pop(0)
    return out
pre = [float(x) for x in take() if x and x != "nan"]
dec = [float(x) for x in take() if x and x != "nan"]
pt  = [int(x)   for x in take() if x]
ct  = [int(x)   for x in take() if x]
warmups  = int(args[0])
cooldown = int(args[1])

def ms(v):
    if not v: return None, None
    if len(v) == 1: return v[0], 0.0
    return s.fmean(v), s.pstdev(v)

pm, pst = ms(pre)
dm, dst = ms(dec)

print(json.dumps({
    "model_id":          model_id,
    "n_iter":            len(pre),
    "warmups":           warmups,
    "prefill_tps_mean":  pm,
    "prefill_tps_stdev": pst,
    "decode_tps_mean":   dm,
    "decode_tps_stdev":  dst,
    "prompt_tokens_mean":     (sum(pt) / len(pt)) if pt else None,
    "completion_tokens_mean": (sum(ct) / len(ct)) if ct else None,
    "prefill_per_iter":  pre,
    "decode_per_iter":   dec,
    "cooldown_s":        cooldown,
}, indent=2))
PY
