#!/usr/bin/env bash
# A.3: replicated-run bench through MLCChat to measure variance and
# test H4 (the Phase-5 single-prompt single-run protocol was statistically
# thin, and the 12% / 28% gaps may be measurement variance + thermal).
#
# Drives the chat via adb input + uiautomator dump. Captures the
# in-app "prefill: X tok/s, decode: Y tok/s" indicator after each
# generation completes.
#
# Usage:
#   bash experiments/profile/A3_replicated_bench.sh <runs-per-model>
#
# Output: experiments/profile/A3_replicated_results.json
set -euo pipefail

RUNS=${1:-3}
OUT=/Users/artemkatolikov/DEV/Triad-ML/experiments/profile/A3_replicated_results.json
PROMPT="Write a short poem about the ocean and the moonlight in simple words that a child could read and enjoy as a bedtime story now please."

# Tap coordinates from the Z Flip7 main display (1080x2520)
RESET_XY="996 212"
BACK_XY="84 212"
INPUT_XY="396 2364"
SEND_XY="1011 2366"

# We assume the app is open on a chat screen (any model). The script
# will toggle between the currently loaded model and the OTHER one by
# tapping the back arrow + the appropriate model card on the home page.

dump_text() {
    adb shell uiautomator dump >/dev/null 2>&1
    adb pull /sdcard/window_dump.xml /tmp/win_a3.xml >/dev/null 2>&1
    cat /tmp/win_a3.xml
}

current_model() {
    dump_text | grep -oE 'text="MLCChat: [^"]*"' | head -1 | sed 's/.*MLCChat: //;s/"$//'
}

current_metrics() {
    dump_text | grep -oE 'text="prefill: [^"]*"' | head -1 | sed 's/text="//;s/"$//'
}

is_generating() {
    # Generation is in progress if the input EditText hint says NOT "Input"
    # (i.e. the send button is in stop state). A simpler heuristic: the
    # "Input" hint reappears only when ready for the next prompt.
    dump_text | grep -q 'text="Input"' && return 1
    return 0
}

wait_until_done() {
    # Poll metrics every 2 s. Generation is complete when the metrics
    # line text is non-empty AND the Input hint is back (means input
    # cleared, send button re-enabled).
    local last="" cur=""
    for _ in $(seq 1 60); do
        sleep 2
        cur="$(current_metrics)"
        if [ -n "$cur" ] && dump_text | grep -q 'text="Input"'; then
            # Stable check: same metric two polls in a row
            if [ "$cur" = "$last" ]; then
                echo "$cur"
                return 0
            fi
            last="$cur"
        fi
    done
    echo "$cur"
    return 1
}

send_prompt() {
    adb shell input tap $RESET_XY
    sleep 1.5
    adb shell input tap $INPUT_XY
    sleep 0.5
    # Encode spaces in the prompt for adb input
    local p="${PROMPT// /%s}"
    adb shell input text "\"$p\""
    sleep 0.5
    adb shell input tap $SEND_XY
}

go_to_home() {
    adb shell input tap $BACK_XY
    sleep 1.5
}

go_to_other_model() {
    # On the home page, tap the chat icon for the OTHER model card.
    # Home-page layout: each model row has a chat icon at the right.
    # We dump and find it dynamically.
    go_to_home
    adb shell uiautomator dump >/dev/null 2>&1
    adb pull /sdcard/window_dump.xml /tmp/win_a3_home.xml >/dev/null 2>&1
    # Find chat icons (content-desc="Chat") with their bounds.
    python3 - <<PY > /tmp/a3_chat_icons.txt
import xml.etree.ElementTree as ET
tree = ET.parse('/tmp/win_a3_home.xml')
icons = []
texts = []
for n in tree.iter():
    a = n.attrib
    cd = a.get('content-desc', '')
    txt = a.get('text', '')
    if 'Chat' in cd or txt.startswith('triad') or txt.startswith('tinyllama') or 'tinyllama' in txt.lower():
        icons.append((cd, txt, a.get('bounds')))
for ic in icons:
    print('\t'.join(str(x) for x in ic))
PY
    cat /tmp/a3_chat_icons.txt
}

run_once() {
    local model="$1"
    local label="$2"
    local n="$3"
    send_prompt
    local metrics
    metrics="$(wait_until_done || true)"
    echo "$model,$label,$n,$metrics"
}

main() {
    echo "[A.3] starting replicated bench, runs=$RUNS"
    local results=()
    local current
    current="$(current_model)"
    echo "currently loaded: $current"

    for n in $(seq 1 $RUNS); do
        echo ">>> run $n on '$current' (alternating)"
        sleep 1
        local m
        m="$(run_once "$current" "first" $n)"
        echo "  $m"
        results+=("$m")
    done

    # Try to switch to the other model via home page
    echo "switching model..."
    go_to_other_model
    # User instruction: Tap the chat icon of the model that is NOT $current.
    # Without machine-vision we'll guess by sortable order.
    local other_xy
    if [ "$current" = "tinyllama" ] || echo "$current" | grep -qi "ref"; then
        # Tap the OTHER (TRIAD) entry. The chat icon is typically at the
        # right edge of the second row -- adjust empirically below.
        other_xy="990 540"
    else
        other_xy="990 380"
    fi
    adb shell input tap $other_xy
    sleep 8

    local current2
    current2="$(current_model)"
    echo "after switch: $current2"

    if [ "$current2" != "$current" ] && [ -n "$current2" ]; then
        for n in $(seq 1 $RUNS); do
            echo ">>> run $n on '$current2' (second model)"
            sleep 1
            local m
            m="$(run_once "$current2" "second" $n)"
            echo "  $m"
            results+=("$m")
        done
    else
        echo "WARN: model switch failed; remained on $current2"
    fi

    {
        echo "{"
        echo "  \"prompt\": \"$PROMPT\","
        echo "  \"runs_per_model\": $RUNS,"
        echo "  \"results\": ["
        local first=1
        for r in "${results[@]}"; do
            [ $first -eq 1 ] && first=0 || echo ","
            echo -n "    \"$r\""
        done
        echo
        echo "  ]"
        echo "}"
    } > "$OUT"
    echo "wrote $OUT"
    cat "$OUT"
}

main "$@"
