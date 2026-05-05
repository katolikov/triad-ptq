#!/usr/bin/env bash
# tools/verify_kernel_identity.sh — verify that v2's exported MLC bundle
# yields a BIT-IDENTICAL OpenCL device-code object to the q4f16_1
# community baseline. This is the structural guarantee that "v2 ships
# with zero kernel changes" — we can verify it without running on a
# device.
#
# Inputs (positional):
#   $1 — path to TRIAD-v2 bundle directory (must contain `lib/`)
#   $2 — path to community-baseline bundle directory (must contain `lib/`)
#
# Behaviour:
#   - Extracts the MLC TIR-emitted device-code object
#     (`*_devc.o`) from each bundle.
#   - md5 sums both.
#   - Returns 0 on match, 1 on mismatch.
#
# This script does NOT itself run `mlc_llm compile`; it operates on the
# already-compiled bundle pairs that the calibration runbook produces.
# The PHASE-C unit test for kernel identity is in
# `tests/v2/test_sign_perm_rotation.py::test_md5_invariance_smoke` which
# verifies this script's logic against a synthetic byte-identical fixture.

set -euo pipefail

V2="${1:?usage: verify_kernel_identity.sh <v2_bundle> <baseline_bundle>}"
REF="${2:?usage: verify_kernel_identity.sh <v2_bundle> <baseline_bundle>}"

find_devc() {
    local dir="$1"
    # MLC emits foo_q4f16_1_devc.o under the bundle's lib/ subdir.
    find "$dir" -name '*_devc.o' -type f 2>/dev/null | head -1
}

V2_DEVC="$(find_devc "$V2")"
REF_DEVC="$(find_devc "$REF")"

if [[ -z "$V2_DEVC" || -z "$REF_DEVC" ]]; then
    echo "[verify_kernel_identity] missing *_devc.o in v2=$V2 or ref=$REF" >&2
    exit 2
fi

V2_HASH="$(md5sum "$V2_DEVC" | awk '{print $1}')"
REF_HASH="$(md5sum "$REF_DEVC" | awk '{print $1}')"

cat <<JSON
{
  "v2_devc":   "$V2_DEVC",
  "ref_devc":  "$REF_DEVC",
  "v2_md5":    "$V2_HASH",
  "ref_md5":   "$REF_HASH",
  "match":     $([[ "$V2_HASH" == "$REF_HASH" ]] && echo true || echo false)
}
JSON

if [[ "$V2_HASH" != "$REF_HASH" ]]; then
    echo "[verify_kernel_identity] MISMATCH — kernel identity broken" >&2
    exit 1
fi

echo "[verify_kernel_identity] match" >&2
