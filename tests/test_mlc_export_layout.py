"""Phase 5.5 layout & compile-artefact tests.

Run only when the device-side bundle exists at /tmp/triad-tinyllama-int4-mlc/.
These are smoke checks that the canonical MLC q4f16_1 layout is intact
and that the compile step produced an Android-target tar.
"""
from __future__ import annotations

import json
import math
import os
import tarfile
from pathlib import Path

import pytest

BUNDLE = Path("/tmp/triad-tinyllama-int4-mlc")

requires_artifacts = pytest.mark.skipif(
    not BUNDLE.exists(),
    reason=f"MLC bundle not present at {BUNDLE}; run experiments/14_export_mlc.py first",
)


@requires_artifacts
def test_mlc_bundle_has_required_files():
    must_have = [
        "mlc-chat-config.json",
        "tensor-cache.json",
        "tokenizer.json",
        "tokenizer.model",
    ]
    for fname in must_have:
        assert (BUNDLE / fname).exists(), f"missing {fname} from MLC bundle"
    shards = list(BUNDLE.glob("params_shard_*.bin"))
    assert len(shards) >= 1, "no params_shard_*.bin files found"


@requires_artifacts
def test_mlc_bundle_size_within_expected_range():
    """Total params byte size must be within +/-20% of the theoretical
    INT4-with-fp16-scales/zeros size for TinyLlama-1.1B at group_size=32.
    """
    cache = json.loads((BUNDLE / "tensor-cache.json").read_text())
    total_bytes = sum(rec["nbytes"] for rec in cache["records"])
    total_mb = total_bytes / (1024 * 1024)

    # TinyLlama-1.1B = 1.1e9 params -> 4 bits per weight + group overhead.
    # Canonical q4f16_1 reports ~4.5 bits/param across the model card; the
    # tensor-cache embeds 24 records covering quantized weights + fp16 norms.
    expected_mb = 1.1e9 * 4.5 / 8 / (1024 * 1024)  # ~590 MB
    lo, hi = expected_mb * 0.80, expected_mb * 1.20
    assert lo < total_mb < hi, (
        f"bundle size {total_mb:.1f} MB outside [{lo:.1f}, {hi:.1f}] MB range"
    )


@requires_artifacts
def test_mlc_compile_artifact_present():
    tar_path = BUNDLE / "lib" / "triad-tinyllama-android.tar"
    assert tar_path.exists(), "expected mlc compile output .tar missing"

    # Tar may be gzip-compressed (mlc_llm uses tarfile with default options)
    with tarfile.open(tar_path, "r:*") as tar:
        names = tar.getnames()
    # Compile output must contain at least one .o or .so for Vulkan/OpenCL kernels
    assert any(n.endswith((".o", ".so")) for n in names), (
        f"compile output tar has no .o/.so members: {names!r}"
    )


@requires_artifacts
def test_mlc_chat_config_records_quantization():
    cfg = json.loads((BUNDLE / "mlc-chat-config.json").read_text())
    assert cfg.get("quantization") == "q4f16_1", (
        f"expected q4f16_1, got {cfg.get('quantization')!r}"
    )
    # context_window must be reasonable
    cw = cfg.get("context_window_size") or cfg.get("model_config", {}).get(
        "context_window_size"
    )
    assert cw and cw >= 512, f"context_window too small: {cw}"
