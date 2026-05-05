"""Test-fixture helpers.

These helpers exist so unit tests can load a TRIAD-quantized model
without re-running calibration in every test. The first call to a
loader builds the fixture and caches it under
``/tmp/triad-test-fixtures/<name>/``; subsequent calls return the
cached artefact instantly.

Build invocation (opt-in):
    TRIAD_BUILD_FIXTURES=1 uv run pytest tests/test_generation_smoke.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

FIXTURE_ROOT = Path("/tmp/triad-test-fixtures")
SMOLLM_FIXTURE_DIR = FIXTURE_ROOT / "smollm135-int4"


def smollm135_fixture_available() -> bool:
    """True if a previously-built SmolLM-135M INT4 fixture is on disk."""
    return (SMOLLM_FIXTURE_DIR / "model.pt").exists()


def _build_smollm135_int4(force: bool = False):
    """Calibrate SmolLM-135M to INT4 with TRIAD and cache the result.

    Reuses the same pipeline ``experiments/01_calibrate_smollm.py``
    exercises; uses a small 4-batch calibration so the cold build
    completes in under a minute on M1.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433

    from triad_ptq import optimize
    from triad_ptq.eval.calib import build_wikitext_calib

    SMOLLM_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = SMOLLM_FIXTURE_DIR / "model.pt"
    tokenizer_dir = SMOLLM_FIXTURE_DIR / "tokenizer"
    if model_path.exists() and not force:
        return

    name = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    calib = build_wikitext_calib(tokenizer, n_samples=4, seq_len=512, device=dev)
    optimize(
        model,
        bits=4,
        calibration=calib,
        super_weight_frac=5e-4,
        bit_allocator="trace",
        cov_grid="analytic",
        n_calib=4,
        rho_probe_n=1,
        group_size=64,
        progress=False,
        device=str(dev),
        a_device="cpu",
    )

    cpu_model = model.to("cpu")
    torch.save(cpu_model.state_dict(), str(model_path))
    tokenizer.save_pretrained(str(tokenizer_dir))


def load_smollm135_quantized_int4() -> Tuple["torch.nn.Module", "object"]:
    """Return (model, tokenizer) for the cached SmolLM-135M TRIAD-INT4 fixture.

    Builds the fixture if absent AND ``TRIAD_BUILD_FIXTURES=1`` is set
    in the environment. Otherwise raises ``FileNotFoundError`` with a
    pointer to the build invocation.
    """
    import torch  # noqa: WPS433
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433

    if not smollm135_fixture_available():
        if os.environ.get("TRIAD_BUILD_FIXTURES") != "1":
            raise FileNotFoundError(
                f"SmolLM-135M INT4 fixture not built at {SMOLLM_FIXTURE_DIR}. "
                "Run with TRIAD_BUILD_FIXTURES=1 to build it (~60 s on M1)."
            )
        _build_smollm135_int4()

    from triad_ptq.compile import _set_module
    from triad_ptq.core.modules import TriadLinear
    from triad_ptq.core.quantize import QuantizedWeight

    name = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(SMOLLM_FIXTURE_DIR / "tokenizer")
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    sd = torch.load(SMOLLM_FIXTURE_DIR / "model.pt", map_location="cpu", weights_only=False)

    # Re-attach TriadLinear modules from the saved state_dict.
    triad_keys: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        for suf in ("q", "scales", "zeros", "U", "Lam_pow_beta",
                    "sw_rows", "sw_cols", "sw_vals"):
            if k.endswith("." + suf):
                prefix = k[: -(len(suf) + 1)]
                triad_keys.setdefault(prefix, {})[suf] = v
                break

    import torch.nn as nn
    name_to_mod = dict(model.named_modules())
    for prefix, parts in triad_keys.items():
        old = name_to_mod.get(prefix)
        if not isinstance(old, nn.Linear):
            continue
        in_f = old.in_features
        n_groups = parts["scales"].shape[1]
        group_size = max(1, in_f // max(1, n_groups))
        qw = QuantizedWeight(
            q=parts["q"].clone(),
            scales=parts["scales"].clone().to(torch.float32),
            zeros=parts["zeros"].clone().to(torch.int32),
            bits=4, group_size=group_size,
        )
        new_mod = TriadLinear.from_linear(
            old, qw,
            U=parts.get("U"), Lam_pow_beta=parts.get("Lam_pow_beta"),
            sw_rows=parts.get("sw_rows"), sw_cols=parts.get("sw_cols"),
            sw_vals=parts.get("sw_vals"),
            dtype=torch.float32,
        )
        _set_module(model, prefix, new_mod)

    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, tokenizer


__all__ = [
    "FIXTURE_ROOT",
    "SMOLLM_FIXTURE_DIR",
    "smollm135_fixture_available",
    "load_smollm135_quantized_int4",
]
