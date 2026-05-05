"""Phase-4: load TRIAD-INT4 TinyLlama checkpoint, export MLC bundle.

Runs after experiments/13_tinyllama_phase3.py has produced
/tmp/triad-tinyllama-int4/model.pt. Output goes to
/tmp/triad-tinyllama-int4-mlc/.

The follow-up `mlc_llm compile` step (which builds the SPIR-V/Vulkan
shader library for Android) requires the mlc_llm and tvm-unity Python
packages plus an Android NDK toolchain. We do not run it here -- see
ADR-003 -- but we emit a small `compile.sh` next to the output so the
user can run the final step with one command on a properly provisioned
host.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from triad_ptq.compile import _set_module  # noqa: E402
from triad_ptq.core.modules import TriadLinear  # noqa: E402
from triad_ptq.export.mlc import export_to_mlc  # noqa: E402


MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CKPT = Path("/tmp/triad-tinyllama-int4/model.pt")
OUT = Path("/tmp/triad-tinyllama-int4-mlc")


def _hf_snapshot_dir(hf_id: str) -> Path | None:
    """Find the local HF cache snapshot directory for a model id."""
    repo = hf_id.replace("/", "--")
    base = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    cand = base / f"models--{repo}" / "snapshots"
    if not cand.exists():
        return None
    snaps = sorted(cand.iterdir())
    return snaps[-1] if snaps else None


def _attach_triad_modules_from_state_dict(model, state_dict):
    """The state_dict was saved from a model whose Linear modules had been
    replaced by TriadLinear. To load it back we need to re-instantiate the
    TriadLinear scaffolds with the right shapes, then load_state_dict.
    """
    # Walk the state_dict for keys that match the TriadLinear naming
    # convention. For each unique layer prefix, instantiate a TriadLinear
    # with the shapes inferred from `q`, `scales`, etc., place it in the
    # model graph, and let load_state_dict handle the buffers.

    # Group keys by linear-layer prefix
    triad_keys: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in state_dict.items():
        # heuristic: any key whose suffix is in {q, scales, zeros, U,
        # Lam_pow_beta, sw_rows, sw_cols, sw_vals, bias} and whose parent
        # exists as nn.Linear in the original (pre-TRIAD) model.
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
        out_f = old.out_features
        bits = 4
        group_size = 64
        if "scales" in parts:
            n_groups = parts["scales"].shape[1]
            group_size = max(1, in_f // max(1, n_groups))

        from triad_ptq.core.quantize import QuantizedWeight
        qw = QuantizedWeight(
            q=parts["q"].clone(),
            scales=parts["scales"].clone().to(torch.float32),
            zeros=parts["zeros"].clone().to(torch.int32),
            bits=bits, group_size=group_size,
        )
        U = parts.get("U")
        Lam = parts.get("Lam_pow_beta")
        sw_rows = parts.get("sw_rows")
        sw_cols = parts.get("sw_cols")
        sw_vals = parts.get("sw_vals")

        new_mod = TriadLinear.from_linear(
            old, qw,
            U=U, Lam_pow_beta=Lam,
            sw_rows=sw_rows, sw_cols=sw_cols, sw_vals=sw_vals,
            dtype=torch.float32,
        )
        _set_module(model, prefix, new_mod)


def main():
    print(f"loading HF skeleton: {MODEL}")
    cfg = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"loading TRIAD checkpoint: {CKPT}")
    sd = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    print(f"  keys={len(sd)}, total bytes ~{sum(v.numel()*v.element_size() for v in sd.values() if isinstance(v, torch.Tensor))/1e6:.1f} MB")

    print("re-attaching TriadLinear modules into HF model graph...")
    _attach_triad_modules_from_state_dict(model, sd)
    # Now load_state_dict to populate any remaining tensors (norms, embeds,
    # bias). Use strict=False to ignore the keys we manually injected via
    # buffer registration in TriadLinear.from_linear.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected[:5]:
        print(f"  unexpected (first 5): {unexpected[:5]}")
    if missing[:5]:
        print(f"  missing (first 5): {missing[:5]}")

    snap = _hf_snapshot_dir(MODEL)
    print(f"hf snapshot dir: {snap}")

    print(f"exporting to MLC bundle: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = export_to_mlc(
        model, OUT,
        hf_model_id=MODEL,
        hf_snapshot_dir=snap,
        fold_U=True,
        fold_super_weights=True,
    )
    print(json.dumps(summary, indent=2))

    # Drop a one-shot compile.sh for the user to run once mlc_llm + NDK
    # are installed (per ADR-003: not in this autonomous session).
    compile_sh = OUT / "compile.sh"
    compile_sh.write_text(f"""#!/usr/bin/env bash
# Final-mile MLC compile for Android Vulkan. Requires mlc_llm + tvm-unity.
# Usage: bash compile.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
python -m mlc_llm compile \\
  "$HERE/mlc-chat-config.json" \\
  --device android \\
  --quantization q4f16_1 \\
  --output "$HERE/lib/triad-tinyllama-vulkan.so"
""")
    compile_sh.chmod(0o755)

    out_summary = ROOT / "results" / "phase4_export_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_summary}")


if __name__ == "__main__":
    main()
