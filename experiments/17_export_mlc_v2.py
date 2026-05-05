"""B.6: Export the v2 TRIAD checkpoint (n_calib=64 + clip_search) to a
runtime-loadable MLC q4f16_1 bundle, mirroring experiments/14_export_mlc.py
but pointed at /tmp/triad-tinyllama-int4-v2/.

Run:
    HF_HOME=$(pwd)/.cache/hf uv run python experiments/17_export_mlc_v2.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from triad_ptq.compile import _set_module  # noqa: E402
from triad_ptq.core.modules import TriadLinear  # noqa: E402
from triad_ptq.export.hf_safetensors import export_triad_to_hf_safetensors  # noqa: E402

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CKPT = Path("/tmp/triad-tinyllama-int4-v2/model.pt")
HF_OUT = Path("/tmp/triad-tinyllama-int4-v2-hf")
MLC_OUT = Path("/tmp/triad-tinyllama-int4-v2-mlc")
MLC_VENV = Path("/tmp/mlc-venv")


def _hf_snapshot_dir(hf_id: str) -> Path | None:
    repo = hf_id.replace("/", "--")
    base = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    cand = base / f"models--{repo}" / "snapshots"
    if not cand.exists():
        return None
    snaps = sorted(cand.iterdir())
    return snaps[-1] if snaps else None


def _attach_triad_modules(model, state_dict):
    triad_keys: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in state_dict.items():
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
        from triad_ptq.core.quantize import QuantizedWeight
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


def _run(cmd, **kw):
    print(">>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    if not CKPT.exists():
        sys.exit(f"v2 checkpoint not found: {CKPT}; run experiments/16_tinyllama_phase3_v2.py first")
    mlc_python = MLC_VENV / "bin" / "python"
    if not mlc_python.exists():
        sys.exit(f"mlc venv not found: {MLC_VENV}")

    print("loading HF skeleton + v2 TRIAD checkpoint...", flush=True)
    cfg = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sd = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    _attach_triad_modules(model, sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    snap = _hf_snapshot_dir(MODEL)
    if HF_OUT.exists():
        shutil.rmtree(HF_OUT)
    print(f"writing v2 TRIAD-folded HF safetensors to {HF_OUT}", flush=True)
    summary = export_triad_to_hf_safetensors(
        model, HF_OUT, hf_snapshot_dir=snap, dtype=torch.float16,
    )
    print(json.dumps(summary, indent=2), flush=True)
    del model, sd

    if MLC_OUT.exists():
        shutil.rmtree(MLC_OUT)
    MLC_OUT.mkdir(parents=True, exist_ok=True)
    _run([mlc_python, "-m", "mlc_llm", "gen_config", HF_OUT,
          "--quantization", "q4f16_1",
          "--conv-template", "llama-2",
          "--output", MLC_OUT])
    _run([mlc_python, "-m", "mlc_llm", "convert_weight", HF_OUT,
          "--quantization", "q4f16_1",
          "--output", MLC_OUT])
    (MLC_OUT / "lib").mkdir(parents=True, exist_ok=True)
    _run([mlc_python, "-m", "mlc_llm", "compile",
          MLC_OUT / "mlc-chat-config.json",
          "--device", "android",
          "--quantization", "q4f16_1",
          "--output", MLC_OUT / "lib" / "triad-tinyllama-v2-android.tar"])

    out = ROOT / "results" / "phase4_v2_export_summary.json"
    bundle_size_mb = sum(p.stat().st_size for p in MLC_OUT.rglob("*") if p.is_file()) / 1e6
    out.write_text(json.dumps({
        "hf_safetensors_dir": str(HF_OUT),
        "mlc_bundle_dir": str(MLC_OUT),
        "mlc_bundle_size_mb": bundle_size_mb,
        "tar_path": str(MLC_OUT / "lib" / "triad-tinyllama-v2-android.tar"),
        "tar_present": (MLC_OUT / "lib" / "triad-tinyllama-v2-android.tar").exists(),
        "hf_safetensors_size_mb": summary["model_safetensors_size_mb"],
    }, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
