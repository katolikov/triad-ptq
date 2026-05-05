"""Phase-4 (revised per ADR-004): TRIAD checkpoint -> HF safetensors -> MLC bundle.

Three steps:

  1. Load TRIAD-INT4 state_dict from /tmp/triad-tinyllama-int4/model.pt.
     Re-attach TriadLinear modules into a fresh HF skeleton.
  2. Materialise the deployed-side dense weight (TRIAD-folded fp16) as
     `model.safetensors` in /tmp/triad-tinyllama-int4-hf/.
  3. Run mlc_llm gen_config + convert_weight on (2), then mlc_llm
     compile, producing /tmp/triad-tinyllama-int4-mlc/ with the
     canonical MLC q4f16_1 layout (24 shards, tensor-cache.json,
     fused QKV / gate-up records, runtime-loadable).

The mlc_llm tool is invoked as a subprocess and does not need to be
in this repo's uv venv -- by default we look for it in /tmp/mlc-venv
which the calling shell created. Override with --mlc-venv.

Run:
    HF_HOME=$(pwd)/.cache/hf uv run python experiments/14_export_mlc.py
"""
from __future__ import annotations

import argparse
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
CKPT = Path("/tmp/triad-tinyllama-int4/model.pt")
HF_OUT = Path("/tmp/triad-tinyllama-int4-hf")
MLC_OUT = Path("/tmp/triad-tinyllama-int4-mlc")


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


def _run(cmd: list[str], cwd: Path | None = None, env_extra: dict | None = None) -> None:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlc-venv", default="/tmp/mlc-venv",
                    help="Path to the venv containing mlc_llm (Phase 1.1.1).")
    ap.add_argument("--device", default="android",
                    help="MLC compile target device (android | vulkan | metal | ...).")
    ap.add_argument("--skip-compile", action="store_true",
                    help="Stop after convert_weight; do not produce the .tar.")
    ap.add_argument(
        "--quantization", default="q4f16_1",
        choices=["q4f16_0", "q4f16_1", "both"],
        help=(
            "MLC weight-packing layout. q4f16_1 is the canonical pack used by "
            "ai/mlc.ai community models. q4f16_0 is the SoA (struct-of-arrays) "
            "layout that the callstack.com Adreno write-up reports as faster on "
            "mobile RDNA-class GPUs (LDS coalescing). Both produce identical "
            "PPL within fp16 noise — pick on tok/s. 'both' emits two bundles "
            "with -q4f16_0 / -q4f16_1 dir suffixes for side-by-side device bench."
        ),
    )
    args = ap.parse_args()

    mlc_python = Path(args.mlc_venv) / "bin" / "python"
    if not mlc_python.exists():
        sys.exit(f"mlc_llm venv not found at {args.mlc_venv}; "
                 f"create it per Phase 1.1.1 first.")

    # ---- 1. Re-attach TriadLinear and load weights ------------------
    print(f"loading HF skeleton: {MODEL}", flush=True)
    cfg = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"loading TRIAD checkpoint: {CKPT}", flush=True)
    sd = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    print(f"  keys={len(sd)}", flush=True)

    print("re-attaching TriadLinear modules into HF model graph...", flush=True)
    _attach_triad_modules(model, sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    # ---- 2. Materialise TRIAD-folded HF safetensors -----------------
    snap = _hf_snapshot_dir(MODEL)
    print(f"hf snapshot dir: {snap}", flush=True)
    if HF_OUT.exists():
        shutil.rmtree(HF_OUT)
    print(f"writing TRIAD-folded HF safetensors to {HF_OUT}", flush=True)
    summary = export_triad_to_hf_safetensors(
        model, HF_OUT, hf_snapshot_dir=snap, dtype=torch.float16,
    )
    print(json.dumps(summary, indent=2), flush=True)

    # Free the model -- next steps use mlc_llm subprocesses.
    del model, sd

    # ---- 3. mlc_llm gen_config + convert_weight + compile ----------
    quants = ["q4f16_0", "q4f16_1"] if args.quantization == "both" else [args.quantization]
    bundles: list[dict] = []
    for q in quants:
        # Suffix the bundle dir when emitting more than one.
        bundle_dir = (
            MLC_OUT.with_name(MLC_OUT.name + f"-{q}") if len(quants) > 1 else MLC_OUT
        )
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        _run([
            str(mlc_python), "-m", "mlc_llm", "gen_config",
            str(HF_OUT),
            "--quantization", q,
            "--conv-template", "llama-2",
            "--output", str(bundle_dir),
        ])
        _run([
            str(mlc_python), "-m", "mlc_llm", "convert_weight",
            str(HF_OUT),
            "--quantization", q,
            "--output", str(bundle_dir),
        ])

        tar_path = bundle_dir / "lib" / f"triad-tinyllama-android-{q}.tar"
        if not args.skip_compile:
            tar_path.parent.mkdir(parents=True, exist_ok=True)
            _run([
                str(mlc_python), "-m", "mlc_llm", "compile",
                str(bundle_dir / "mlc-chat-config.json"),
                "--device", args.device,
                "--quantization", q,
                "--output", str(tar_path),
            ])

        bundle_size_mb = sum(
            p.stat().st_size for p in bundle_dir.rglob("*") if p.is_file()
        ) / 1e6
        bundles.append({
            "quantization": q,
            "mlc_bundle_dir": str(bundle_dir),
            "mlc_bundle_size_mb": bundle_size_mb,
            "tar_path": str(tar_path),
            "tar_present": tar_path.exists(),
        })

    out_summary = ROOT / "results" / "phase4_export_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps({
        "hf_safetensors_dir": str(HF_OUT),
        "compile_device": args.device,
        "hf_safetensors_size_mb": summary["model_safetensors_size_mb"],
        "bundles": bundles,
    }, indent=2))
    print(f"wrote {out_summary}", flush=True)


if __name__ == "__main__":
    main()
