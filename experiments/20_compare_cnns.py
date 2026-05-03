"""Tier-1 CNN sweep on ImageNetV2 (matched-frequency, 10K images).

For each model x method we record top-1 / top-5 accuracy on a sampled
subset (default 5000 images, configurable). Results are written
incrementally to results/tables/cnn_sweep.json.

Sample predictions for 10 images per method are saved to
results/samples/cnn_<model>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from rich.console import Console
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))

from triad_ptq import optimize  # noqa: E402
from triad_ptq.baselines.awq import awq_like_quantize  # noqa: E402
from triad_ptq.baselines.rtn import quantize_rtn  # noqa: E402
from triad_ptq.eval.vision import (  # noqa: E402
    HFImageNetV2,
    build_imagenet_calibration,
    imagenet_default_transform,
    topk_accuracy,
)

console = Console()


def load_model(tag: str) -> nn.Module:
    if tag == "mobilenet_v2":
        import torchvision
        return torchvision.models.mobilenet_v2(weights="IMAGENET1K_V2").eval()
    if tag == "efficientnet_b0":
        import torchvision
        return torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1").eval()
    if tag == "mobilevit_s":
        import timm
        return timm.create_model("mobilevit_s", pretrained=True).eval()
    raise ValueError(tag)


def make_transform(tag: str):
    """Per-model preprocessing. Some timm models need their own transform."""
    from triad_ptq.eval.vision import imagenet_default_transform
    if tag == "mobilevit_s":
        import timm
        m = timm.create_model("mobilevit_s", pretrained=True)
        cfg = timm.data.resolve_model_data_config(m)
        return timm.data.create_transform(**cfg, is_training=False)
    return imagenet_default_transform()


def _vision_forward(model, batch, device):
    if isinstance(batch, dict):
        x = batch["pixel_values"]
    elif isinstance(batch, (list, tuple)):
        x = batch[0]
    else:
        x = batch
    x = x.to(device, dtype=torch.float32)
    return model(x)


def _vision_output(out):
    if hasattr(out, "logits"):
        return out.logits
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["mobilenet_v2", "efficientnet_b0", "mobilevit_s"])
    ap.add_argument("--methods", nargs="*", default=["FP32", "RTN", "AWQ-like", "TRIAD"])
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--n-eval", type=int, default=5000)
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    console.log(f"device={dev}")

    out_path = ROOT / "results" / "tables" / "cnn_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples_dir = ROOT / "results" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        results = json.loads(out_path.read_text())
    else:
        results = {"runs": []}
    done_keys = {(r["model"], r["method"], r["bits"]) for r in results["runs"]}

    for tag in args.models:
        console.log(f"\n=== {tag} ===")
        # Per-model transform & dataset (necessary for timm models with
        # non-standard preprocessing like MobileViT-S).
        transform = make_transform(tag)
        eval_ds = HFImageNetV2(transform=transform, max_n=args.n_eval)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch, num_workers=0)
        calib_batches = build_imagenet_calibration(transform=transform, n=args.n_calib, batch_size=16)
        sample_xs = [eval_ds[i] for i in range(10)]
        sample_x_stack = torch.stack([x for x, _ in sample_xs]).to(dev)
        sample_y = [y for _, y in sample_xs]
        per_model_samples: dict = {"model": tag, "labels": sample_y, "predictions": {}}
        for method in args.methods:
            bits = 32 if method == "FP32" else args.bits
            key = (tag, method, bits)
            if key in done_keys:
                console.log(f"  [skip] {method} bits={bits} already done")
                continue
            console.log(f"  --- {method} bits={bits} ---")
            t0 = time.perf_counter()
            try:
                m = load_model(tag).to(dev)
                if method == "FP32":
                    pass
                elif method == "RTN":
                    quantize_rtn(m, bits=bits, group_size=32, device=dev)
                elif method == "AWQ-like":
                    awq_like_quantize(
                        m, calib_batches, bits=bits, group_size=32,
                        n_calib=args.n_calib, n_grid=10, device=dev,
                        forward_fn=_vision_forward,
                    )
                elif method == "TRIAD":
                    optimize(
                        m, bits=bits, calibration=calib_batches,
                        super_weight_frac=5e-3,
                        bit_allocator="trace", cov_grid="analytic",
                        n_calib=args.n_calib,
                        rho_probe_n=2, group_size=32, progress=False,
                        forward_fn=_vision_forward, output_fn=_vision_output,
                    )
                else:
                    raise ValueError(method)
                calib_sec = time.perf_counter() - t0

                acc = topk_accuracy(m, eval_loader, device=dev, progress=False)
                # Sample predictions
                with torch.no_grad():
                    sl = m(sample_x_stack)
                    if hasattr(sl, "logits"):
                        sl = sl.logits
                    top3 = sl.topk(3, dim=-1).indices.cpu().tolist()
                per_model_samples["predictions"][method] = top3

                row = {
                    "model": tag, "method": method, "bits": bits,
                    "top1": acc["top1"], "top5": acc["top5"], "n_eval": acc["n"],
                    "calib_sec": calib_sec, "eval_sec": acc["sec"],
                }
                console.log(
                    f"    top1={acc['top1']*100:.2f}%  top5={acc['top5']*100:.2f}%  "
                    f"calib={calib_sec:.0f}s  eval={acc['sec']:.0f}s"
                )
                results["runs"].append(row)
                out_path.write_text(json.dumps(results, indent=2))
                del m
            except Exception:
                tb = traceback.format_exc()
                console.log(f"    [FAIL] {tb}")
                results["runs"].append({
                    "model": tag, "method": method, "bits": bits,
                    "error": tb[-1000:],
                })
                out_path.write_text(json.dumps(results, indent=2))
            if dev.type == "mps":
                torch.mps.empty_cache()

        (samples_dir / f"cnn_{tag}.json").write_text(json.dumps(per_model_samples, indent=2))

    console.log(f"\nWrote {out_path} ({len(results['runs'])} runs)")


if __name__ == "__main__":
    main()
