"""Phase-4 R1 driver: rotate TinyLlama-1.1B in place + verify forward
equivalence on a held-out prompt, then save the rotated FP16
state_dict for the next calibration pass.

Run:  uv run python experiments/19_r1_rotate_tinyllama.py

Output:
    /tmp/triad-tinyllama-r1/model_rotated_fp16.pt   (state_dict)
    results/phase4_r1_rotation_summary.json

This experiment does NOT do calibration or export — those are the
next-stage runs. Its job is:
  1. Apply R1 (random signed Hadamard) to TinyLlama's residual stream,
     fold the RMSNorms, rotate the embedding + lm_head.
  2. Verify cosine similarity between rotated and unrotated FP32
     forward outputs on N=8 sample inputs (Phase-4 acceptance gate
     ≥ 0.9999).
  3. Persist the rotated weights so the next calibration session can
     skip step 1.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from triad_ptq.core.rotate import apply_r1_to_llama  # noqa: E402

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
N_PROBE = 8
SEQ = 256
SEED = 0xACE1
OUT_DIR = Path("/tmp/triad-tinyllama-r1")
SUMMARY = ROOT / "results" / "phase4_r1_rotation_summary.json"

console = Console()


@torch.no_grad()
def _probe_outputs(model, tok, n_samples=N_PROBE, seq_len=SEQ, device="cpu"):
    """Run forward on n_samples random-text prompts and return concatenated
    fp32 logits (T_total, vocab) for cosine comparison."""
    text_pool = [
        "The quick brown fox jumps over the lazy dog. ",
        "In the beginning was the Word, and the Word was with God. ",
        "All happy families are alike; each unhappy family is unhappy in its own way. ",
        "It was the best of times, it was the worst of times. ",
        "I think therefore I am. The unexamined life is not worth living. ",
        "Mathematics is the language with which God has written the universe. ",
        "Hardware is irrelevant; the software is what matters in the end. ",
        "Once upon a midnight dreary, while I pondered weak and weary. ",
    ]
    outs = []
    for s in text_pool[:n_samples]:
        ids = tok(s * 32, return_tensors="pt", truncation=True, max_length=seq_len).input_ids
        ids = ids.to(device)
        logits = model(input_ids=ids).logits.detach().float()
        outs.append(logits)
    return torch.cat([o.reshape(-1, o.size(-1)) for o in outs], dim=0)


def main() -> None:
    dev = torch.device("cpu")  # FP32 forward on CPU keeps the bit-exact comparison clean
    console.log(f"loading {MODEL} on {dev}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    t0 = time.perf_counter()
    out_pre = _probe_outputs(model, tok, device=dev)
    t_pre = time.perf_counter() - t0
    console.log(f"  probed pre-rotation logits {tuple(out_pre.shape)} in {t_pre:.1f}s")

    info = apply_r1_to_llama(model, seed=SEED)
    console.log(f"  applied R1: {info}")

    t0 = time.perf_counter()
    out_post = _probe_outputs(model, tok, device=dev)
    t_post = time.perf_counter() - t0

    # The lm_head was rotated, so logits are scrambled along the vocab axis…
    # actually no: lm_head's INPUT axis was rotated to match the residual
    # stream's rotation. The vocab dim (output axis) is unchanged. So
    # logits should be element-wise close, not just cosine-close. We
    # report both.
    diff = (out_pre - out_post).norm() / out_pre.norm().clamp_min(1e-12)
    cos = torch.nn.functional.cosine_similarity(out_pre, out_post, dim=-1).mean().item()
    console.log(f"  rel L2 = {diff.item():.3e},  mean cos = {cos:.6f}")
    console.log(f"  acceptance gate (cos ≥ 0.9999): {'PASS' if cos > 0.9999 else 'FAIL'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pt = OUT_DIR / "model_rotated_fp16.pt"
    sd_fp16 = {k: v.detach().to(torch.float16).cpu() for k, v in model.state_dict().items()}
    torch.save(sd_fp16, out_pt)
    pt_size_mb = out_pt.stat().st_size / 1e6
    console.log(f"  wrote rotated state_dict to {out_pt}  ({pt_size_mb:.1f} MB)")

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps({
        "model": MODEL,
        "seed": SEED,
        "n_probe": N_PROBE,
        "seq_len": SEQ,
        "rel_L2": float(diff.item()),
        "cos_mean": cos,
        "acceptance_gate": "cos >= 0.9999",
        "passes": cos > 0.9999,
        "rotated_state_dict": str(out_pt),
        "rotated_state_dict_mb": pt_size_mb,
        "pre_forward_sec": t_pre,
        "post_forward_sec": t_post,
        "rotation_info": info,
    }, indent=2))
    console.log(f"wrote {SUMMARY}")


if __name__ == "__main__":
    main()
