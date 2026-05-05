"""Phase-2 smoke test: GPTAQ asymmetric calibration on SmolLM-135M.

Compares two calibrations on the same model and the same eval window:

* TRIAD-INT4 baseline                                        (asymmetric_calib=False)
* TRIAD-INT4 + GPTAQ asymmetric weight transfer              (asymmetric_calib=True)

PPL is evaluated on a fixed WikiText-2 test slice. The asymmetric path
is expected to produce a small but consistent PPL improvement; this
smoke test does NOT prove the TinyLlama gating gain (Phase-2 acceptance
needs ≥0.08 PPL drop on TinyLlama-1.1B), but it does verify the
implementation is correct and the streaming pipeline still converges.

Run:  uv run python experiments/18_gptaq_smoke_smollm.py
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

from triad_ptq import optimize  # noqa: E402
from triad_ptq.eval.calib import build_wikitext_calib  # noqa: E402
from triad_ptq.eval.ppl import load_wikitext2, perplexity  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM-135M"
SEQ = 1024
N_CALIB = 16
EVAL_TOKENS = 16_384

console = Console()


def main() -> None:
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    console.log(f"device={dev}, model={MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    text_eval = load_wikitext2("test")

    def fresh_model() -> torch.nn.Module:
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
        m.to(dev).eval()
        return m

    def calib() -> list:
        return build_wikitext_calib(tok, n_samples=N_CALIB, seq_len=SEQ, device=dev)

    results = []

    for asym, label in [(False, "TRIAD-INT4 (baseline)"),
                        (True,  "TRIAD-INT4 + GPTAQ asym")]:
        console.log(f"[{label}]  asymmetric_calib={asym}")
        m = fresh_model()
        c = calib()
        t0 = time.perf_counter()
        m, meta = optimize(
            m, bits=4, calibration=c,
            super_weight_frac=5e-4,
            bit_allocator="trace",
            cov_grid="analytic",
            n_calib=N_CALIB,
            rho_probe_n=2,
            group_size=64,
            progress=True,
            asymmetric_calib=asym,
            return_meta=True,
        )
        calib_sec = time.perf_counter() - t0

        res = perplexity(m, tok, text_eval, device=dev,
                         seq_len=SEQ, max_tokens=EVAL_TOKENS, progress=False)
        rec = {
            "label": label,
            "asymmetric_calib": asym,
            "ppl": res["ppl"],
            "n_tokens": res["n_tokens"],
            "calib_sec": calib_sec,
            "eval_sec": res["sec"],
            "n_layers": meta.get("n_layers"),
            "n_asym_layers": len(meta.get("asymmetry_per_layer", {})),
        }
        # Per-layer Phase-2 diagnostic dump (only present when asym=True).
        if asym and meta.get("asymmetry_per_layer"):
            per_layer = meta["asymmetry_per_layer"]
            slim = []
            for layer_name, mm in per_layer.items():
                slim.append({
                    "name": layer_name,
                    "layer_idx": mm.get("layer_idx"),
                    "bits": mm.get("bits"),
                    "frob_delta_rel": mm.get("frob_delta_rel"),
                    "cross_off_diag_rel": mm.get("cross_off_diag_rel"),
                    "W_norm_pre": mm.get("W_norm_pre"),
                    "W_norm_post": mm.get("W_norm_post"),
                    "W_delta_rel": mm.get("W_delta_rel"),
                    "row_max_ratio_p50": mm.get("row_max_ratio_p50"),
                    "row_max_ratio_p99": mm.get("row_max_ratio_p99"),
                    "row_max_ratio_max": mm.get("row_max_ratio_max"),
                    "asym_recon_loss_proxy": mm.get("asym_recon_loss_proxy"),
                    "sym_recon_err_under_Hpost": mm.get("sym_recon_err_under_Hpost"),
                })
            rec["per_layer"] = slim
            import statistics as _s
            wdrs = [x["W_delta_rel"] for x in slim if x["W_delta_rel"] is not None]
            rmrs = [x["row_max_ratio_max"] for x in slim if x["row_max_ratio_max"] is not None]
            srl = [x["sym_recon_err_under_Hpost"] for x in slim
                   if x["sym_recon_err_under_Hpost"] is not None]
            if wdrs:
                rec["W_delta_rel_p50"] = _s.median(wdrs)
                rec["W_delta_rel_max"] = max(wdrs)
            if rmrs:
                rec["row_max_ratio_p50_overall"] = _s.median(rmrs)
                rec["row_max_ratio_max_overall"] = max(rmrs)
            if srl:
                rec["sym_recon_err_total"] = sum(srl)
        results.append(rec)
        console.log(
            f"  PPL={rec['ppl']:.4f} on {rec['n_tokens']} tokens "
            f"(calib {calib_sec:.1f}s, eval {res['sec']:.1f}s, "
            f"asym applied to {rec['n_asym_layers']} Linears)"
        )
        del m

    base = next(r for r in results if not r["asymmetric_calib"])
    asym_r = next(r for r in results if r["asymmetric_calib"])
    delta = base["ppl"] - asym_r["ppl"]
    cost_factor = asym_r["calib_sec"] / max(base["calib_sec"], 1e-6)
    console.log(
        f"  Δppl = {delta:+.4f}   (asym vs baseline)\n"
        f"  calib slowdown = {cost_factor:.2f}× ({base['calib_sec']:.1f}s → {asym_r['calib_sec']:.1f}s)"
    )

    out_path = ROOT / "results" / "tables" / "smollm135_gptaq_smoke.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": MODEL, "results": results,
                   "delta_ppl": delta, "cost_factor": cost_factor}, f, indent=2)
    console.log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
