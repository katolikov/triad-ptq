"""Phase-5: assemble the final exynos_comparison.md table.

Reads three result JSONs (FP16 reference, MLC q4f16_1 community baseline,
TRIAD-INT4 this work) and writes a Markdown comparison table to
results/exynos_comparison.md.

This script does NOT run anything on the device. It is the final-stage
collation that runs once the three source JSONs exist. In autonomous
mode, the FP16 number is the only one we have on M1; the other two
require the deferred device benchmarks (ADR-003) and will be filled in
by the user after running the manual MLC install + bench step.

Inputs (all under results/):
  fp16_tinyllama_m1.json
  baseline_tinyllama_q4f16_1.json
  triad_tinyllama_int4_exynos.json   <-- written by on-device run
  triad_tinyllama_int4_m1.json       <-- written by 13_tinyllama_phase3.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def _read_optional(name: str) -> dict | None:
    p = RES / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"_error": str(e), "_path": str(p)}


def _fmt(x, precision: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    if isinstance(x, (int,)):
        return str(x)
    try:
        return f"{float(x):.{precision}f}"
    except Exception:
        return str(x)


def main():
    fp16 = _read_optional("fp16_tinyllama_m1.json")
    base = _read_optional("baseline_tinyllama_q4f16_1.json")
    triad_dev = _read_optional("triad_tinyllama_int4_exynos.json")
    triad_m1 = _read_optional("triad_tinyllama_int4_m1.json")

    rows = []

    def _row(label, src, *, ppl_key="ppl_wikitext2",
             tps_key="decode_tok_per_sec", mem_key="peak_gpu_mem_mb",
             disk_key="disk_size_mb", bits=None):
        if not src:
            return [label, _fmt(bits) or "—", "—", "—", "—", "—", "—"]
        ppl = src.get(ppl_key) or src.get("ppl")
        hella = src.get("hellaswag_acc")
        tps = src.get(tps_key) or src.get("tok_per_sec_decode") or src.get("tok_per_sec")
        mem = src.get(mem_key) or src.get("peak_gpu_mb") or src.get("peak_mem_mb")
        disk = src.get(disk_key) or src.get("size_mb")
        return [label, _fmt(bits) or "—", _fmt(ppl, 3), _fmt(hella, 3),
                _fmt(tps, 1), _fmt(mem, 1), _fmt(disk, 1)]

    rows.append(_row("FP16 (reference)", fp16, bits=16))
    rows.append(_row("MLC q4f16_1 (community baseline)", base, bits=4))
    rows.append(_row("**TRIAD-INT4 (this work)**", triad_dev, bits=4))

    # Render Markdown
    header = ["Method", "Bits", "WikiText-2 PPL", "HellaSwag",
              "Tok/s decode", "Peak GPU MB", "Disk MB"]
    lines = []
    lines.append("# TRIAD-PTQ on Exynos 2500 — final comparison\n")
    lines.append("Acceptance criteria (top of session prompt):\n")
    lines.append("- WikiText-2 PPL TRIAD-INT4 vs FP16: **≤ +1.0**")
    lines.append("- Decode throughput on device (batch=1): **≥ 25 tok/s**")
    lines.append("- Peak GPU memory during decode: **≤ 1.2 GB**\n")

    if triad_m1 is not None:
        lines.append("## M1-side checkpoint summary\n")
        lines.append(f"- Model: `{triad_m1.get('model', '?')}`")
        lines.append(f"- Calibration time (M1, fp32): {_fmt(triad_m1.get('calib_sec'), 1)} s")
        lines.append(
            f"- Peak MPS allocation during calib: "
            f"{_fmt(triad_m1.get('peak_mps_gb_during_calib'), 2)} GB"
        )
        lines.append(
            f"- Simulated INT4 PPL on M1: "
            f"{_fmt(triad_m1.get('ppl_wikitext2'), 3)} "
            f"(on {triad_m1.get('ppl_n_tokens', '?')} tokens)"
        )
        lines.append(
            f"- Checkpoint: `{triad_m1.get('checkpoint_path', '?')}` "
            f"({_fmt(triad_m1.get('checkpoint_size_mb'), 1)} MB)\n"
        )

    lines.append("## On-device comparison\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")

    notes = []
    if base is None:
        notes.append("- MLC q4f16_1 community baseline row is empty: Phase 1 deferred per ADR-003.")
    if triad_dev is None:
        notes.append("- TRIAD-INT4 device row is empty: Phase 5 device bench requires the manual MLC runtime install (see STATUS.md).")
    if notes:
        lines.append("## Notes\n")
        lines.extend(notes)

    out = RES / "exynos_comparison.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
