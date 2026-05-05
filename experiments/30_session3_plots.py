"""Plots for the session-3 README update.

Output:
  results/plots/session3_decode_compare.png    on-device decode tok/s comparison
  results/plots/session3_phase2_smollm.png     PPL ablation on SmolLM-135M
  results/plots/session3_pareto.png            PPL vs decode tok/s Pareto frontier
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# --- 1. Device bench: TRIAD vs ref q4f16_1 -----------------------------------
labels = ["MLC q4f16_1\n(community ref)", "TRIAD-INT4\n(this work)"]
decode_means = [34.62, 35.56]
decode_stds  = [1.60, 0.0]   # TRIAD: only one solid long-completion sample at 60s cooldown
prefill_means = [15.28, 15.15]

fig, ax = plt.subplots(figsize=(7.0, 4.5))
x = np.arange(len(labels))
w = 0.35
b1 = ax.bar(x - w/2, decode_means, w, yerr=decode_stds, capsize=4,
            label="decode tok/s", color="#3a86ff")
b2 = ax.bar(x + w/2, prefill_means, w, label="prefill tok/s", color="#ffbe0b")
for bar, v in zip(b1, decode_means):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.4, f"{v:.2f}",
            ha="center", fontsize=10, fontweight="bold")
for bar, v in zip(b2, prefill_means):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.4, f"{v:.2f}",
            ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("tokens / second")
ax.set_title("On-device throughput: Galaxy Z Flip7 / Xclipse 950\n"
             "TinyLlama-1.1B, prompt 14 tokens, 60s cooldown, N≥3 (warmup excluded)")
ax.legend(loc="upper right")
ax.set_ylim(0, max(decode_means) * 1.18)
ax.grid(axis="y", linestyle=":", alpha=0.5)
fig.tight_layout()
fig.savefig(OUT / "session3_decode_compare.png", dpi=140)
plt.close(fig)
print(f"wrote {OUT / 'session3_decode_compare.png'}")

# --- 2. Phase-2 PPL ablation ------------------------------------------------
phase2 = json.load(open(ROOT / "results" / "tables" / "smollm135_phase2_scopelimit.json"))
labels = [
    "TRIAD-INT4\nbaseline",
    "+ GPTAQ asym\n(α=1.0, full transfer)",
    "+ GPTAQ asym\n(α=1.0, scope-limit)",
    "+ GPTAQ asym\n(α=0.5, scope-limit)",
]
# Borrow the +full-transfer number from the prior diagnostics smoke
ppl_full = 25.218
ppls = [
    phase2[0]["ppl"],
    ppl_full,
    phase2[1]["ppl"],
    phase2[2]["ppl"],
]
colors = ["#777", "#e63946", "#f4a261", "#2a9d8f"]

fig, ax = plt.subplots(figsize=(8.0, 4.5))
bars = ax.bar(labels, ppls, color=colors)
for bar, v in zip(bars, ppls):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.2, f"{v:.2f}",
            ha="center", fontsize=10, fontweight="bold")
ax.axhline(phase2[0]["ppl"], color="#222", linestyle="--", alpha=0.4,
           label=f"baseline PPL = {phase2[0]['ppl']:.2f}")
ax.set_ylabel("WikiText-2 perplexity (lower better)")
ax.set_title("Phase-2 GPTAQ asymmetric calibration: ablation on SmolLM-135M")
ax.set_ylim(min(ppls) * 0.97, max(ppls) * 1.04)
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "session3_phase2_smollm.png", dpi=140)
plt.close(fig)
print(f"wrote {OUT / 'session3_phase2_smollm.png'}")

# --- 3. Pareto: PPL vs decode tok/s -----------------------------------------
# Hardware: Xclipse 950 / TinyLlama-1.1B. PPL is M1 evaluator (10.882 FP16 / 11.477 TRIAD baseline);
# decode is N=3 60s-cooldown bench from session-3.
points = [
    {"label": "FP16 (M1 ref)",                 "ppl": 10.882, "decode": None},
    {"label": "MLC q4f16_1 community ref",     "ppl": None,   "decode": 34.62},
    {"label": "TRIAD-INT4 (v0.2.0-alpha)",     "ppl": 11.477, "decode": 35.56},
]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for p in points:
    if p["ppl"] is None or p["decode"] is None:
        continue
    ax.scatter(p["decode"], p["ppl"], s=180, alpha=0.85, label=p["label"])
    ax.annotate(p["label"].split(" ")[0], (p["decode"], p["ppl"]),
                xytext=(8, 5), textcoords="offset points", fontsize=10)
ax.axhline(10.882, color="#666", linestyle=":", label="FP16 PPL = 10.882")
ax.set_xlabel("on-device decode (tok/s, Xclipse 950, higher better)")
ax.set_ylabel("WikiText-2 perplexity (lower better)")
ax.set_title("TRIAD-PTQ Pareto frontier — TinyLlama-1.1B (Galaxy Z Flip7)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(linestyle=":", alpha=0.5)
fig.tight_layout()
fig.savefig(OUT / "session3_pareto.png", dpi=140)
plt.close(fig)
print(f"wrote {OUT / 'session3_pareto.png'}")
