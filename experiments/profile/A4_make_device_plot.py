"""Generate the device-bench bar plot for the README.

Reads experiments/profile/A3_replicated_results.json (N=3 means + std)
and emits results/plots/exynos_device_bench.png with two side-by-side
bars (decode tok/s, prefill tok/s) per model with std error bars.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments" / "profile" / "A3_replicated_results.json"
OUT = ROOT / "results" / "plots" / "exynos_device_bench.png"


def main():
    data = json.loads(SRC.read_text())
    s = data["summary"]

    labels = ["MLC q4f16_1\n(community)", "TRIAD-INT4\n(this work)"]
    prefill_means = [s["ref"]["prefill_mean_tok_s"], s["triad"]["prefill_mean_tok_s"]]
    prefill_stds = [s["ref"]["prefill_std_tok_s"], s["triad"]["prefill_std_tok_s"]]
    decode_means = [s["ref"]["decode_mean_tok_s"], s["triad"]["decode_mean_tok_s"]]
    decode_stds = [s["ref"]["decode_std_tok_s"], s["triad"]["decode_std_tok_s"]]

    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=140)
    b1 = ax.bar(x - w / 2, prefill_means, w, yerr=prefill_stds,
                capsize=4, label="Prefill (tok/s)", color="#5B8DEF")
    b2 = ax.bar(x + w / 2, decode_means, w, yerr=decode_stds,
                capsize=4, label="Decode (tok/s)", color="#F08856")

    # Acceptance line at 25 tok/s decode
    ax.axhline(y=25.0, linestyle="--", color="#888", linewidth=1)
    ax.text(len(labels) - 0.5, 25.5, "Decode acceptance \u2265 25 tok/s",
            color="#444", fontsize=8, ha="right")

    for bars, vals, errs in ((b1, prefill_means, prefill_stds),
                              (b2, decode_means, decode_stds)):
        for bar, v, e in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2, v + e + 0.5,
                    f"{v:.1f} \u00b1 {e:.1f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title(
        "TinyLlama-1.1B on Galaxy Z Flip7 / Exynos 2500 / Xclipse 950\n"
        "MLC q4f16_1 OpenCL kernels, N=3 averaged"
    )
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(max(prefill_means + decode_means) + 6, 50))
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
