"""Generate comparison plots from results/tables/*.json.

Outputs:
  results/plots/llm_ppl_bar.png         (PPL by method per LLM)
  results/plots/cnn_top1_bar.png        (top-1 by method per CNN)
  results/tables/llm_results.md         (markdown table for README)
  results/tables/cnn_results.md         (markdown table for README)
  results/tables/all_results.csv        (combined CSV)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(p: Path) -> list:
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("runs", [])


def _short_model(name: str) -> str:
    return name.split("/")[-1]


def llm_table(rows: list) -> pd.DataFrame:
    keep = [r for r in rows if "ppl" in r]
    df = pd.DataFrame(keep)
    if df.empty:
        return df
    df["model"] = df["model"].map(_short_model)
    cols = ["model", "method", "bits", "ppl", "tok_per_sec", "calib_sec", "n_eval_tokens"]
    return df[[c for c in cols if c in df.columns]].sort_values(["model", "method"])


def cnn_table(rows: list) -> pd.DataFrame:
    keep = [r for r in rows if "top1" in r]
    df = pd.DataFrame(keep)
    if df.empty:
        return df
    cols = ["model", "method", "bits", "top1", "top5", "calib_sec", "n_eval"]
    return df[[c for c in cols if c in df.columns]].sort_values(["model", "method"])


def plot_llm(df: pd.DataFrame, out: Path):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot = df.pivot_table(index="model", columns="method", values="ppl")
    pivot.plot.bar(ax=ax)
    ax.set_ylabel("WikiText-2 perplexity (lower = better)")
    ax.set_title("LLM perplexity vs method (M1 / MPS)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140)
    plt.close()


def plot_cnn(df: pd.DataFrame, out: Path):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot = df.pivot_table(index="model", columns="method", values="top1")
    (pivot * 100).plot.bar(ax=ax)
    ax.set_ylabel("ImageNetV2 top-1 accuracy (%)")
    ax.set_title("CNN/ViT top-1 vs method (M1 / MPS, ImageNetV2 matched-frequency)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140)
    plt.close()


def df_to_md(df: pd.DataFrame, float_fmt: dict[str, str]) -> str:
    if df.empty:
        return "_(no results)_\n"
    df = df.copy()
    for col, fmt in float_fmt.items():
        if col in df.columns:
            df[col] = df[col].map(lambda v: fmt.format(v) if pd.notna(v) else "—")
    md = df.to_markdown(index=False)
    return md + "\n"


def main():
    tables_dir = ROOT / "results" / "tables"
    plots_dir = ROOT / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    llm_rows = _load(tables_dir / "llm_sweep.json")
    cnn_rows = _load(tables_dir / "cnn_sweep.json")

    llm_df = llm_table(llm_rows)
    cnn_df = cnn_table(cnn_rows)

    plot_llm(llm_df, plots_dir / "llm_ppl_bar.png")
    plot_cnn(cnn_df, plots_dir / "cnn_top1_bar.png")

    (tables_dir / "llm_results.md").write_text(
        "# LLM results (WikiText-2 perplexity)\n\n"
        + df_to_md(llm_df, {"ppl": "{:.2f}", "tok_per_sec": "{:.1f}", "calib_sec": "{:.0f}"})
    )
    (tables_dir / "cnn_results.md").write_text(
        "# CNN results (ImageNetV2 matched-frequency)\n\n"
        + df_to_md(cnn_df, {"top1": "{:.4f}", "top5": "{:.4f}", "calib_sec": "{:.0f}"})
    )

    # Combined CSV
    combined = pd.concat(
        [
            llm_df.assign(task="llm"),
            cnn_df.assign(task="cnn"),
        ],
        ignore_index=True,
    )
    combined.to_csv(tables_dir / "all_results.csv", index=False)
    print(f"wrote tables and plots in {tables_dir} / {plots_dir}")
    print(f"  LLM rows: {len(llm_df)}  CNN rows: {len(cnn_df)}")


if __name__ == "__main__":
    main()
