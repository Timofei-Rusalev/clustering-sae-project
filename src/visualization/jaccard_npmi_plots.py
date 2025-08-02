# visualization/jaccard_npmi_plots.py
# -----------------------------------------------------------------------------
# Visualization of robust relative Jaccard and NPMI metrics across layers.
# Reads precomputed CSVs (jaccard_layer{L}.csv / npmi_layer{L}.csv) from
# co_cfg.results_dir and saves plots to results/plots/{jaccard,npmi}/
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# we only need paths from co_cfg; no heavy recomputation is called here
from coactivations_study import cfg as co_cfg

# ─────────────────────────────────────────────────────────────────────────────
# Visualization configuration (plotting only – no heavy compute parameters)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class VizCfg:
    # number of layers (GPT‑2 small by default)
    layers: int = 12

    # plot appearance
    figsize: tuple[int, int] = (8, 4)
    style: str = "whitegrid"
    context: str = "talk"

    # display options
    show_mean: bool = False
    show_trimmed: bool = False
    save: bool = True               # save plots to disk
    stat: str = "both"              # "jaccard", "npmi", or "both"

    # paths
    results_root: Path = co_cfg.results_dir
    plots_root: Path = Path("results") / "plots"

cfg = VizCfg()

sns.set(style=cfg.style, context=cfg.context)
cfg.plots_root.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, subdir: str, name: str) -> None:
    """Save figure if saving is enabled."""
    if not cfg.save:
        return
    out_dir = cfg.plots_root / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=150)


def _plot_trend(
    df: pd.DataFrame,
    metric_label: str,
    ylabel: str,
    fname_prefix: str,
) -> None:
    """
    Generic function to plot layer-wise trend of a metric.
    Expects columns: layer, median_ratio, mean_ratio, trimmed_mean_ratio.
    """
    fig = plt.figure(figsize=cfg.figsize)
    plt.plot(df["layer"], df["median_ratio"], "o-", label="Median")

    if cfg.show_mean:
        plt.plot(df["layer"], df["mean_ratio"], "s--", label="Mean")

    if cfg.show_trimmed:
        plt.plot(df["layer"], df["trimmed_mean_ratio"], "d:", label="Trimmed mean")

    plt.xlabel("Layer ℓ")
    plt.ylabel(ylabel)
    plt.xticks(df["layer"])
    if cfg.show_mean or cfg.show_trimmed:
        plt.legend()
    plt.tight_layout()

    _save(fig, subdir=metric_label, name=f"{fname_prefix}_trend")
    plt.show()


def _load_metric_csv(metric: str, layer: int) -> pd.DataFrame:
    """
    Load precomputed CSV for a given metric ('jaccard' or 'npmi') and layer.
    Returns an empty DataFrame if the file does not exist.
    """
    path = cfg.results_root / f"{metric}_layer{layer}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ─────────────────────────────────────────────────────────────────────────────
# Build summaries (read-only, from CSVs)
# ─────────────────────────────────────────────────────────────────────────────
def build_jaccard_summary() -> pd.DataFrame:
    """
    Build summary DataFrame for Jaccard from saved CSVs:
    columns = [layer, median_ratio, mean_ratio, trimmed_mean_ratio, n_clusters].
    """
    records: List[Dict[str, float]] = []
    for ℓ in range(cfg.layers):
        df_layer = _load_metric_csv("jaccard", ℓ)
        # skip if no file or missing 'ratio' column
        if df_layer.empty or "ratio" not in df_layer.columns:
            continue

        # extract the clipped ratio values (always non‑negative)
        ratios = df_layer["ratio"].dropna().astype(float)
        if ratios.empty:
            continue

        # compute median and mean
        median_r = ratios.median()
        mean_r   = ratios.mean()

        # compute 10% trimmed mean
        sorted_r = ratios.sort_values()
        lo = int(len(sorted_r) * 0.10)
        hi = int(len(sorted_r) * 0.90)
        trimmed_mean = sorted_r.iloc[lo:hi].mean() if hi > lo else np.nan

        records.append({
            "layer": ℓ,
            "median_ratio": float(median_r),
            "mean_ratio":   float(mean_r),
            "trimmed_mean_ratio": float(trimmed_mean),
            "n_clusters":   int(len(ratios)),
        })

    return pd.DataFrame(records).sort_values("layer")

def build_npmi_summary() -> pd.DataFrame:
    """
    Build summary DataFrame for NPMI from saved CSVs:
    columns = [layer, median_ratio, mean_ratio, trimmed_mean_ratio, n_clusters].
    """
    records: List[Dict[str, float]] = []
    for ℓ in range(cfg.layers):
        df_layer = _load_metric_csv("npmi", ℓ)
        # skip if no file or missing 'ratio' column
        if df_layer.empty or "ratio" not in df_layer.columns:
            continue

        # extract the clipped ratio values, dropping NaNs
        ratios = df_layer["ratio"].dropna().astype(float)
        if ratios.empty:
            continue

        # compute median and mean
        median_r = ratios.median()
        mean_r   = ratios.mean()

        # compute 10% trimmed mean
        sorted_r = ratios.sort_values()
        lo = int(len(sorted_r) * 0.10)
        hi = int(len(sorted_r) * 0.90)
        trimmed_mean = sorted_r.iloc[lo:hi].mean() if hi > lo else np.nan

        records.append({
            "layer": ℓ,
            "median_ratio": float(median_r),
            "mean_ratio":   float(mean_r),
            "trimmed_mean_ratio": float(trimmed_mean),
            "n_clusters":   int(len(ratios)),
        })

    return pd.DataFrame(records).sort_values("layer")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    CLI:
      --stat {jaccard,npmi,both}
      --show-mean
      --show-trimmed
      --no-save
      --layers N
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot robust relative Jaccard & NPMI from precomputed CSVs"
    )
    parser.add_argument(
        "--stat",
        choices=["jaccard", "npmi", "both"],
        default="both",
        help="Which metric(s) to plot",
    )
    parser.add_argument("--show-mean", action="store_true", help="Overlay mean ratio")
    parser.add_argument(
        "--show-trimmed", action="store_true", help="Overlay trimmed mean ratio"
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save plots")
    parser.add_argument(
        "--layers", type=int, default=cfg.layers, help="Number of layers to include"
    )

    args = parser.parse_args()

    # update cfg
    cfg.stat = args.stat
    cfg.show_mean = args.show_mean
    cfg.show_trimmed = args.show_trimmed
    cfg.save = not args.no_save
    cfg.layers = args.layers

    if cfg.stat in ("jaccard", "both"):
        df_j = build_jaccard_summary()
        if df_j.empty:
            print("No Jaccard CSVs found — run coactivations_study metrics first.")
        else:
            _plot_trend(
                df_j,
                metric_label="jaccard",
                ylabel="Robust relative Jaccard (clipped)",
                fname_prefix="jaccard",
            )

    if cfg.stat in ("npmi", "both"):
        df_n = build_npmi_summary()
        if df_n.empty:
            print("No NPMI CSVs found — run coactivations_study metrics first.")
        else:
            _plot_trend(
                df_n,
                metric_label="npmi",
                ylabel="Robust relative NPMI (clipped)",
                fname_prefix="npmi",
            )


if __name__ == "__main__":
    main()