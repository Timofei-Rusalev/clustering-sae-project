# evolution_of_clusters_sae_plots.py
# ---------------------------------------------------------------------------
# Four evolution plots across ALL metrics (cos, mse, text):
#   1) survive fraction
#   2) split fraction
#   3) merge fraction
#   4) mean leak
#
# Figures are saved to <PROJECT_ROOT>/results/plots/
# ---------------------------------------------------------------------------

from __future__ import annotations

# stdlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------#
# Resolve project root and ensure we can import core evolution utilities     #
# ---------------------------------------------------------------------------#
# This file lives in: <project_root>/src/visualization/
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Make <project_root>/src importable (for evolution_of_clusters_sae.py)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Core metrics computation (split/merge/survive/leak)
from evolution_of_clusters_sae import make_cfg, split_merge_metrics  # noqa: E402


# ---------------------------------------------------------------------------#
# Configuration                                                              #
# ---------------------------------------------------------------------------#
@dataclass
class PlotCfg:
    # Metrics to visualize
    metrics: List[str] = ("cos", "mse", "text")

    # Thresholds used by split_merge_metrics
    T_SPLIT: float = 0.20
    T_SURV: float = 0.50

    # Output directory for figures
    plot_dir: Path = PROJECT_ROOT / "results" / "plots"

    # Plot aesthetics
    style: str = "whitegrid"
    context: str = "talk"
    figsize: tuple[int, int] = (9, 4)


_cfg = PlotCfg()
_cfg.plot_dir.mkdir(parents=True, exist_ok=True)
sns.set(style=_cfg.style, context=_cfg.context)


# ---------------------------------------------------------------------------#
# Data collection                                                            #
# ---------------------------------------------------------------------------#
def collect_evolution_df() -> pd.DataFrame:
    """Collect split/merge/survive/mean_leak across all metrics and layers."""
    recs: List[pd.DataFrame] = []
    for m in _cfg.metrics:
        cfg = make_cfg(m)
        df = split_merge_metrics(cfg, T_split=_cfg.T_SPLIT, T_surv=_cfg.T_SURV)
        df["metric"] = m
        recs.append(df)
    return pd.concat(recs, ignore_index=True)


# ---------------------------------------------------------------------------#
# Plot helpers                                                               #
# ---------------------------------------------------------------------------#
def _plot_one(df_all: pd.DataFrame, column: str, ylabel: str, fname: str) -> None:
    """Plot one metric column across layers with one line per metric."""
    fig, ax = plt.subplots(figsize=_cfg.figsize)

    # line styles, marker shapes, sizes and widths per metric
    line_styles = {"cos": "-", "mse": "--", "text": "-"}
    markers     = {"cos": "s", "mse": "^",  "text": "o"}
    markersize  = {"cos": 8,   "mse": 6,    "text": 8}
    # make all lines a bit thicker, but keep mse thinner than cos/text
    linewidth   = {"cos": 2.5, "mse": 1.8,  "text": 2.5}

    for m in _cfg.metrics:
        sub = df_all[df_all["metric"] == m]
        ax.plot(
            sub["layer"],
            sub[column],
            marker=markers[m],
            markersize=markersize[m],
            linestyle=line_styles[m],
            linewidth=linewidth[m],
            label=m,
        )

    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel(ylabel)

    # legend: slightly larger entry text, balanced title size, smaller markers
    leg = ax.legend(
        title="SAE-Match metric",
        title_fontsize=11,
        prop={"size": 11},
        markerscale=0.7,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
    )

    fig.subplots_adjust(right=0.75)
    fig.tight_layout()
    fig.savefig(_cfg.plot_dir / fname, dpi=150)
    plt.show()

# ---------------------------------------------------------------------------#
# Public API                                                                 #
# ---------------------------------------------------------------------------#
def make_all_four_plots() -> None:
    """Build and save 4 figures: survive/split/merge/mean leak for all metrics."""
    df_all = collect_evolution_df()
    _plot_one(df_all, "survive_frac", "Survive fraction", "survive_frac.png")
    _plot_one(df_all, "split_frac",    "Split fraction",   "split_frac.png")
    _plot_one(df_all, "merge_frac",    "Merge fraction",   "merge_frac.png")
    _plot_one(df_all, "mean_leak",     "Mean leak",        "mean_leak.png")


# ---------------------------------------------------------------------------#
# 5) Combined Entropy & Mutual Information evolution plots                   #
# ---------------------------------------------------------------------------#

from metrics_for_cluster_evolution import make_cfg as _make_mi_cfg

def collect_entropy_mi_df() -> pd.DataFrame:
    """
    Load per‑metric evolution CSVs:
      results/transitions_2step[_mse/_text]/evolution_entropy_mi_{metric}.csv
    and concatenate into one DataFrame with a 'metric' column.
    """
    recs: List[pd.DataFrame] = []
    for m in _cfg.metrics:
        mi_cfg = _make_mi_cfg(m)
        csv_fp = mi_cfg.transitions_root / f"evolution_entropy_mi_{m}.csv"
        if csv_fp.exists():
            df = pd.read_csv(csv_fp)
            df["metric"] = m
            recs.append(df)
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()

def make_entropy_mi_combined_plots() -> None:
    """
    Build and save 2 figures in <PROJECT_ROOT>/results/plots/:
      - entropy_evolution.png  (H_cond_mean solid & H_aug_mean dashed)
      - mi_evolution.png       (mi_cond       solid & mi_aug       dashed)
    """
    df_all = collect_entropy_mi_df()
    if df_all.empty:
        print("No entropy/MI data found; skipping combined plots.")
        return

    layers = sorted(df_all["layer"].unique())
    colors     = {"cos": "C0", "mse": "C1", "text": "C2"}
    linestyles = {"cond": "-",   "aug": "--"}
    labels     = {"cond": "conditional", "aug": "augmented"}

    # 1) Entropy combined
    fig, ax = plt.subplots(figsize=_cfg.figsize)
    for m in _cfg.metrics:
        sub = df_all[df_all["metric"] == m]
        # slightly thinner orange, default for others
        lw = 1.5 if m == "mse" else 2.5
        # slightly smaller markers for orange
        ms = 6   if m == "mse" else 8
        zo = 3   if m == "mse" else 1

        # conditional
        ax.plot(
            sub["layer"], sub["H_cond_mean"],
            color=colors[m],
            linestyle=linestyles["cond"] if m!="mse" else "--",
            marker="o",
            markersize=ms,
            linewidth=lw,
            label=f"{m} ({labels['cond']})",
            zorder=zo
        )
        # augmented
        ax.plot(
            sub["layer"], sub["H_aug_mean"],
            color=colors[m],
            linestyle=linestyles["aug"] if m!="mse" else "--",
            marker="s",
            markersize=ms,
            linewidth=lw,
            label=f"{m} ({labels['aug']})",
            zorder=zo
        )
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Entropy")
    ax.set_xticks(layers)
    ax.legend(
        title="SAE-Match metric",
        title_fontsize=11,
        prop={"size":11},
        loc="center left",
        bbox_to_anchor=(1.0, 0.5)
    )
    fig.subplots_adjust(right=0.75)
    fig.tight_layout()
    fig.savefig(_cfg.plot_dir / "entropy_evolution.png", dpi=150)
    plt.show()

    # 2) MI combined
    fig, ax = plt.subplots(figsize=_cfg.figsize)
    for m in _cfg.metrics:
        sub = df_all[df_all["metric"] == m]
        lw = 1.5 if m == "mse" else 2.5
        ms = 6   if m == "mse" else 8
        zo = 3   if m == "mse" else 1

        # conditional
        ax.plot(
            sub["layer"], sub["mi_cond"],
            color=colors[m],
            linestyle=linestyles["cond"] if m!="mse" else "--",
            marker="o",
            markersize=ms,
            linewidth=lw,
            label=f"{m} ({labels['cond']})",
            zorder=zo
        )
        # augmented
        ax.plot(
            sub["layer"], sub["mi_aug"],
            color=colors[m],
            linestyle=linestyles["aug"] if m!="mse" else "--",
            marker="s",
            markersize=ms,
            linewidth=lw,
            label=f"{m} ({labels['aug']})",
            zorder=zo
        )
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Mutual Information")
    ax.set_xticks(layers)
    ax.legend(
        title="SAE-Match metric",
        title_fontsize=11,
        prop={"size":11},
        loc="center left",
        bbox_to_anchor=(1.0, 0.5)
    )
    fig.subplots_adjust(right=0.75)
    fig.tight_layout()
    fig.savefig(_cfg.plot_dir / "mi_evolution.png", dpi=150)
    plt.show()


# --- NEW: OT helpers ---------------------------------------------------------
from types import SimpleNamespace
from cluster_transport import ClusterOTConfig  # reuse tag + paths

def _ot_cfg_for(metric: str):
    """
    Build a tiny cfg-like object with attributes required by split_merge_metrics:
      - transitions_root : Path
      - n_layers         : int
    """
    cfg_ot = ClusterOTConfig()
    if metric == "text":
        cfg_ot.space = "text";    cfg_ot.metric = "text"
    else:
        cfg_ot.space = "decoder"; cfg_ot.metric = metric
    root = cfg_ot.transitions_root()  # results/transitions_2step_ot/<tag>
    return SimpleNamespace(
        transitions_root = root,
        n_layers         = cfg_ot.num_layers
    )

def collect_evolution_df_ot() -> pd.DataFrame:
    """
    Collect survive/split/merge/mean_leak across all metrics from OT transitions.
    """
    recs: List[pd.DataFrame] = []
    for m in _cfg.metrics:
        cfg_like = _ot_cfg_for(m)
        df = split_merge_metrics(cfg_like, T_split=_cfg.T_SPLIT, T_surv=_cfg.T_SURV)
        df["metric"] = m
        recs.append(df)
    return pd.concat(recs, ignore_index=True)

def _plot_one_ot(
    df_all: pd.DataFrame,
    column: str,
    ylabel: str,
    fname: str,
) -> None:
    """
    Like _plot_one, but for OT transitions:
      • Uses solid lines of equal thickness for all metrics (including mse)
      • Legend title reflects “OT cost metric”
    """
    fig, ax = plt.subplots(figsize=_cfg.figsize)

    # all metrics use solid lines, same width & marker size
    line_styles = {"cos": "-", "mse": "-", "text": "-"}
    markers     = {"cos": "s", "mse": "^", "text": "o"}
    markersize  = {"cos": 8,   "mse": 8,    "text": 8}
    linewidth   = {"cos": 2.5, "mse": 2.5,  "text": 2.5}

    for m in _cfg.metrics:
        sub = df_all[df_all["metric"] == m]
        ax.plot(
            sub["layer"],
            sub[column],
            marker=markers[m],
            markersize=markersize[m],
            linestyle=line_styles[m],
            linewidth=linewidth[m],
            label=m,
        )

    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel(ylabel)

    # legend title changed to OT cost metric
    ax.legend(
        title="OT cost metric",
        title_fontsize=11,
        prop={"size": 11},
        markerscale=0.7,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
    )

    fig.subplots_adjust(right=0.75)
    fig.tight_layout()
    fig.savefig(_cfg.plot_dir / fname, dpi=150)
    plt.show()

def make_all_four_plots_ot() -> None:
    """
    Same 4 plots as make_all_four_plots(), but from OT transitions.
    Files are prefixed with 'ot_' to avoid overwriting SAE‑Match figures.
    """
    df_all = collect_evolution_df_ot()
    _plot_one_ot(df_all, "survive_frac", "Survive fraction", "ot_survive_frac.png")
    _plot_one_ot(df_all, "split_frac",    "Split fraction",   "ot_split_frac.png")
    _plot_one_ot(df_all, "merge_frac",    "Merge fraction",   "ot_merge_frac.png")
    _plot_one_ot(df_all, "mean_leak",     "Mean leak",        "ot_mean_leak.png")


def collect_entropy_mi_df_ot() -> pd.DataFrame:
    """
    Read evolution_entropy_mi_{metric}.csv from results/transitions_2step_ot/<tag>/...
    """
    recs: List[pd.DataFrame] = []
    for m in _cfg.metrics:
        cfg_like = _ot_cfg_for(m)
        csv_fp = cfg_like.transitions_root / f"evolution_entropy_mi_{m}.csv"
        if csv_fp.exists():
            df = pd.read_csv(csv_fp)
            df["metric"] = m
            recs.append(df)
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()

def make_entropy_mi_combined_plots_ot() -> None:
    """
    Build and save 2 figures from OT CSVs:
      - ot_entropy_evolution.png (H_cond_mean solid & H_aug_mean dashed)
      - ot_mi_evolution.png      (mi_cond       solid & mi_aug       dashed)
    Uses uniform line width & solid conditional lines (including mse).
    Legend title reflects “OT cost metric”.
    """
    df_all = collect_entropy_mi_df_ot()
    if df_all.empty:
        print("No OT entropy/MI data found; skipping combined plots.")
        return

    layers = sorted(df_all["layer"].unique())
    colors     = {"cos": "C0", "mse": "C1", "text": "C2"}
    linestyles = {"cond": "-", "aug": "--"}
    labels     = {"cond": "conditional", "aug": "augmented"}

    # Entropy
    fig, ax = plt.subplots(figsize=_cfg.figsize)
    for m in _cfg.metrics:
        sub = df_all[df_all["metric"] == m]
        lw, ms, zo = 2.5, 8, 1
        # conditional (solid for all, including mse)
        ax.plot(
            sub["layer"], sub["H_cond_mean"],
            color=colors[m], linestyle=linestyles["cond"],
            marker="o", markersize=ms, linewidth=lw,
            label=f"{m} ({labels['cond']})", zorder=zo
        )
        # augmented (dashed for all)
        ax.plot(
            sub["layer"], sub["H_aug_mean"],
            color=colors[m], linestyle=linestyles["aug"],
            marker="s", markersize=ms, linewidth=lw,
            label=f"{m} ({labels['aug']})", zorder=zo
        )

    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Entropy")
    ax.set_xticks(layers)
    ax.legend(
        title="OT cost metric",
        title_fontsize=11,
        prop={"size":11},
        loc="center left",
        bbox_to_anchor=(1.0, 0.5)
    )
    fig.subplots_adjust(right=0.75)
    fig.tight_layout()
    fig.savefig(_cfg.plot_dir / "ot_entropy_evolution.png", dpi=150)
    plt.show()

    # Mutual Information
    fig, ax = plt.subplots(figsize=_cfg.figsize)
    for m in _cfg.metrics:
        sub = df_all[df_all["metric"] == m]
        lw, ms, zo = 2.5, 8, 1
        ax.plot(
            sub["layer"], sub["mi_cond"],
            color=colors[m], linestyle=linestyles["cond"],
            marker="o", markersize=ms, linewidth=lw,
            label=f"{m} ({labels['cond']})", zorder=zo
        )
        ax.plot(
            sub["layer"], sub["mi_aug"],
            color=colors[m], linestyle=linestyles["aug"],
            marker="s", markersize=ms, linewidth=lw,
            label=f"{m} ({labels['aug']})", zorder=zo
        )

    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Mutual Information")
    ax.set_xticks(layers)
    ax.legend(
        title="OT cost metric",
        title_fontsize=11,
        prop={"size":11},
        loc="center left",
        bbox_to_anchor=(1.0, 0.5)
    )
    fig.subplots_adjust(right=0.75)
    fig.tight_layout()
    fig.savefig(_cfg.plot_dir / "ot_mi_evolution.png", dpi=150)
    plt.show()
    
# ---------------------------------------------------------------------------#
# CLI                                                                         #
# ---------------------------------------------------------------------------#
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot survive/split/merge/mean-leak across metrics (cos, mse, text)."
    )
    parser.add_argument("--t_split", type=float, default=_cfg.T_SPLIT,
                        help=f"Threshold for split (default: {_cfg.T_SPLIT})")
    parser.add_argument("--t_surv", type=float, default=_cfg.T_SURV,
                        help=f"Threshold for survive (default: {_cfg.T_SURV})")
    parser.add_argument("--out", type=Path, default=_cfg.plot_dir,
                        help="Output directory for figures (default: results/plots)")
    args = parser.parse_args()

    # Update cfg from CLI
    _cfg.T_SPLIT = float(args.t_split)
    _cfg.T_SURV = float(args.t_surv)
    _cfg.plot_dir = args.out
    _cfg.plot_dir.mkdir(parents=True, exist_ok=True)

    make_all_four_plots()


if __name__ == "__main__":
    make_all_four_plots()