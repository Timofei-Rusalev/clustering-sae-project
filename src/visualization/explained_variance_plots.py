# explained_variance_plots.py
# ---------------------------------------------------------------------------
# Visualise layer‑to‑layer explained‑variance (EV) ratios for SAE‑Match
# permutations (metric: cos / mse / text) and save plots in results/plots/ev.
# Works both with:
#   • per-metric JSONs   <perm_root>/<metric>/ev_all_metrics.json
#   • one aggregated JSON <perm_root>/ev/ev_all_metrics.json
# ---------------------------------------------------------------------------

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sae_match import cfg as _cfg

# ---------------------------------------------------------------------------#
# Config                                                                     #
# ---------------------------------------------------------------------------#
@dataclass
class EVPlotConfig:
    # root where permutations & EV JSONs live (taken from sae_match.cfg)
    perm_root:    Path          = _cfg.perm_root
    # default JSON filename(s)
    default_json_single: str     = "ev_all_metrics.json"
    default_json_per_metric: str = "ev_all_metrics.json"
    # number of layer‑pairs, driven by sae_match.cfg
    layers:       int           = _cfg.num_layers - 1

    # plot_dir: Path = Path("results") / "plots" / "ev"
    plot_dir:     Path          = _cfg.perm_root.parent / "plots"

    style:        str           = "whitegrid"
    context:      str           = "talk"
    figsize:      Tuple[int,int]= (8, 4)

cfg = EVPlotConfig()
cfg.plot_dir.mkdir(parents=True, exist_ok=True)
sns.set(style=cfg.style, context=cfg.context)

# ---------------------------------------------------------------------------#
# Helpers                                                                    #
# ---------------------------------------------------------------------------#
def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"EV JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _series_from_mapping(mapping: Dict[str, float]) -> np.ndarray:
    """Convert mapping {'0->1':v0, '1->2':v1,…} into array [v0, v1, …] of length cfg.layers."""
    arr = np.full(cfg.layers, np.nan, dtype=float)
    for k, v in mapping.items():
        try:
            src, tgt = map(int, k.split("->"))
            if tgt == src + 1 and 0 <= src < cfg.layers:
                arr[src] = float(v)
        except Exception:
            continue
    return arr

def _save(fig, name: str = "ev_ratio_curves"):
    """Save figure under cfg.plot_dir."""
    fig.savefig(cfg.plot_dir / f"{name}.png", dpi=150)

def _default_aggregated_path() -> Path:
    """<perm_root>/ev/ev_all_metrics.json"""
    return cfg.perm_root / "ev" / cfg.default_json_single

def _default_per_metric_path(metric: str) -> Path:
    """<perm_root>/<metric>/ev_all_metrics.json"""
    return cfg.perm_root / metric / cfg.default_json_per_metric

# ---------------------------------------------------------------------------#
# Plotting                                                                   #
# ---------------------------------------------------------------------------#
def plot_ev_curves(
    cos_json:  Optional[Path] = None,
    mse_json:  Optional[Path] = None,
    text_json: Optional[Path] = None,
    aggregated_json: Optional[Path] = None,
) -> None:
    """
    Draw EV‑ratio curves (lower is better) for any subset of metrics:
      - Pass per-metric JSONs (cos_json/mse_json/text_json), OR
      - Pass one aggregated JSON (aggregated_json) that contains keys
        {"cos": {...}, "mse": {...}, "text": {...}}.
    """
    layers = np.arange(cfg.layers)
    fig = plt.figure(figsize=cfg.figsize)

    if aggregated_json and aggregated_json.exists():
        data_all = _load_json(aggregated_json)
        if "cos"  in data_all: plt.plot(layers, _series_from_mapping(data_all["cos"]),  "o-",  label="cosine")
        if "mse"  in data_all: plt.plot(layers, _series_from_mapping(data_all["mse"]),  "s--", label="mse")
        if "text" in data_all: plt.plot(layers, _series_from_mapping(data_all["text"]), "x-.", label="text")
    else:
        if cos_json  and cos_json.exists():
            plt.plot(layers, _series_from_mapping(_load_json(cos_json)),  "o-",  label="cosine")
        if mse_json  and mse_json.exists():
            plt.plot(layers, _series_from_mapping(_load_json(mse_json)),  "s--", label="mse")
        if text_json and text_json.exists():
            plt.plot(layers, _series_from_mapping(_load_json(text_json)), "x-.", label="text")

    plt.xlabel("Layer pair ℓ→ℓ+1")
    plt.ylabel("Explained Variance")
    plt.xticks(layers)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    _save(fig)
    plt.show()

# ---------------------------------------------------------------------------#
# CLI                                                                        #
# ---------------------------------------------------------------------------#
def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Plot explained‑variance ratios for SAE‑Match permutations."
    )
    p.add_argument("--cos",  type=Path, default=None,
                   help=f"Per-metric JSON for cosine metric (default: {_default_per_metric_path('cos')})")
    p.add_argument("--mse",  type=Path, default=None,
                   help=f"Per-metric JSON for MSE   metric (default: {_default_per_metric_path('mse')})")
    p.add_argument("--text", type=Path, default=None,
                   help=f"Per-metric JSON for text  metric (default: {_default_per_metric_path('text')})")
    p.add_argument("--all",  type=Path, default=None,
                   help=f"Aggregated all-metrics JSON (default: {_default_aggregated_path()})")

    args = p.parse_args()

    # Resolve defaults
    aggregated_json = args.all or (_default_aggregated_path() if _default_aggregated_path().exists() else None)
    if aggregated_json and aggregated_json.exists():
        plot_ev_curves(aggregated_json=aggregated_json)
        return

    cos_json  = args.cos  or (_default_per_metric_path("cos")  if _default_per_metric_path("cos").exists()  else None)
    mse_json  = args.mse  or (_default_per_metric_path("mse")  if _default_per_metric_path("mse").exists()  else None)
    text_json = args.text or (_default_per_metric_path("text") if _default_per_metric_path("text").exists() else None)

    if not (cos_json or mse_json or text_json):
        p.error("No EV JSON files found or specified (neither aggregated nor per-metric).")

    plot_ev_curves(cos_json=cos_json, mse_json=mse_json, text_json=text_json)

if __name__ == "__main__":
    main()