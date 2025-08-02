# visualization/cluster_plots.py
# ============================================================================
# Minimal visualisation helpers for two‑stage clustering (UMAP + HDBSCAN)
# Each plot saves to results/plots/ **and**, when cfg.display == True,
# is shown inline (ideal for notebooks).
# ============================================================================

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------- #
# Configuration                                                              #
# -------------------------------------------------------------------------- #

# Determine project root two levels above this file
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class VizConfig:
    # data locations (absolute paths from project root)
    out_dir:    Path = _PROJECT_ROOT / "results" / "clusters_all_layers"
    labels_dir: Path = _PROJECT_ROOT / "results" / "labels"
    plot_dir:   Path = _PROJECT_ROOT / "results" / "plots"

    # global options
    layers:      int  = 12     # number of GPT-2 layers (0–11)
    min_primary: int  = 8      # cluster size ≥ 8 ⇒ “primary”, else “refined”
    style:       str  = "whitegrid"
    context:     str  = "talk"
    figsize_small: tuple[int, int] = (8, 4)
    figsize_large: tuple[int, int] = (10, 5)
    stage_palette: dict[str, str] = field(
        default_factory=lambda: {"primary": "C0", "refined": "C1"}
    )

    # whether to display plots inline
    display: bool = False

# initialize config, create plot directory, and apply seaborn style
cfg = VizConfig()
cfg.plot_dir.mkdir(parents=True, exist_ok=True)
sns.set(style=cfg.style, context=cfg.context)

# -------------------------------------------------------------------------- #
# Helpers                                                                    #
# -------------------------------------------------------------------------- #
def _show():
    if cfg.display:
        plt.show()

def _load_metrics(layer: int) -> tuple[dict, dict]:
    with (cfg.out_dir / f"layer{layer:02d}_metrics.json").open(encoding="utf-8") as f:
        js = json.load(f)
    return js["primary_metrics"], js["refined_metrics"]

def _load_label_vector(layer: int) -> np.ndarray:
    path = cfg.labels_dir / f"cluster_labels_L{layer:02d}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)

# -------------------------------------------------------------------------- #
# 1. Global metrics                                                          #
# -------------------------------------------------------------------------- #
def build_global_metrics_df() -> pd.DataFrame:
    rows = []
    for ℓ in range(cfg.layers):
        meta = cfg.out_dir / f"layer{ℓ:02d}_metrics.json"
        if not meta.exists():
            continue
        pri, ref = _load_metrics(ℓ)
        rows.append(
            dict(layer=ℓ,
                 noise_primary=pri["noise_ratio"],
                 noise_refined=ref["noise_ratio"],
                 coh_primary=pri["mean_coherence"],
                 coh_refined=ref["mean_coherence"])
        )
    return (pd.DataFrame(rows).set_index("layer")
            if rows else pd.DataFrame())

def plot_noise_ratio(df: pd.DataFrame) -> None:
    if df.empty:
        return
    plt.figure(figsize=cfg.figsize_small)
    plt.plot(df.index, df["noise_primary"], "o-", label="Noise (primary)")
    plt.plot(df.index, df["noise_refined"], "s-", label="Noise (refined)")
    plt.xlabel("Layer ℓ"); plt.ylabel("Noise ratio"); plt.xticks(df.index)
    plt.legend(); plt.tight_layout()
    plt.savefig(cfg.plot_dir / "noise_ratio.png")
    _show(); plt.close()

def plot_mean_coherence(df: pd.DataFrame) -> None:
    if df.empty:
        return
    plt.figure(figsize=cfg.figsize_small)
    plt.plot(df.index, df["coh_primary"], "o-", label="Coherence (primary)")
    plt.plot(df.index, df["coh_refined"], "s-", label="Coherence (refined)")
    plt.xlabel("Layer ℓ"); plt.ylabel("Mean coherence"); plt.xticks(df.index)
    plt.legend(); plt.tight_layout()
    plt.savefig(cfg.plot_dir / "coherence.png")
    _show(); plt.close()

# -------------------------------------------------------------------------- #
# 2. Cluster‑size distribution                                               #
# -------------------------------------------------------------------------- #
def build_cluster_size_df() -> pd.DataFrame:
    rows: List[dict] = []
    for ℓ in range(cfg.layers):
        csv = cfg.out_dir / f"layer{ℓ:02d}_clusters.csv"
        if not csv.exists():
            continue
        df_u = pd.read_csv(csv)
        counts = df_u.loc[df_u["cluster"] >= 0, "cluster"].value_counts()
        for cid, sz in counts.items():
            rows.append(dict(layer=ℓ,
                             cluster_id=cid,
                             size=sz,
                             stage="primary" if sz >= cfg.min_primary else "refined"))
    return pd.DataFrame(rows)

def plot_cluster_size_hist(df: pd.DataFrame) -> None:
    if df.empty:
        return
    plt.figure(figsize=cfg.figsize_large)
    sns.histplot(data=df, x="size", hue="stage", bins=50, multiple="stack",
                 palette=cfg.stage_palette, log_scale=(True, False))
    plt.xlabel("Cluster size"); plt.ylabel("Number of clusters")
    plt.tight_layout()
    plt.savefig(cfg.plot_dir / "cluster_size_hist.png")
    _show(); plt.close()

# -------------------------------------------------------------------------- #
# 3. Layer‑wise stats                                                        #
# -------------------------------------------------------------------------- #
def compute_layer_stats() -> pd.DataFrame:
    rows = []
    for ℓ in range(cfg.layers):
        lbl = _load_label_vector(ℓ)
        sem = lbl[lbl >= 0]
        rows.append(dict(layer=ℓ,
                         n_clusters=np.unique(sem).size,
                         mean_size=np.bincount(sem).mean() if sem.size else np.nan,
                         n_semantic=int(sem.size),
                         n_noise=int((lbl == -1).sum()),
                         n_empty=int((lbl == -2).sum())))
    return pd.DataFrame(rows)

def plot_n_clusters(df: pd.DataFrame) -> None:
    plt.figure(figsize=cfg.figsize_small)
    plt.plot(df["layer"], df["n_clusters"], "o-")
    plt.xlabel("Layer ℓ"); plt.ylabel("Number of clusters")
    plt.xticks(df["layer"]); plt.tight_layout()
    plt.savefig(cfg.plot_dir / "n_clusters.png")
    _show(); plt.close()

def plot_mean_cluster_size(df: pd.DataFrame) -> None:
    plt.figure(figsize=cfg.figsize_small)
    plt.plot(df["layer"], df["mean_size"], "s-", color="C1")
    plt.xlabel("Layer ℓ"); plt.ylabel("Mean cluster size")
    plt.xticks(df["layer"]); plt.tight_layout()
    plt.savefig(cfg.plot_dir / "mean_cluster_size.png")
    _show(); plt.close()

def plot_layer_composition(df: pd.DataFrame) -> None:
    plt.figure(figsize=cfg.figsize_large)
    width = 0.7
    plt.bar(df["layer"], df["n_semantic"], width, label="Semantic", color="C2")
    plt.bar(df["layer"], df["n_noise"], width, bottom=df["n_semantic"],
            label="Noise", color="C3")
    plt.bar(df["layer"], df["n_empty"], width,
            bottom=df["n_semantic"] + df["n_noise"],
            label="No description", color="C4")
    plt.xlabel("Layer ℓ"); plt.ylabel("Elements"); plt.xticks(df["layer"])
    plt.legend(); plt.tight_layout()
    plt.savefig(cfg.plot_dir / "layer_composition.png")
    _show(); plt.close()

# -------------------------------------------------------------------------- #
# 4. Quality metrics                                                         #
# -------------------------------------------------------------------------- #
def build_quality_df() -> pd.DataFrame:
    rows = []
    for ℓ in range(cfg.layers):
        meta = cfg.out_dir / f"layer{ℓ:02d}_metrics.json"
        if not meta.exists():
            continue
        _, ref = _load_metrics(ℓ)
        rows.append(dict(layer=ℓ,
                         silhouette=ref.get("silhouette", np.nan),
                         db=ref.get("davies_bouldin", np.nan),
                         ch=ref.get("calinski_harabasz", np.nan)))
    return pd.DataFrame(rows).set_index("layer")

def _plot_line(df: pd.DataFrame, col: str, fname: str, marker: str, color: str):
    if df.empty:
        return
    plt.figure(figsize=cfg.figsize_small)
    plt.plot(df.index, df[col], marker + "-", color=color)
    plt.xlabel("Layer ℓ"); plt.ylabel(col.replace("_", " ").title())
    plt.xticks(df.index); plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout(); plt.savefig(cfg.plot_dir / f"{fname}.png")
    _show(); plt.close()

def plot_silhouette(df):         _plot_line(df, "silhouette",      "silhouette",      "o", "C0")
def plot_davies_bouldin(df):     _plot_line(df, "db",              "davies_bouldin",  "s", "C1")
def plot_calinski_harabasz(df):  _plot_line(df, "ch",              "calinski_harabasz","d", "C2")

# -------------------------------------------------------------------------- #
# Entry‑point                                                                #
# -------------------------------------------------------------------------- #
def main(layers: int | None = None, display: bool = False) -> None:
    if layers is not None:
        cfg.layers = layers
    cfg.display = display

    # global metrics
    g = build_global_metrics_df(); plot_noise_ratio(g); plot_mean_coherence(g)
    # cluster‑size
    s = build_cluster_size_df();   plot_cluster_size_hist(s)
    # layer‑wise
    w = compute_layer_stats();     plot_n_clusters(w); plot_mean_cluster_size(w); plot_layer_composition(w)
    # quality
    q = build_quality_df();        plot_silhouette(q); plot_davies_bouldin(q); plot_calinski_harabasz(q)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Visualise clustering metrics.")
    p.add_argument("--layers", type=int, default=None, help="Number of GPT‑2 layers (default 12)")
    p.add_argument("--show",   action="store_true",     help="Display plots inline")
    args = p.parse_args()
    main(args.layers, args.show)