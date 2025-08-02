# evolution_of_clusters_sae.py
# -----------------------------------------------------------------------------
# Build inter-layer transition matrices and compute evolution metrics
# for two-stage UMAP-HDBSCAN clusters, using SAE-Match permutations.
# Supports three SAE-Match metrics: cos | mse | text.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch  # only for CUDA check
from tqdm.std import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Resolve project root once, so that all outputs go under PROJECT_ROOT/results
# (src/ is the folder where this file lives → project root is its parent)
# ─────────────────────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SRC_DIR.parent

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EvolConfig:
    """I/O locations and global constants."""
    # where clustering labels are written
    labels_dir:       Path = PROJECT_ROOT / "results" / "labels"
    # where sae_match.py saved its permutations (we'll append /{metric})
    perm_root:        Path = PROJECT_ROOT / "results" / "permutations"
    # where we will dump our transition matrices + leaks (we'll suffix by metric)
    transitions_root: Path = PROJECT_ROOT / "results" / "transitions_2step"

    # constants
    n_layers: int = 12
    F:        int = 32_768

    # runtime defaults
    metric: str = "cos"  # cos | mse | text
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def make_cfg(metric: str) -> EvolConfig:
    """Create a config object that points to metric-specific folders."""
    cfg = EvolConfig()
    cfg.metric = metric
    # transitions_2step[_metric]
    base = "transitions_2step" if metric == "cos" else f"transitions_2step_{metric}"
    cfg.transitions_root = PROJECT_ROOT / "results" / base
    cfg.transitions_root.mkdir(parents=True, exist_ok=True)
    # permutations/{metric}
    cfg.perm_root = cfg.perm_root / metric
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Low-level utils
# ─────────────────────────────────────────────────────────────────────────────
def _load_labels(cfg: EvolConfig, layer: int) -> np.ndarray:
    """
    Load (and cache as .npy) cluster labels for a given layer.
    Falls back to reading CSV from results/clusters_all_layers if .npy is missing.
    """
    npy = cfg.labels_dir / f"cluster_labels_L{layer:02d}.npy"
    if npy.exists():
        return np.load(npy)

    csv_path = PROJECT_ROOT / "results" / "clusters_all_layers" / f"layer{layer:02d}_full_labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find cluster labels CSV at {csv_path}")
    df = pd.read_csv(csv_path)
    if "cluster" not in df.columns:
        raise ValueError(f"{csv_path} has no 'cluster' column")

    lbl = df["cluster"].to_numpy(np.int32)
    cfg.labels_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy, lbl)
    return lbl


def _save_sparse_counts(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    shape: Tuple[int, int],
    out_path: Path,
) -> None:
    """Save a sparse counts matrix in (rows, cols, vals, shape) NPZ format."""
    np.savez_compressed(
        out_path,
        rows=rows.astype(np.int32),
        cols=cols.astype(np.int32),
        vals=vals.astype(np.int32),
        shape=np.asarray(shape, np.int32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core builders
# ─────────────────────────────────────────────────────────────────────────────
def build_transition(cfg: EvolConfig, layer: int) -> Dict[str, float]:
    """
    Build a sparse transition matrix for layer→layer+1 using SAE-Match permutation.
    Returns high-level summary stats for convenience.
    """
    assert 0 <= layer < cfg.n_layers - 1

    fn_counts = cfg.transitions_root / f"trans_L{layer:02d}{layer+1:02d}.npz"
    fn_leak   = cfg.transitions_root / f"leak_L{layer:02d}.npy"

    src_lbl = _load_labels(cfg, layer)
    tgt_lbl = _load_labels(cfg, layer + 1)
    perm    = np.load(cfg.perm_root / f"P_{layer:02d}_{layer+1:02d}.npy")

    # consider only semantic (non-negative) clusters on source side
    sem_mask = src_lbl >= 0
    if not sem_mask.any():
        _save_sparse_counts(np.empty(0), np.empty(0), np.empty(0), (0, 0), fn_counts)
        np.save(fn_leak, np.empty(0, np.float32))
        return dict(Kc=0, Kn=0, purity=math.nan, leak=math.nan)

    src = src_lbl[sem_mask].astype(np.int64)
    tgt = tgt_lbl[perm][sem_mask].astype(np.int64)

    Kc = int(src.max()) + 1
    tgt_good = tgt >= 0
    Kn = int(tgt[tgt_good].max()) + 1 if tgt_good.any() else 0

    if Kn == 0:
        counts = np.zeros((Kc, 0), np.int32)
    else:
        flat   = src[tgt_good] * np.int64(Kn) + tgt[tgt_good]
        counts = np.bincount(flat, minlength=Kc * Kn).reshape(Kc, Kn).astype(np.int32)

    sizes = np.bincount(src.astype(np.int32), minlength=Kc).astype(np.int32)
    leak  = (1.0 - counts.sum(1) / sizes).astype(np.float32)

    rows, cols = np.nonzero(counts)
    vals       = counts[rows, cols]
    _save_sparse_counts(rows, cols, vals, counts.shape, fn_counts)
    np.save(fn_leak, leak)

    purity = float((counts.max(1) / sizes).mean()) if Kc else math.nan
    return dict(Kc=Kc, Kn=Kn, purity=purity, leak=float(leak.mean()))


def build_all_transitions(cfg: EvolConfig) -> None:
    """Build transitions + leak vectors for all neighbouring layer pairs."""
    stats = []
    for ℓ in range(cfg.n_layers - 1):
        stats.append(dict(layer=ℓ, **build_transition(cfg, ℓ)))
    stats_path = cfg.transitions_root / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))


def split_merge_metrics(
    cfg: EvolConfig,
    T_split: float = 0.20,
    T_surv:  float = 0.50,
) -> pd.DataFrame:
    """
    Compute per-layer split, merge, survive and mean_leak metrics
    from the transition files produced by build_all_transitions.
    """
    recs: List[Dict[str, float]] = []

    for ℓ in range(cfg.n_layers - 1):
        trans_fp = cfg.transitions_root / f"trans_L{ℓ:02d}{ℓ+1:02d}.npz"
        leak_fp  = cfg.transitions_root / f"leak_L{ℓ:02d}.npy"

        if not trans_fp.exists():
            recs.append(dict(
                layer=ℓ,
                split_frac=np.nan,
                merge_frac=np.nan,
                survive_frac=np.nan,
                mean_leak=np.nan,
            ))
            continue

        data = np.load(trans_fp)
        vals = data["vals"]
        if vals.size == 0:
            recs.append(dict(
                layer=ℓ,
                split_frac=np.nan,
                merge_frac=np.nan,
                survive_frac=np.nan,
                mean_leak=np.nan,
            ))
            continue

        rows, cols = data["rows"], data["cols"]
        shape = tuple(data["shape"].tolist())

        # dense reconstruction
        counts = np.zeros(shape, dtype=int)
        counts[rows, cols] = vals.astype(int)

        leak_vec = np.load(leak_fp)

        # normalised flows
        sizes = counts.sum(axis=1).astype(float) / (1.0 - leak_vec + 1e-12)
        M = counts / sizes[:, None]

        split_frac   = float(((M >= T_split).sum(axis=1) >= 2).mean())
        merge_frac   = float(((M >= T_split).sum(axis=0) >= 2).mean()) if M.shape[1] > 0 else np.nan
        survive_frac = float((M.max(axis=1) >= T_surv).mean())
        mean_leak    = float(leak_vec.mean())

        recs.append(dict(
            layer=ℓ,
            split_frac=split_frac,
            merge_frac=merge_frac,
            survive_frac=survive_frac,
            mean_leak=mean_leak,
        ))

    return pd.DataFrame(recs)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Cluster-evolution utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build transition matrices + leak vectors")
    b.add_argument(
        "--metric",
        choices=("cos", "mse", "text"),
        default="cos",
        help="Which SAE-Match metric to read (cos, mse, or text)"
    )

    s = sub.add_parser("splitmerge", help="Compute split/merge statistics")
    s.add_argument(
        "--metric",
        choices=("cos", "mse", "text"),
        default="cos",
        help="Which SAE-Match metric to use"
    )
    s.add_argument("--csv",     type=Path, help="Optional CSV output path")
    s.add_argument("--json",    type=Path, help="Optional JSON output path")
    s.add_argument("--t_split", type=float, default=0.20)
    s.add_argument("--t_surv",  type=float, default=0.50)

    args = p.parse_args()
    cfg = make_cfg(args.metric)

    if args.cmd == "build":
        build_all_transitions(cfg)
    elif args.cmd == "splitmerge":
        df = split_merge_metrics(cfg, args.t_split, args.t_surv)
        if args.csv:
            df.to_csv(args.csv, index=False)
            print(f"✓ Stats CSV → {args.csv}")
        if args.json:
            args.json.write_text(df.to_json(orient="records", indent=2))
            print(f"✓ Stats JSON → {args.json}")
        if not args.csv and not args.json:
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()