# metrics_for_cluster_evolution.py
# -----------------------------------------------------------------------------
# Row-entropy (H) and Mutual Information (MI) for cluster evolution
# computed from inter-layer transition counts produced by
# evolution_of_clusters_sae.py (transitions_2step[_metric]/trans_L.. files).
#
# This module:
#   • Loads dense counts N (Kc x Kn) and leak vector for each layer ℓ.
#   • Builds:
#       - T  : row-stochastic (conditional on non-leak), over Kn columns.
#       - M  : sub-stochastic rows (mass that reaches semantic targets).
#       - M~ : augmented row-stochastic [M | leak] over (Kn + 1) columns.
#   • Computes:
#       - H_cond (row entropies of T), NaN for full-leak rows.
#       - H_aug  (row entropies of augmented [M | leak]).
#       - mi_cond = I(A; B | non-leak) with p(A) ∝ non-leak mass per row.
#       - mi_aug  = I(A; B_aug) with p(A) ∝ source sizes, leak as extra column.
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm.auto import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Paths & configuration
# ─────────────────────────────────────────────────────────────────────────────

SRC_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SRC_DIR.parent


@dataclass
class MetricsConfig:
    """I/O locations and global constants for entropy/MI computations."""
    # Where evolution_of_clusters_sae.py saved transitions (we'll append metric)
    transitions_root: Path = PROJECT_ROOT / "results" / "transitions_2step"
    # Number of layers (GPT-2 small uses 12; transitions are 0->1 ... 10->11)
    n_layers: int = 12
    # Runtime defaults
    metric: str = "cos"  # cos | mse | text


def make_cfg(metric: str) -> MetricsConfig:
    """Create a config pointing to metric-specific transitions folder."""
    cfg = MetricsConfig()
    cfg.metric = metric
    base = "transitions_2step" if metric == "cos" else f"transitions_2step_{metric}"
    cfg.transitions_root = PROJECT_ROOT / "results" / base
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Low-level loaders and helpers (now using float32 end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

def _load_dense_counts_and_leak(cfg: MetricsConfig, layer: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Reconstruct dense counts matrix N (Kc x Kn) and leak vector (Kc,)
    from NPZ/NPY files built by evolution_of_clusters_sae.py.
    Both are returned as float32 to reduce memory footprint.
    """
    trans_fp = cfg.transitions_root / f"trans_L{layer:02d}{layer+1:02d}.npz"
    leak_fp  = cfg.transitions_root / f"leak_L{layer:02d}.npy"
    if not trans_fp.exists() or not leak_fp.exists():
        return None, None
    data = np.load(trans_fp)
    rows, cols, vals = data["rows"], data["cols"], data["vals"]
    shape = tuple(int(x) for x in data["shape"].tolist())
    N = np.zeros(shape, dtype=np.float32)
    if vals.size > 0:
        N[rows, cols] = vals.astype(np.float32)
    leak = np.load(leak_fp).astype(np.float32)
    return N, leak


def _row_normalize_safe(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Row-normalize a nonnegative matrix to get row-stochastic rows.
    Returns:
      T          : normalized matrix (rows sum to 1 for valid rows, else zeros),
      row_sum    : row sums of mat,
      valid_mask : boolean mask for rows with positive row_sum.
    """
    row_sum = mat.sum(axis=1, keepdims=True)
    valid = (row_sum.squeeze(1) > 0.0)
    T = np.zeros_like(mat, dtype=np.float32)
    if valid.any():
        T[valid] = mat[valid] / row_sum[valid]
    return T, row_sum.squeeze(1), valid


def _entropy_rows(T: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Row-wise entropy: H_a = -sum_b T_ab log T_ab (0 log 0 := 0).
    For rows that are invalid (e.g., all zeros), returns NaN if valid_mask is provided.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -(np.where(T > 0.0, T * np.log(T), 0.0)).sum(axis=1).astype(np.float32)
    if valid_mask is not None:
        H = H.astype(np.float32)
        H[~valid_mask] = np.nan
    return H


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics per layer
# ─────────────────────────────────────────────────────────────────────────────

def compute_row_entropies_for_layer(cfg: MetricsConfig, layer: int, save_vectors: bool = True) -> Dict[str, np.ndarray]:
    """
    Compute two entropy vectors for a given layer:
      - H_cond: entropy of T (conditional on non-leak), NaN for full-leak rows.
      - H_aug : entropy of augmented row distribution [M | leak] (includes leak as a bin).
    Optionally persist both vectors as .npy files under transitions_root.
    """
    N, leak = _load_dense_counts_and_leak(cfg, layer)
    if N is None or leak is None:
        return dict(H_cond=np.array([], dtype=np.float32), H_aug=np.array([], dtype=np.float32))

    # Non-leak mass per row: row_sum_N = s_a * (1 - leak_a)
    row_sum_N = N.sum(axis=1)  # float32

    # Recover exact sizes s_a where possible; rows with full leak (row_sum_N==0 and leak==1) get size 0
    sizes = np.zeros_like(leak, dtype=np.float32)
    valid_sizes = (1.0 - leak) > 0.0
    sizes[valid_sizes] = row_sum_N[valid_sizes] / (1.0 - leak[valid_sizes])

    # M_ab = N_ab / s_a  (sub-stochastic rows; row-sum equals 1 - leak_a for valid sizes)
    M = np.zeros_like(N, dtype=np.float32)
    mask_sizes = sizes > 0.0
    if mask_sizes.any():
        M[mask_sizes] = N[mask_sizes] / sizes[mask_sizes, None]

    # T: conditional on non-leak (row-stochastic over Kn where row_sum_N > 0)
    T, _, valid_T = _row_normalize_safe(N)

    # Entropies
    H_cond = _entropy_rows(T, valid_mask=valid_T)  # float32
    # Augmented distribution with leak as an extra column → row-stochastic over Kn+1
    leak_col = leak.reshape(-1, 1)
    M_aug = np.concatenate([M, leak_col], axis=1).astype(np.float32)
    H_aug = _entropy_rows(M_aug)  # float32

    if save_vectors:
        np.save(cfg.transitions_root / f"H_cond_L{layer:02d}.npy", H_cond.astype(np.float32))
        np.save(cfg.transitions_root / f"H_aug_L{layer:02d}.npy",  H_aug.astype(np.float32))

    return dict(H_cond=H_cond, H_aug=H_aug)


def compute_mutual_info_for_layer(cfg: MetricsConfig, layer: int) -> Dict[str, float]:
    """
    Compute two MI variants between source clusters A and targets B:
      - mi_cond: I(A;B | non-leak). Uses p(a) ∝ non-leak mass per row and T = row-normalized N.
      - mi_aug : I(A;B_aug) with leak as an extra target column. Uses p(a) ∝ source sizes, M_aug.
    Returns Python floats (double precision) for stability in reporting, but
    internal tensors are float32.
    """
    N, leak = _load_dense_counts_and_leak(cfg, layer)
    if N is None or leak is None:
        return dict(mi_cond=float("nan"), mi_aug=float("nan"))

    # Non-leak mass per row; zero means full leak
    row_sum_N = N.sum(axis=1)  # float32

    # --- Conditional MI on non-leak: p(a) ∝ row_sum_N, T row-stochastic on valid rows
    T, _, valid_T = _row_normalize_safe(N)
    w = np.where(valid_T, row_sum_N, 0.0).astype(np.float32)
    if float(w.sum()) == 0.0 or T.shape[1] == 0:
        mi_cond = float("nan")
    else:
        p_a = (w / w.sum()).astype(np.float32)        # p(A=a | non-leak globally)
        P_ab = (p_a[:, None] * T).astype(np.float32)  # joint over (A,B) | non-leak
        P_b  = P_ab.sum(axis=0, keepdims=True).astype(np.float32)
        P_a  = p_a[:, None].astype(np.float32)
        denom = (P_a * P_b).astype(np.float32)
        nz = P_ab > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            mi_cond_val = (P_ab[nz] * np.log(P_ab[nz] / denom[nz])).sum(dtype=np.float64)
        mi_cond = float(mi_cond_val)

    # --- Augmented MI with leak as extra outcome: p(a) ∝ sizes, M_aug row-stochastic
    sizes = np.zeros_like(leak, dtype=np.float32)
    valid_sizes = (1.0 - leak) > 0.0
    sizes[valid_sizes] = row_sum_N[valid_sizes] / (1.0 - leak[valid_sizes])

    M = np.zeros_like(N, dtype=np.float32)
    mask_sizes = sizes > 0.0
    if mask_sizes.any():
        M[mask_sizes] = N[mask_sizes] / sizes[mask_sizes, None]
    M_aug = np.concatenate([M, leak.reshape(-1, 1)], axis=1).astype(np.float32)

    if float(sizes.sum()) == 0.0:
        mi_aug = float("nan")
    else:
        p_a_aug = (sizes / sizes.sum()).astype(np.float32)         # true source marginal
        P_ab_aug = (p_a_aug[:, None] * M_aug).astype(np.float32)   # joint over (A,B_aug)
        P_b_aug  = P_ab_aug.sum(axis=0, keepdims=True).astype(np.float32)
        P_a_aug  = p_a_aug[:, None].astype(np.float32)
        denom_aug = (P_a_aug * P_b_aug).astype(np.float32)
        nz_aug = P_ab_aug > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            mi_aug_val = (P_ab_aug[nz_aug] * np.log(P_ab_aug[nz_aug] / denom_aug[nz_aug])).sum(dtype=np.float64)
        mi_aug = float(mi_aug_val)

    return dict(mi_cond=mi_cond, mi_aug=mi_aug)


# ─────────────────────────────────────────────────────────────────────────────
# Batch runners
# ─────────────────────────────────────────────────────────────────────────────

def entropy_mi_all_layers(cfg: MetricsConfig, save_vectors: bool = True) -> 'np.ndarray':
    """
    For every layer ℓ, compute H vectors (H_cond/H_aug) and MI scalars (mi_cond/mi_aug).
    Persists H vectors (optional) and writes a JSON summary in transitions_root.
    Returns a NumPy structured array with per-layer summaries (float32 fields).
    """
    recs: List[Dict[str, float]] = []
    for ℓ in range(cfg.n_layers - 1):
        H = compute_row_entropies_for_layer(cfg, ℓ, save_vectors=save_vectors)
        MI = compute_mutual_info_for_layer(cfg, ℓ)

        Hc_mean = float(np.nanmean(H["H_cond"])) if H["H_cond"].size else float("nan")
        Ha_mean = float(np.nanmean(H["H_aug"]))  if H["H_aug"].size  else float("nan")

        recs.append(dict(
            layer=ℓ,
            H_cond_mean=Hc_mean,
            H_aug_mean=Ha_mean,
            mi_cond=MI["mi_cond"],
            mi_aug=MI["mi_aug"],
        ))

    # Convert to structured numpy array (lightweight, no pandas dependency)
    dtype = np.dtype([
        ("layer", np.int32),
        ("H_cond_mean", np.float32),
        ("H_aug_mean",  np.float32),
        ("mi_cond",     np.float32),
        ("mi_aug",      np.float32),
    ])
    arr = np.empty(len(recs), dtype=dtype)
    for i, r in enumerate(recs):
        arr[i] = (int(r["layer"]),
                  np.float32(r["H_cond_mean"]) if np.isfinite(r["H_cond_mean"]) else np.float32(np.nan),
                  np.float32(r["H_aug_mean"])  if np.isfinite(r["H_aug_mean"])  else np.float32(np.nan),
                  np.float32(r["mi_cond"])     if np.isfinite(r["mi_cond"])     else np.float32(np.nan),
                  np.float32(r["mi_aug"])      if np.isfinite(r["mi_aug"])      else np.float32(np.nan))

    (cfg.transitions_root / "entropy_mi_stats.json").write_text(
        json.dumps([{k: (float(v) if isinstance(v, (np.floating,)) else int(v))
                    for k, v in zip(arr.dtype.names, row)} for row in arr], indent=2)
    )
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _print_struct_array(arr: np.ndarray) -> None:
    """Pretty-print a structured NumPy array to stdout."""
    if arr.size == 0:
        print("(empty)")
        return
    # Build header
    names = arr.dtype.names
    rows = []
    for row in arr:
        rows.append([row[n] for n in names])
    # Simple fixed-width formatting
    col_widths = [max(len(n), max(len(f"{v}") for v in col)) for n, col in zip(names, zip(*rows))]
    header = "  ".join(n.ljust(w) for n, w in zip(names, col_widths))
    print(header)
    for r in rows:
        print("  ".join(f"{v}".ljust(w) for v, w in zip(r, col_widths)))


def main():
    p = argparse.ArgumentParser(description="Entropy (H) and Mutual Information (MI) for cluster evolution")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("info", help="Compute row entropies H and mutual information I(A;B)")
    i.add_argument(
        "--metric",
        choices=("cos", "mse", "text"),
        default="cos",
        help="Which SAE-Match metric's transitions to use (cos, mse, or text)"
    )
    i.add_argument("--csv",  type=Path, help="Optional CSV output path for per-layer summaries")
    i.add_argument("--json", type=Path, help="Optional JSON output path for per-layer summaries")
    i.add_argument("--no-save-vectors", action="store_true", help="Do not persist per-layer H vectors")
    i.add_argument("--layer", type=int, help="If provided, compute only for this layer index (0..n_layers-2)")

    args = p.parse_args()
    cfg = make_cfg(args.metric)

    if args.cmd == "info":
        if args.layer is not None:
            ℓ = int(args.layer)
            if not (0 <= ℓ < cfg.n_layers - 1):
                raise ValueError(f"Layer must be in [0, {cfg.n_layers-2}]")
            H = compute_row_entropies_for_layer(cfg, ℓ, save_vectors=(not args.no_save_vectors))
            MI = compute_mutual_info_for_layer(cfg, ℓ)
            rec = dict(
                layer=ℓ,
                H_cond_mean=float(np.nanmean(H["H_cond"])) if H["H_cond"].size else float("nan"),
                H_aug_mean=float(np.nanmean(H["H_aug"]))  if H["H_aug"].size  else float("nan"),
                mi_cond=MI["mi_cond"],
                mi_aug=MI["mi_aug"],
            )
            print(json.dumps(rec, indent=2))
        else:
            arr = entropy_mi_all_layers(cfg, save_vectors=(not args.no_save_vectors))
            # Optional CSV/JSON outputs
            if args.csv:
                # Write CSV without pandas
                names = arr.dtype.names
                lines = [",".join(names)]
                for row in arr:
                    lines.append(",".join(str(row[n]) for n in names))
                args.csv.write_text("\n".join(lines))
                print(f"✓ Entropy/MI CSV → {args.csv}")
            if args.json:
                args.json.write_text(
                    json.dumps([{k: (float(v) if isinstance(v, (np.floating,)) else int(v))
                               for k, v in zip(arr.dtype.names, row)} for row in arr],
                               indent=2)
                )
                print(f"✓ Entropy/MI JSON → {args.json}")
            if not args.csv and not args.json:
                _print_struct_array(arr)


if __name__ == "__main__":
    main()