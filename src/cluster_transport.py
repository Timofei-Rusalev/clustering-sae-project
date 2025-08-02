# -*- coding: utf-8 -*-
from __future__ import annotations

"""
cluster_transport.py
--------------------------------------------------------------------------------
Optimal-Transport evolution between SAE clusters, made *compatible* with
the SAE‑Match transition pipeline.

Key compatibility invariants (unchanged relative to your original):
  • Leak is computed from the FULL (un-pruned) OT plan in raw masses.
  • Exported N (semantic→semantic) is taken from the FULL raw plan (no renorm,
    no prune on semantics) with integer rounding (np.rint), exactly like SAE‑Match.
  • The noise column is NEVER masked by k‑NN and NEVER pruned.
  • Any post-OT pruning is ONLY for visualization artifacts (edges.csv), not for export.

Outputs (SAE‑Match‑compatible):
  • results/transitions_2step_ot/<tag>/trans_Lℓℓ+1.npz   (sparse N: rows/cols/vals/shape)
  • results/transitions_2step_ot/<tag>/leak_Lℓ.npy        (per-source leak)

Additional OT artifacts for analysis:
  • results/ot_plans/<tag>/layer_ℓ_ℓ+1/{plan.npy, cost.npy, edges.csv, meta.json}

Requirements:
  - POT (python optimal transport): pip install pot
  - Access to decoder weights and text-embeddings (reuses sae_match helpers)
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from functools import lru_cache
from tqdm.std import tqdm

import ot  # POT

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class ClusterOTConfig:
    # Project paths
    project_root: Path = Path(__file__).resolve().parent.parent
    clusters_dir: Path = project_root / "results" / "clusters_all_layers"

    # Where to write: (1) OT artifacts, (2) SAE-compatible transitions
    ot_plan_root:   Path = project_root / "results" / "ot_plans"
    transitions_root_base: Path = project_root / "results" / "transitions_2step_ot"

    # Model specifics
    f_latents: int = 32_768
    num_layers: int = 12

    # Representation & metric
    space:  str = "decoder"          # "decoder" | "text" | "hybrid"
    metric: str = "cos"              # for decoder/hybrid: "cos" | "mse"
    beta_text: float = 0.5           # hybrid weight for text-cost
    noise_cost_shift: float = 0.0    # optional positive shift on noise column in Cost

    # OT solver parameters
    solver: str = "sinkhorn"         # "sinkhorn" | "emd" | "unbalanced_sinkhorn"
    reg: float = 0.05                # Sinkhorn epsilon (scaled by median(cost) unless absolute_reg=True)
    absolute_reg: bool = False
    unbalanced_tau: float = 1.0      # KL marginal relaxation (only for unbalanced)
    normalize_masses: bool = True    # Normalize w, v to sum 1 before OT

    # Pre-cost sparsification (k-NN) to mimic SAE kNN locality (0=off)
    knn_k: int = 0

    # Plan pruning after OT (for visualization only; export is from full plan)
    rel_row_thresh: float = 0.0      # keep edges with F[i,j]/w[i] > this
    abs_flow_thresh: float = 0.0     # absolute threshold on F[i,j]
    top_k_per_row: int = 0           # keep at most k largest flows per row (0 = no cap)

    # Noise handling
    merge_noise_ids: Tuple[int, ...] = (-1, -2)

    # Determinism
    seed: int = 42

    # Performance / precision
    n_jobs: int = 1                  # parallel layer pairs in run_all
    use_float32: bool = False        # compute Cost/Plan in float32 (default False to preserve bitwise results)

    verbose: bool = False

    # ---- Derived tag for output subfolders
    def tag(self) -> str:
        if self.space == "decoder":
            return f"decoder_{self.metric}"
        elif self.space == "text":
            return "text"
        else:
            return f"hybrid_{self.metric}_b{self.beta_text:g}"

    # Paths that depend on tag
    def transitions_root(self) -> Path:
        p = self.transitions_root_base / self.tag()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def ot_run_root(self) -> Path:
        p = self.ot_plan_root / self.tag()
        p.mkdir(parents=True, exist_ok=True)
        return p


cfg = ClusterOTConfig()
np.random.seed(cfg.seed)


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def _load_full_labels(layer: int) -> pd.DataFrame:
    """Load layerXX_full_labels.csv produced by clustering.py."""
    path = cfg.clusters_dir / f"layer{layer:02d}_full_labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    if "cluster" not in df.columns:
        raise ValueError(f"{path} has no 'cluster' column")
    return df


# -----------------------------------------------------------------------------
# Cached heavy tensors (reusing sae_match loaders)
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _get_decoder_np(layer: int) -> np.ndarray:
    """(F, d_model) decoder matrix as CPU numpy float32."""
    from sae_match import _load_decoder as _ld  # returns torch tensor (F, d_model)
    W = _ld(layer).cpu().numpy().astype(np.float32)
    return W

@lru_cache(maxsize=None)
def _get_text_embeddings_np(layer: int) -> np.ndarray:
    """(F, d_text) full text embedding for layer, CPU numpy float32 (L2-normalized in sae_match)."""
    from sae_match import _load_text_embeddings
    E, _no_desc = _load_text_embeddings(layer)  # torch tensors on CPU
    return E.numpy().astype(np.float32)


# -----------------------------------------------------------------------------
# Merge noise and remap ids
# -----------------------------------------------------------------------------

def _merge_noise_and_remap(
    labels: np.ndarray,
    noise_ids: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Optional[int], Dict[int, int]]:
    """
    Merge given noise ids into a single 'noise' bucket (last), remap all ids to [0..K-1].
    Returns:
        new_labels : (F,) int32 remapped labels
        masses     : (K,)  float64 per-cluster mass (counts)
        id_map     : dict old_id -> new_id
        noise_idx  : int | None (index of merged noise), None if no noise present
        noise_counts_raw: per old noise id counts (diagnostic)
    """
    labels = labels.astype(int)
    uniq = np.unique(labels)

    present_noise = [nid for nid in noise_ids if nid in uniq]
    noise_counts_raw = {nid: int((labels == nid).sum()) for nid in present_noise}

    non_noise_ids = sorted([x for x in uniq if x not in noise_ids])
    id_map: Dict[int, int] = {}

    for new_id, old_id in enumerate(non_noise_ids):
        id_map[old_id] = new_id

    noise_idx: Optional[int] = None
    if present_noise:
        noise_idx = len(non_noise_ids)
        for old_noise_id in present_noise:
            id_map[old_noise_id] = noise_idx

    new_labels = np.array([id_map[x] for x in labels], dtype=np.int32)

    K = len(non_noise_ids) + (1 if present_noise else 0)
    masses = np.bincount(new_labels, minlength=K).astype(np.float64)

    return new_labels, masses, id_map, noise_idx, noise_counts_raw


# -----------------------------------------------------------------------------
# Centroids (vectorized)
# -----------------------------------------------------------------------------

def _compute_centroids_from_rows(X: np.ndarray, labels_new: np.ndarray) -> np.ndarray:
    """
    Generic centroid computation: X is (F,d), labels_new are [0..K-1].
    Returns (K,d) float32.
    """
    K = int(labels_new.max()) + 1
    d = X.shape[1]
    mu = np.zeros((K, d), dtype=np.float32)
    # accumulate per-cluster sums
    np.add.at(mu, labels_new, X)

    counts = np.bincount(labels_new, minlength=K).astype(np.float32)
    counts[counts == 0.0] = 1.0
    mu /= counts[:, None]
    return mu

def _compute_decoder_centroids(layer: int, labels_new: np.ndarray) -> np.ndarray:
    W = _get_decoder_np(layer)  # (F,d)
    if W.shape[0] != labels_new.shape[0]:
        raise ValueError(f"Decoder rows ({W.shape[0]}) != labels length ({labels_new.shape[0]})")
    return _compute_centroids_from_rows(W, labels_new)

def _compute_text_centroids(layer: int, labels_new: np.ndarray) -> np.ndarray:
    E = _get_text_embeddings_np(layer)  # (F,d_text)
    if E.shape[0] != labels_new.shape[0]:
        raise ValueError(f"Text embeddings rows ({E.shape[0]}) != labels length ({labels_new.shape[0]})")
    mu = _compute_centroids_from_rows(E, labels_new)
    # L2-normalize (cosine distance usage)
    mu_norm = np.linalg.norm(mu, axis=1, keepdims=True) + 1e-12
    mu = mu / mu_norm
    return mu


# -----------------------------------------------------------------------------
# Cost construction (+ vectorized kNN sparsification)
# -----------------------------------------------------------------------------

def _build_cost(mu_a: np.ndarray, mu_b: np.ndarray, metric: str, out_dtype) -> np.ndarray:
    """
    Cost (K_a, K_b):
      - "cos":  1 - cosine similarity  (mu assumed normalized for text; for decoder we normalize here)
      - "mse":  squared Euclidean distance
    """
    if metric == "cos":
        # normalize both sides (idempotent for text centroids)
        mu_a = mu_a / (np.linalg.norm(mu_a, axis=1, keepdims=True) + 1e-12)
        mu_b = mu_b / (np.linalg.norm(mu_b, axis=1, keepdims=True) + 1e-12)
        C = 1.0 - (mu_a @ mu_b.T)
    elif metric == "mse":
        aa = (mu_a ** 2).sum(axis=1, keepdims=True)
        bb = (mu_b ** 2).sum(axis=1, keepdims=True).T
        C = aa + bb - 2.0 * (mu_a @ mu_b.T)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return C.astype(out_dtype, copy=False)

def _knn_mask_cost(C: np.ndarray, k: int, keep_col: Optional[int]) -> np.ndarray:
    """
    Vectorized row-wise k-NN masking: keep k smallest costs per row, others set to a large penalty.
    The column `keep_col` (noise) is ALWAYS kept (never masked).
    If k<=0, returns C unchanged (except we still ensure noise column is unchanged).
    """
    if k is None or k <= 0:
        return C

    K_a, K_b = C.shape
    # indices of k smallest per row (argpartition along axis=1)
    # NOTE: argpartition is O(K_b), more stable than full sort
    idx_small = np.argpartition(C, kth=min(k, K_b-1), axis=1)[:, :k]

    # Build a boolean mask of "to mask" entries (True where we will replace with penalty)
    mask = np.ones_like(C, dtype=bool)

    # For each row i, set mask[i, idx_small[i]] = False (keep these)
    rows = np.arange(K_a)[:, None]
    mask[rows, idx_small] = False

    # Never mask noise column if provided
    if keep_col is not None and 0 <= keep_col < K_b:
        mask[:, keep_col] = False

    # Row-wise penalty = 10 * row_max (same semantics as the original code)
    row_max = np.nanmax(C, axis=1, keepdims=True)
    # In rare pathological cases (all inf), protect with 1.0
    row_max = np.where(np.isfinite(row_max), row_max, 1.0)

    C_out = C.copy()
    # Broadcast penalty across columns
    C_out[mask] = (10.0 * row_max)[mask]
    return C_out


# -----------------------------------------------------------------------------
# OT solvers
# -----------------------------------------------------------------------------

def _solve_ot(
    w: np.ndarray,
    v: np.ndarray,
    Cost: np.ndarray,
    solver: str,
    reg: float,
    absolute_reg: bool = False,
    unbalanced_tau: float = 1.0,
) -> np.ndarray:
    """
    Return a transport plan F (K_src, K_tgt).  Uses POT.
    For balanced problems with normalized w,v, row sums of F are ~w, col sums ~v.
    """
    if solver == "emd":
        return ot.emd(w, v, Cost)

    elif solver == "sinkhorn":
        eps = reg if absolute_reg else max(1e-12, reg * float(np.median(Cost)))
        return ot.sinkhorn(w, v, Cost, reg=eps)

    elif solver == "unbalanced_sinkhorn":
        eps = max(1e-12, reg * float(np.median(Cost)))
        return ot.unbalanced.sinkhorn_unbalanced(w, v, Cost, reg=eps, reg_m=unbalanced_tau)

    else:
        raise ValueError(f"Unknown solver: {solver}")


# -----------------------------------------------------------------------------
# Plan pruning (vectorized; visualization only; noise column is protected)
# -----------------------------------------------------------------------------

def _prune_mask(F: np.ndarray,
                w: np.ndarray,
                rel_thr: float,
                abs_thr: float,
                topk: int,
                keep_col: Optional[int]) -> np.ndarray:
    """
    Build a boolean mask of kept edges based on thresholds applied to *F* (normalized plan).
    The column `keep_col` (noise) is ALWAYS kept wherever F>0.
    Returns:
        keep : bool array of shape F.shape (True = keep edge; False = drop)
    """
    K_src, K_tgt = F.shape
    keep = np.ones_like(F, dtype=bool)

    # relative threshold wrt row mass (≈ w[i])
    if rel_thr > 0.0:
        row_mass = w[:, None] + 1e-12
        rel = F / row_mass
        keep &= (rel >= rel_thr)

    # absolute threshold on F
    if abs_thr > 0.0:
        keep &= (F >= abs_thr)

    # top-k per row by F value (vectorized)
    if topk and topk > 0 and topk < K_tgt:
        # take indices of topk by descending F per row using argpartition
        idx_topk = np.argpartition(F, kth=K_tgt - topk, axis=1)[:, -topk:]
        keep_topk = np.zeros_like(F, dtype=bool)
        rows = np.arange(K_src)[:, None]
        keep_topk[rows, idx_topk] = True
        # preserve noise col if present and F>0 there
        if keep_col is not None and 0 <= keep_col < K_tgt:
            keep_topk[:, keep_col] |= (F[:, keep_col] > 0)
        keep &= keep_topk

    # noise column must never be dropped where F>0
    if keep_col is not None and 0 <= keep_col < K_tgt:
        keep[:, keep_col] |= (F[:, keep_col] > 0)

    return keep


# -----------------------------------------------------------------------------
# Export in SAE-Match-compatible format (N + leak) from FULL plan
# -----------------------------------------------------------------------------

def _export_transitions_compatible(
    F_raw_full: np.ndarray,
    w_raw: np.ndarray,
    src_noise_idx: Optional[int],
    tgt_noise_idx: Optional[int],
    out_dir: Path,
    layer: int,
) -> None:
    """
    Convert FULL (un-pruned) raw-mass plan into:
      - N (semantic-to-semantic counts) saved as trans_Lℓℓ+1.npz (int32 via np.rint)
      - leak vector saved as leak_Lℓ.npy
    """
    K_src, K_tgt = F_raw_full.shape
    assert w_raw.shape[0] == K_src

    # semantic rows/cols
    src_rows = np.arange(K_src)
    if src_noise_idx is not None and 0 <= src_noise_idx < K_src:
        src_rows = np.delete(src_rows, src_noise_idx)

    tgt_cols = np.arange(K_tgt)
    if tgt_noise_idx is not None and 0 <= tgt_noise_idx < K_tgt:
        tgt_cols = np.delete(tgt_cols, tgt_noise_idx)

    # N from FULL plan (no prune, no renorm)
    N = F_raw_full[np.ix_(src_rows, tgt_cols)]

    # Sparse save with integer rounding (not truncation)
    rows, cols = np.nonzero(N > 0)
    vals = np.rint(N[rows, cols]).astype(np.int32)
    shape = np.array([N.shape[0], N.shape[1]], dtype=np.int32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"trans_L{layer:02d}{layer+1:02d}.npz",
        rows=rows.astype(np.int32),
        cols=cols.astype(np.int32),
        vals=vals,
        shape=shape,
    )

    # leak from FULL plan
    leak = np.zeros(len(src_rows), dtype=np.float32)
    if tgt_noise_idx is not None and 0 <= tgt_noise_idx < K_tgt:
        flow_to_noise = F_raw_full[:, tgt_noise_idx]  # FULL (unpruned)
        w_sem = w_raw[src_rows]
        with np.errstate(divide="ignore", invalid="ignore"):
            leak = np.where(w_sem > 0, flow_to_noise[src_rows] / w_sem, 0.0).astype(np.float32)

    np.save(out_dir / f"leak_L{layer:02d}.npy", leak)


# -----------------------------------------------------------------------------
# One layer pair end-to-end
# -----------------------------------------------------------------------------

def _assert_invariants_for_layer(lab_l: np.ndarray,
                                 lab_lp1: np.ndarray,
                                 F: int) -> None:
    """Sanity checks to avoid silent shape errors."""
    if lab_l.ndim != 1 or lab_lp1.ndim != 1:
        raise ValueError("labels must be 1-D arrays")
    if lab_l.shape[0] != F or lab_lp1.shape[0] != F:
        raise ValueError(f"labels length must be == f_latents={F}, "
                         f"got L={lab_l.shape[0]} and L+1={lab_lp1.shape[0]}")

def run_layer_pair(layer: int, cfg: ClusterOTConfig) -> None:
    """
    OT pipeline for pair (ℓ -> ℓ+1), plus SAE-compatible export.
    """
    assert 0 <= layer < cfg.num_layers - 1, "layer must be in [0..num_layers-2]"

    # 1) load labels
    df_l   = _load_full_labels(layer)
    df_lp1 = _load_full_labels(layer + 1)
    lab_l   = df_l["cluster"].to_numpy(dtype=np.int32)
    lab_lp1 = df_lp1["cluster"].to_numpy(dtype=np.int32)
    _assert_invariants_for_layer(lab_l, lab_lp1, cfg.f_latents)

    # 2) merge noise and remap
    lab_l_new, w_raw, map_l, noise_src_idx, noise_counts_src = _merge_noise_and_remap(lab_l, cfg.merge_noise_ids)
    lab_p_new, v_raw, map_p, noise_tgt_idx, noise_counts_tgt = _merge_noise_and_remap(lab_lp1, cfg.merge_noise_ids)
    if v_raw.size == 0:
        raise RuntimeError(f"No target clusters for layer pair {layer}->{layer+1}")

    # 3) compute centroids
    mu_dec_l = mu_dec_p = mu_txt_l = mu_txt_p = None
    if cfg.space in ("decoder", "hybrid"):
        mu_dec_l = _compute_decoder_centroids(layer,     lab_l_new)
        mu_dec_p = _compute_decoder_centroids(layer + 1, lab_p_new)
    if cfg.space in ("text", "hybrid"):
        mu_txt_l = _compute_text_centroids(layer,     lab_l_new)
        mu_txt_p = _compute_text_centroids(layer + 1, lab_p_new)

    # 4) build cost matrix
    out_dtype = np.float32 if cfg.use_float32 else np.float64
    if cfg.space == "decoder":
        Cost = _build_cost(mu_dec_l, mu_dec_p, cfg.metric, out_dtype)
    elif cfg.space == "text":
        Cost = _build_cost(mu_txt_l, mu_txt_p, "cos", out_dtype)
    else:
        Cost_dec = _build_cost(mu_dec_l, mu_dec_p, "cos", out_dtype)
        Cost_txt = _build_cost(mu_txt_l, mu_txt_p, "cos", out_dtype)
        med_dec = max(1e-8, float(np.median(Cost_dec)))
        med_txt = max(1e-8, float(np.median(Cost_txt)))
        Cost = ((1.0 - cfg.beta_text) * (Cost_dec / med_dec)
                + cfg.beta_text * (Cost_txt / med_txt)).astype(out_dtype, copy=False)

    # 5) optionally shift noise cost
    if noise_tgt_idx is not None and cfg.noise_cost_shift > 0.0:
        Cost[:, noise_tgt_idx] += out_dtype(cfg.noise_cost_shift)

    # 6) apply k-NN mask
    Cost_knn = _knn_mask_cost(Cost, cfg.knn_k, keep_col=noise_tgt_idx)

    # 7) prepare masses
    w = w_raw.astype(out_dtype, copy=True)
    v = v_raw.astype(out_dtype, copy=True)
    if cfg.normalize_masses:
        total_w = w.sum() + out_dtype(1e-12)
        total_v = v.sum() + out_dtype(1e-12)
        w /= total_w
        v /= total_v

    # 8) solve OT
    F_plan = _solve_ot(
        w.astype(np.float64, copy=False),
        v.astype(np.float64, copy=False),
        Cost_knn.astype(np.float64, copy=False),
        solver=cfg.solver,
        reg=cfg.reg,
        absolute_reg=cfg.absolute_reg,
        unbalanced_tau=cfg.unbalanced_tau,
    )

    # 9) recover full raw plan
    F = F_plan.astype(np.float64, copy=False)
    row_sumF = F.sum(axis=1, keepdims=True) + 1e-12
    scale = w_raw[:, None] / row_sumF
    F_raw_full = F * scale

    # 10) export SAE‑Match‑compatible transitions
    trans_root = cfg.transitions_root()
    _export_transitions_compatible(
        F_raw_full=F_raw_full,
        w_raw=w_raw,
        src_noise_idx=noise_src_idx,
        tgt_noise_idx=noise_tgt_idx,
        out_dir=trans_root,
        layer=layer,
    )

    # 11) optional verbose logging
    total_flow_full = float(F_raw_full.sum())
    mean_cost_full  = float((F_raw_full * Cost.astype(np.float64)).sum() / (total_flow_full + 1e-12)) if total_flow_full > 0 else float("nan")
    total_flow_vis  = float(F.sum() * scale.sum())  # same as full for unpruned
    mean_cost_vis   = mean_cost_full  # identical in this export
    if cfg.verbose:
        print(
            f"✓ OT {layer:02d}->{layer+1:02d} [{cfg.tag()}] "
            f"K_src={F_raw_full.shape[0]}, K_tgt={F_raw_full.shape[1]}, "
            f"flow_full={total_flow_full:.1f}, mean_cost_full={mean_cost_full:.4f}"
        )


# -----------------------------------------------------------------------------
# Run all pairs
# -----------------------------------------------------------------------------

def _run_pair_entry(layer_and_cfg: Tuple[int, ClusterOTConfig]) -> None:
    """Helper to allow process-pool execution."""
    layer, cfg_local = layer_and_cfg
    # Reset numpy RNG for determinism per process
    np.random.seed(cfg_local.seed + layer)
    run_layer_pair(layer, cfg_local)

def run_all(cfg: ClusterOTConfig) -> None:
    """
    Run OT across all layer pairs with a single simple ASCII progress bar.
    """
    # prepare layer indices and metric name
    layers = list(range(cfg.num_layers - 1))
    metric_name = "text" if cfg.space == "text" else cfg.metric
    # header for readability
    print(f"\n=== Building transitions for metric = {metric_name} ===")
    # bar description: e.g. "COS layers"
    desc = f"{metric_name.upper()} layers"

    if cfg.n_jobs and cfg.n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        tasks = [(l, cfg) for l in layers]
        with ProcessPoolExecutor(max_workers=cfg.n_jobs) as ex:
            futures = [ex.submit(_run_pair_entry, t) for t in tasks]
            # simple ASCII bar in stdout
            with tqdm(
                total=len(futures),
                desc=desc,
                ascii=True,
                ncols=80,
                file=sys.stdout
            ) as pbar:
                for fut in as_completed(futures):
                    fut.result()  # raise if error
                    pbar.update(1)
    else:
        # sequential with same simple bar
        for l in tqdm(
            layers,
            desc=desc,
            ascii=True,
            ncols=80,
            file=sys.stdout
        ):
            run_layer_pair(l, cfg)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)

def main():
    import argparse

    p = argparse.ArgumentParser(description="Cluster-level OT with SAE-compatible transitions export")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Common args helper
    def add_common_args(ap):
        ap.add_argument("--space", choices=["decoder", "text", "hybrid"], default=cfg.space)
        ap.add_argument("--metric", choices=["cos", "mse"], default=cfg.metric)
        ap.add_argument("--solver", choices=["sinkhorn", "emd", "unbalanced_sinkhorn"], default=cfg.solver)
        ap.add_argument("--reg", type=float, default=cfg.reg)
        ap.add_argument("--absolute-reg", action="store_true")
        ap.add_argument("--beta-text", type=float, default=cfg.beta_text)
        ap.add_argument("--noise-cost-shift", type=float, default=cfg.noise_cost_shift)
        ap.add_argument("--normalize-masses", action="store_true", default=cfg.normalize_masses)
        ap.add_argument("--knn-k", type=int, default=cfg.knn_k)

        ap.add_argument("--rel-row-thresh", type=float, default=cfg.rel_row_thresh)
        ap.add_argument("--abs-flow-thresh", type=float, default=cfg.abs_flow_thresh)
        ap.add_argument("--top-k-per-row", type=int, default=cfg.top_k_per_row)

        ap.add_argument("--ot-plan-root", type=str, default=str(cfg.ot_plan_root))
        ap.add_argument("--transitions-root-base", type=str, default=str(cfg.transitions_root_base))

        ap.add_argument("--n-jobs", type=int, default=cfg.n_jobs)
        ap.add_argument("--use-float32", action="store_true", default=cfg.use_float32)

    # all pairs
    ra = sub.add_parser("run_all", help="Run OT across all layer pairs and export transitions")
    add_common_args(ra)

    # single pair
    sp = sub.add_parser("run_pair", help="Run OT for a single (ℓ -> ℓ+1) and export transitions")
    sp.add_argument("layer", type=int, help="Layer index ℓ (pair is ℓ -> ℓ+1)")
    add_common_args(sp)

    args = p.parse_args()

    # update cfg
    cfg.space = args.space
    cfg.metric = args.metric
    cfg.solver = args.solver
    cfg.reg = args.reg
    cfg.absolute_reg = args.absolute_reg
    cfg.beta_text = args.beta_text
    cfg.noise_cost_shift = args.noise_cost_shift
    cfg.normalize_masses = args.normalize_masses
    cfg.knn_k = args.knn_k

    cfg.rel_row_thresh = args.rel_row_thresh
    cfg.abs_flow_thresh = args.abs_flow_thresh
    cfg.top_k_per_row = args.top_k_per_row

    cfg.ot_plan_root = _as_path(args.ot_plan_root)
    cfg.transitions_root_base = _as_path(args.transitions_root_base)

    cfg.n_jobs = args.n_jobs
    cfg.use_float32 = args.use_float32

    if args.cmd == "run_all":
        run_all(cfg)
    else:
        run_layer_pair(args.layer, cfg)


if __name__ == "__main__":
    main()