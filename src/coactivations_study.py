# coactivations_study.py
# -----------------------------------------------------------------------------
# Latent‑level co‑activation analysis for SAE v5‑32k
#
# 1.  Stream‑tokenise a text corpus and store a fixed‑length **sub‑sampled**
#     token sequence (every 5‑th token ⇒ weaker local autocorrelation).
# 2.  For each GPT‑2 layer:
#       • build a sparse activation matrix A  (N_eff × F, CSR‑float32)
#       • derive per‑feature counts  n_f  and co‑activation matrix  C = AᵀA.
# 3.  Compute robust “relative” Jaccard & NPMI scores that quantify how
#     coherently each semantic cluster fires vs. a random background.
#
# Heavy outputs (*.npy, *.npz, *.csv) live under PROJECT_ROOT/results/
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import math
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from datasets import load_dataset
from tqdm.std import tqdm
import sys
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
TQDM_OPTS = dict(ascii=True, ncols=80, file=sys.stdout)
from scipy.stats import trim_mean

# ────────────────────────────────────────────────────────────────────────────
# Resolve paths relative to project root
# ────────────────────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_RESULTS = PROJECT_ROOT / "results"
DEFAULT_COACT = DEFAULT_RESULTS / "coactivations"
DEFAULT_CHECKPOINTS = PROJECT_ROOT / "checkpoints"
DEFAULT_CLUSTERS = DEFAULT_RESULTS / "clusters_all_layers"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class CoActConfig:
    # path to save co-activation matrices and counts
    results_dir: Path = DEFAULT_COACT
    # path to save sub-sampled token sequence
    token_file: Path = DEFAULT_RESULTS / "tokens_seq.npy"

    # directory with pretrained SAE checkpoints
    checkpoints_dir: Path = DEFAULT_CHECKPOINTS
    # directory with cluster CSV files
    clusters_dir: Path = DEFAULT_CLUSTERS

    # token stream settings
    ctx_len: int = 512
    batch_win: int = 4
    tokens_n: int = 3_000_000
    stride: int = 5
    dataset: str = "Bingsu/openwebtext_20p"
    subset: Optional[str] = None

    # compute device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# instantiate config and create the coactivations folder
cfg = CoActConfig()
cfg.results_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1) Token sequence collection  (sub‑sampling stride = cfg.stride)
# ---------------------------------------------------------------------------

def collect_and_save_tokens_seq(
    model,
    tokens_n: int        = cfg.tokens_n,
    ctx_len:  int        = cfg.ctx_len,
    dataset:  str        = cfg.dataset,
    subset:   Optional[str] = cfg.subset,
    stride:   int        = cfg.stride,
    save_to:  Path       = cfg.token_file,
) -> np.ndarray:
    """
    Stream a dataset, accumulate ``tokens_n + ctx_len`` raw tokens,
    then keep every `stride`‑th token (default = 5), save to *.npy
    and return the NumPy array (int64).
    """
    
    ds = load_dataset(
        dataset,
        split="train",
        streaming=True
    )

    buf, total = [], 0
    for ex in tqdm(ds, desc="Tokenising corpus", mininterval=1.0, **TQDM_OPTS):
        text = ex["text"].strip()
        if not text:
            continue
        toks = model.to_tokens(text, prepend_bos=False).squeeze(0)
        buf.append(toks)
        total += toks.numel()
        if total >= tokens_n + ctx_len:
            break

    all_tokens = torch.cat(buf)[: tokens_n + ctx_len]
    tokens_seq = all_tokens.cpu().numpy().astype(np.int64, copy=False)

    # sub‑sample every `stride`‑th token
    if stride > 1:
        tokens_seq = tokens_seq[::stride]

    # ensure directory exists and save
    save_to.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_to, tokens_seq)

    # minimal console feedback (no full path)
    tqdm.write(f"✓ Saved token sequence ({tokens_seq.size:,} tokens, stride={stride})")
    return tokens_seq

# ---------------------------------------------------------------------------
# 2) Build sparse activation matrix A  per layer
# ---------------------------------------------------------------------------
def build_A_for_layer(
    layer_idx: int,
    sae,                       # SingleSAEv5 instance
    model,                     # transformer‑lens model
    tokens_seq: np.ndarray,
    ctx_len: int = cfg.ctx_len,
    batch_wins: int = cfg.batch_win,
    device: str = cfg.device,
    out_dir: Path = cfg.results_dir,
    binarize: bool = True,     # <— NEW: default True
) -> sp.csr_matrix:
    """
    Return CSR activation matrix A and save it to A_layer{ℓ}.npz.
    Each row corresponds to a token; columns are SAE latents (F).
    If binarize=True, all non‑zero entries are set to 1 (top‑k indicator),
    which is the right thing to do for Jaccard / PMI-style co-activation stats.
    """
    F = sae.F
    step = ctx_len * batch_wins
    N_eff = (len(tokens_seq) // ctx_len) * ctx_len

    rows_all, cols_all, data_all = [], [], []
    hook = f"blocks.{layer_idx}.hook_resid_post"

    with torch.no_grad():
        for global_start in range(0, N_eff, step):
            wins = min(batch_wins, (N_eff - global_start) // ctx_len)
            idx = global_start + np.arange(wins * ctx_len).reshape(wins, ctx_len)
            toks = torch.as_tensor(tokens_seq[idx], dtype=torch.long, device=device)

            resid = model.run_with_cache(toks, names_filter=[hook])[1][hook]
            resid = resid.reshape(-1, resid.size(-1))  # (N, d)

            # inline SAE.encode (faster)
            if sae.normalize:
                x_norm, _, _ = sae._layer_norm(resid)
            else:
                x_norm = resid
            h_pre = torch.nn.functional.linear(
                x_norm - sae.pre_bias, sae.encoder.weight, sae.latent_bias
            )
            topv, topi = torch.topk(h_pre, sae.k, dim=-1)
            topv = torch.clamp(topv, min=0)

            N = topv.size(0)
            r = torch.arange(N, device=device).unsqueeze(1).expand(-1, sae.k)
            rows_all.append((global_start + r.reshape(-1)).cpu().numpy())
            cols_all.append(topi.reshape(-1).cpu().numpy())

            if binarize:
                # write 1s instead of real activation magnitudes
                data_all.append(np.ones(topv.numel(), dtype=np.float32))
            else:
                data_all.append(topv.reshape(-1).cpu().float().numpy())

    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    data = np.concatenate(data_all)

    A = sp.csr_matrix((data, (rows, cols)), shape=(N_eff, F), dtype=np.float32)
    out_f = out_dir / f"A_layer{layer_idx}.npz"
    sp.save_npz(out_f, A, compressed=True)

    return A

# ---------------------------------------------------------------------------
# 3) Derive  n_f  and co‑activation matrix  C = AᵀA
# ---------------------------------------------------------------------------
def build_n_and_C(
    A: sp.csr_matrix,
    layer_idx: int,
    out_dir: Path = cfg.results_dir,
) -> Tuple[np.ndarray, sp.coo_matrix]:
    """
    From sparse A produce:
      • n_f – per‑feature activation counts (int64, shape F)
      • C   – co‑activation counts (COO, int32)
    Both are stored under  results/.
    """
    n_f = np.asarray(A.sum(axis=0)).ravel().astype(np.int64)
    np.save(out_dir / f"n_layer{layer_idx}.npy", n_f)

    C = (A.T @ A).tocoo().astype(np.int32)
    sp.save_npz(out_dir / f"C_layer{layer_idx}.npz", C, compressed=True)
    return n_f, C

# ---------------------------------------------------------------------------
# 4) Robust Relative Jaccard
# ---------------------------------------------------------------------------

def _pair_nij(i: int, j: int, C_csr: sp.csr_matrix) -> int:
    """Fast lookup of N_ij in a CSR row."""
    start, end = C_csr.indptr[i], C_csr.indptr[i + 1]
    cols, data = C_csr.indices[start:end], C_csr.data[start:end]
    pos = np.searchsorted(cols, j)
    return int(data[pos]) if pos < len(cols) and cols[pos] == j else 0


def _avg_jaccard(
    cluster: List[int],
    n_f: np.ndarray,
    C_csr: sp.csr_matrix,
) -> float:
    """Average pair‑wise Jaccard inside `cluster`."""
    k = len(cluster)
    if k < 2:
        return math.nan
    s = cnt = 0
    for a in range(k):
        i = cluster[a]
        for b in range(a + 1, k):
            j = cluster[b]
            nij = _pair_nij(i, j, C_csr)
            denom = n_f[i] + n_f[j] - nij
            if denom:
                s += nij / denom
            cnt += 1
    return s / cnt if cnt else math.nan


def _bg_stats(
    size: int,
    n_f: np.ndarray,
    C_csr: sp.csr_matrix,
    F: int,
    repeats: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Background mean / std of Jaccard for random sets of given size."""
    vals = []
    for _ in range(repeats):
        samp = rng.choice(F, size, replace=False).tolist()
        vals.append(_avg_jaccard(samp, n_f, C_csr))
    return float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1))


# ---------- main routine ----------------------------------------------------
def rel_jaccard_for_layer_robust(
    layer: int,
    *,
    root: Path = cfg.results_dir,
    clusters_dir: Path = cfg.clusters_dir,
    repeats: int = 50,
    min_size: int = 4,
    max_ratio_clip: float = 150.0,
    trim_pct: float = 0.10,
    seed: int = 42,
    save_csv: bool = True,
    csv_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Robust “relative” Jaccard for all clusters of a layer.
    If save_csv=True, saves the DataFrame as CSV under `root` using
    `csv_name` or default "jaccard_layer{layer}.csv".

    Returns
    -------
    df : pd.DataFrame
      One row per cluster with columns
      [cluster, size, jacc, bg_mean, bg_std, ratio_raw, ratio, z].
    layer_stats : dict
      Aggregate layer‑level stats:
        {'median_ratio', 'trimmed_mean_ratio', 'n_clusters'}
    """
    rng = np.random.default_rng(seed)

    # load counts and co‑activation matrix
    n_f   = np.load(root / f"n_layer{layer}.npy")
    C_csr = sp.load_npz(root / f"C_layer{layer}.npz").tocsr()
    F     = C_csr.shape[0]

    # load and filter clusters
    df_clusters = (
        pd.read_csv(clusters_dir / f"layer{layer:02d}_clusters.csv")
          .query("cluster != -1")
          .groupby("cluster")["original_index"]
          .apply(list)
          .reset_index()
          .rename(columns={"original_index": "members"})
    )
    df_clusters = df_clusters[df_clusters["members"].str.len() >= min_size]
    if df_clusters.empty:
        empty_cols = ["cluster","size","jacc","bg_mean","bg_std","ratio_raw","ratio","z"]
        df_empty = pd.DataFrame(columns=empty_cols)
        stats_empty = {"median_ratio": math.nan,
                       "trimmed_mean_ratio": math.nan,
                       "n_clusters": 0}
        if save_csv:
            name = csv_name or f"jaccard_layer{layer}.csv"
            df_empty.to_csv(root / name, index=False)
        return df_empty, stats_empty

    # precompute background Jaccard stats
    bg_cache: Dict[int, Tuple[float, float]] = {}
    for sz in df_clusters["members"].str.len().unique():
        bg_cache[sz] = _bg_stats(sz, n_f, C_csr, F, repeats, rng)

    # compute metrics per cluster
    records: List[Dict[str, float]] = []
    for _, row in df_clusters.iterrows():
        cid   = int(row["cluster"])
        membs = row["members"]
        size  = len(membs)

        jacc      = _avg_jaccard(membs, n_f, C_csr)
        mu_bg, sd = bg_cache[size]
        ratio_raw = jacc / mu_bg if mu_bg else math.nan
        ratio     = min(ratio_raw, max_ratio_clip) if not math.isnan(ratio_raw) else math.nan
        z         = (jacc - mu_bg) / sd if sd else math.nan

        records.append({
            "cluster": cid,
            "size": size,
            "jacc": jacc,
            "bg_mean": mu_bg,
            "bg_std": sd,
            "ratio_raw": ratio_raw,
            "ratio": ratio,
            "z": z,
        })

    df = pd.DataFrame(records)

    # aggregate layer stats
    median_ratio       = float(df["ratio"].median(skipna=True))
    trimmed_mean_ratio = float(trim_mean(df["ratio"].values, proportiontocut=trim_pct))
    layer_stats = {
        "median_ratio": median_ratio,
        "trimmed_mean_ratio": trimmed_mean_ratio,
        "n_clusters": len(df),
    }

    # auto-save to CSV
    if save_csv:
        name = csv_name or f"jaccard_layer{layer}.csv"
        df.to_csv(root / name, index=False)

    return df, layer_stats

# ---------------------------------------------------------------------------
# 5) Robust Relative NPMI
# ---------------------------------------------------------------------------


def _avg_npmi(
    cluster: List[int],
    n_f: np.ndarray,
    C_csr: sp.csr_matrix,
    N: int,
) -> float:
    """
    Average pair‑wise Normalised PMI inside `cluster`.
    N — total number of tokens (rows of A).
    """
    k = len(cluster)
    if k < 2:
        return math.nan
    s = cnt = 0
    for a in range(k):
        i = cluster[a]
        p_i = n_f[i] / N
        if p_i == 0:
            continue
        for b in range(a + 1, k):
            j = cluster[b]
            C_ij = _pair_nij(i, j, C_csr)
            if C_ij == 0:
                continue
            p_j  = n_f[j] / N
            p_ij = C_ij / N
            pmi  = math.log(p_ij / (p_i * p_j))
            npmi = pmi / (-math.log(p_ij))
            s += npmi
            cnt += 1
    return s / cnt if cnt else math.nan


def _bg_stats_npmi(
    size: int,
    n_f: np.ndarray,
    C_csr: sp.csr_matrix,
    N: int,
    F: int,
    repeats: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Background mean/std of NPMI for random sets of given size."""
    vals = []
    for _ in range(repeats):
        samp = rng.choice(F, size, replace=False).tolist()
        vals.append(_avg_npmi(samp, n_f, C_csr, N))
    return float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1))


# ---------- main routine ----------------------------------------------------
def rel_npmi_for_layer_robust(
    layer: int,
    *,
    root: Path = cfg.results_dir,
    clusters_dir: Path = cfg.clusters_dir,
    repeats: int = 50,
    min_size: int = 4,
    max_ratio_clip: float = 25.0,
    trim_pct: float = 0.10,
    seed: int = 42,
    save_csv: bool = True,
    csv_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Robust “relative” NPMI for all semantic clusters of a layer.
    If save_csv=True, saves the DataFrame as CSV under `root` using
    `csv_name` or default "npmi_layer{layer}.csv".

    Returns
    -------
    df : pd.DataFrame
        One row per cluster with columns
        [cluster, size, npmi, bg_mean, bg_std, ratio_raw, ratio, z].
    layer_stats : dict
        Aggregate layer‑level stats:
          {'median_ratio', 'trimmed_mean_ratio', 'n_clusters'}
    """
    rng = np.random.default_rng(seed)

    # load per-feature counts and co-activation matrix
    n_f   = np.load(root / f"n_layer{layer}.npy")
    C_csr = sp.load_npz(root / f"C_layer{layer}.npz").tocsr()
    F     = C_csr.shape[0]

    # load activation matrix to get total token count for PMI denominator
    A_path = root / f"A_layer{layer}.npz"
    if not A_path.exists():
        raise FileNotFoundError(f"Need {A_path} to determine token count for NPMI.")
    N = sp.load_npz(A_path).shape[0]

    # load and filter clusters
    df_clusters = (
        pd.read_csv(clusters_dir / f"layer{layer:02d}_clusters.csv")
          .query("cluster != -1")
          .groupby("cluster")["original_index"]
          .apply(list)
          .reset_index()
          .rename(columns={"original_index": "members"})
    )
    df_clusters = df_clusters[df_clusters["members"].str.len() >= min_size]
    if df_clusters.empty:
        empty_cols = ["cluster","size","npmi","bg_mean","bg_std","ratio_raw","ratio","z"]
        df_empty = pd.DataFrame(columns=empty_cols)
        stats_empty = {"median_ratio": math.nan,
                       "trimmed_mean_ratio": math.nan,
                       "n_clusters": 0}
        if save_csv:
            name = csv_name or f"npmi_layer{layer}.csv"
            df_empty.to_csv(root / name, index=False)
        return df_empty, stats_empty

    # precompute background NPMI stats for each cluster size
    bg_cache: Dict[int, Tuple[float, float]] = {}
    for sz in df_clusters["members"].str.len().unique():
        bg_cache[sz] = _bg_stats_npmi(sz, n_f, C_csr, N, F, repeats, rng)

    # compute metrics for each cluster
    records: List[Dict[str, float]] = []
    for _, row in df_clusters.iterrows():
        cid   = int(row["cluster"])
        membs = row["members"]
        size  = len(membs)

        npmi      = _avg_npmi(membs, n_f, C_csr, N)
        mu_bg, sd = bg_cache[size]
        ratio_raw = npmi / mu_bg if mu_bg else math.nan
        ratio     = min(ratio_raw, max_ratio_clip) if not math.isnan(ratio_raw) else math.nan
        z         = (npmi - mu_bg) / sd if sd else math.nan

        records.append({
            "cluster": cid,
            "size": size,
            "npmi": npmi,
            "bg_mean": mu_bg,
            "bg_std": sd,
            "ratio_raw": ratio_raw,
            "ratio": ratio,
            "z": z,
        })

    df = pd.DataFrame(records)

    # aggregate layer-level stats
    median_ratio       = float(df["ratio"].median(skipna=True))
    trimmed_mean_ratio = float(trim_mean(df["ratio"].values, proportiontocut=trim_pct))
    layer_stats = {
        "median_ratio": median_ratio,
        "trimmed_mean_ratio": trimmed_mean_ratio,
        "n_clusters": len(df),
    }

    # auto-save to CSV
    if save_csv:
        name = csv_name or f"npmi_layer{layer}.csv"
        df.to_csv(root / name, index=False)

    return df, layer_stats

# ---------------------------------------------------------------------------
# __main__  –  sample end‑to‑end pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Example CLI:
    -----------
    python -m coactivations_study tokens
    python -m coactivations_study build-all
    python -m coactivations_study jaccard --layer 6 [--csv-out path]
    python -m coactivations_study npmi    --layer 6 [--csv-out path]
    """
    import argparse
    import transformer_lens as tl
    from sae_v5_32k import load_sae_layer

    parser = argparse.ArgumentParser(description="SAE co‑activation pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 1) Token collection
    p_tokens = sub.add_parser("tokens", help="Collect (and sub‑sample) token sequence")
    p_tokens.add_argument("--force", action="store_true", help="Overwrite existing tokens.npy")

    # 2) Build A, n_f, C for all layers
    p_build = sub.add_parser("build-all", help="Build A, n_f, C for all layers")

    # 3) Compute relative Jaccard for one layer
    p_jac = sub.add_parser("jaccard", help="Compute robust relative Jaccard for a layer")
    p_jac.add_argument("--layer",  type=int,            required=True, help="Layer index 0–11")
    p_jac.add_argument("--csv-out", type=Path,          help="Optional path to save the CSV")

    # 4) Compute relative NPMI for one layer
    p_npmi = sub.add_parser("npmi", help="Compute robust relative NPMI for a layer")
    p_npmi.add_argument("--layer",  type=int,            required=True, help="Layer index 0–11")
    p_npmi.add_argument("--csv-out", type=Path,          help="Optional path to save the CSV")

    args = parser.parse_args()

    # Step 1: tokens (once)
    if args.cmd in ("tokens", "build-all"):
        if cfg.token_file.exists() and not getattr(args, "force", False):
            tokens_seq = np.load(cfg.token_file)
            print(f"Using cached token sequence ({tokens_seq.size:,} tokens, stride={cfg.stride})")
        else:
            gpt2 = tl.HookedTransformer.from_pretrained(
                "gpt2-small", center_writing_weights=False, device=cfg.device
            ).eval()
            tokens_seq = collect_and_save_tokens_seq(gpt2)
        if args.cmd == "tokens":
            return
    else:
        tokens_seq = np.load(cfg.token_file)

    # Step 2: build A, n_f, C
    if args.cmd == "build-all":
        gpt2 = tl.HookedTransformer.from_pretrained(
            "gpt2-small", center_writing_weights=False, device=cfg.device
        ).eval()
        for ℓ in range(12):
            sae = load_sae_layer(cfg.checkpoints_dir / f"layer{ℓ}.pt", device=cfg.device)
            A = build_A_for_layer(ℓ, sae, gpt2, tokens_seq)
            build_n_and_C(A, ℓ)
        return

    # Step 3: jaccard
    if args.cmd == "jaccard":
        df, stats = rel_jaccard_for_layer_robust(
            args.layer,
            root=cfg.results_dir,
            clusters_dir=cfg.clusters_dir
        )
        if args.csv_out:
            args.csv_out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.csv_out, index=False)
            print(f"✓ Saved Jaccard CSV → {args.csv_out}")
        else:
            print(df.head(20).to_string(index=False))
            print(f"\nLayer stats: {stats}")
        return

    # Step 4: npmi
    if args.cmd == "npmi":
        df, stats = rel_npmi_for_layer_robust(
            args.layer,
            root=cfg.results_dir,
            clusters_dir=cfg.clusters_dir
        )
        if args.csv_out:
            args.csv_out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.csv_out, index=False)
            print(f"✓ Saved NPMI CSV → {args.csv_out}")
        else:
            print(df.head(20).to_string(index=False))
            print(f"\nLayer stats: {stats}")
        return


if __name__ == "__main__":
    main()