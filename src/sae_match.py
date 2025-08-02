# sae_match.py
# -----------------------------------------------------------------------------
# Sparse SAE‑to‑SAE matching + explained‑variance diagnostics
#
# 1.  **Sparse assignment (k‑NN + MinCostFlow)**
#     Creates an exact Hungarian matching between latent spaces of
#     neighbouring SAE layers (l -> l+1) using only the k nearest‑neighbour
#     edges (default k = 256).  Complexity:  O(F·k)  memory instead of O(F²).
#
#     Output:  permutations/<version>/<metric>/P_l_l+1.npy
#
# 2.  **Explained‑Variance (EV) sweep**
#     Quantifies how well the permuted latents of layer l predict the residual
#     stream of layer l+1 on held‑out text windows.
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
import hashlib
import pandas as pd

import numpy as np
import torch
import torch.nn.functional as F
from ortools.graph.python import min_cost_flow
from tqdm.std import tqdm, trange
from typing import Dict
import transformer_lens as tl

# ─────────────────────────────────────────────────────────────────────────────
# Resolve project root and default paths
# ─────────────────────────────────────────────────────────────────────────────

SRC_DIR             = Path(__file__).parent.resolve()
PROJECT_ROOT        = SRC_DIR.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_PERM_ROOT      = DEFAULT_RESULTS_DIR / "permutations"
DEFAULT_TOKENS_PATH    = DEFAULT_RESULTS_DIR / "tokens_seq.npy"

# where clustering.py writes its outputs
DEFAULT_CLUSTERS_DIR = DEFAULT_RESULTS_DIR / "clusters_all_layers"
DEFAULT_LABELS_DIR   = DEFAULT_RESULTS_DIR / "labels"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchConfig:
    # ── Input data ──────────────────────────────────────────────────────────
    checkpoints_dir: Path = DEFAULT_CHECKPOINT_DIR
    tokens_path:     Path = DEFAULT_TOKENS_PATH

    # ── Where to write permutations (cos, mse, text) ───────────────────────
    perm_root:       Path = DEFAULT_PERM_ROOT

    # ── Clustering outputs from clustering.py ─────────────────────────────
    clusters_dir:    Path = DEFAULT_CLUSTERS_DIR
    labels_dir:      Path = DEFAULT_LABELS_DIR

    # ── Model architecture ─────────────────────────────────────────────────
    f_latents:  int = 32_768   # number of SAE latents
    num_layers: int = 12       # GPT‑2 small

    # ── k‑NN assignment ─────────────────────────────────────────────────────
    K: int     = 256           # neighbors per row
    block: int = 512           # search block size

    # ── Supported metrics ──────────────────────────────────────────────────
    metric_choices: Tuple[str, ...] = ("cos", "mse", "text")

    # ── Jitter & “no‐description” penalty (text metric) ──────────────────
    sigma_jitter:   float = 1e-5
    lambda_no_desc: float = 0.3

    # ── Devices ────────────────────────────────────────────────────────────
    device_match: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # ── EV sweep parameters ────────────────────────────────────────────────
    ctx_len:   int = 512
    windows_k: int = 3_000
    batch_gpu: int = 4
    seed:      int = 42

# instantiate
cfg = MatchConfig()

# ─────────────────────────────────────────────────────────────────────────────
# Global determinism (numpy + torch)
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ─────────────────────────────────────────────────────────────────────────────
# Ensure all needed directories exist
# ─────────────────────────────────────────────────────────────────────────────
cfg.perm_root   .mkdir(parents=True, exist_ok=True)
cfg.tokens_path.parent.mkdir(parents=True, exist_ok=True)
cfg.clusters_dir.mkdir(parents=True, exist_ok=True)
cfg.labels_dir  .mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Low‑level helpers
# ─────────────────────────────────────────────────────────────────────────────
def _to_np(x: torch.Tensor) -> np.ndarray:
    """Detach tensor and move to CPU numpy array safely."""
    x = x.detach().cpu()
    try:
        return x.numpy()
    except RuntimeError:
        return np.asarray(x.tolist())

def _load_decoder(layer: int) -> torch.Tensor:
    """Load SAE decoder matrix for a given layer. Returns (F, d_model) float32."""
    from sae_v5_32k import load_sae_layer  # delayed import
    sae = load_sae_layer(cfg.checkpoints_dir / f"layer{layer}.pt", device="cpu")
    return sae.decoder.weight.T.contiguous().float()  # (F, d_model)

def _knn_edges(Wa: torch.Tensor, Wb: torch.Tensor, metric: str):
    """Build sparse k-NN edges between Wa and Wb (rows are Wa indices)."""
    Wa, Wb = Wa.to(cfg.device_match), Wb.to(cfg.device_match)
    if metric == "cos":
        Wa, Wb = F.normalize(Wa, dim=1), F.normalize(Wb, dim=1)

    rows, cols, vals = [], [], []
    n = Wa.size(0)

    for i0 in trange(0, n, cfg.block, desc=f"k‑NN {metric}", disable=True, ascii=True, ncols=80, file=sys.stdout, leave=False):
        i1 = min(i0 + cfg.block, n)
        blk = Wa[i0:i1]
        if metric == "cos":
            # cosine distance = 1 - cosine_similarity
            dist = 1 - blk @ Wb.T
        else:
            # squared Euclidean distance
            dist = (blk.pow(2).sum(1, keepdim=True)
                    + Wb.pow(2).sum(1) - 2 * blk @ Wb.T)

        v, j = torch.topk(dist, cfg.K, largest=False)
        rows.append(torch.arange(i0, i1, device=cfg.device_match)
                    .unsqueeze(1).repeat(1, cfg.K).flatten())
        cols.append(j.flatten())
        vals.append(v.flatten())

    return torch.cat(rows), torch.cat(cols), torch.cat(vals)

def sae_match_sparse(layer_a: int, layer_b: int, metric: str):
    """
    Compute sparse SAE-to-SAE matching between decoder columns of
    layer_a and layer_b via k-NN + MinCostFlow, parameterized by actual F.
    Returns:
        perm : np.ndarray of shape (F,), the matching permutation
        mc   : float, mean cosine between matched decoder vectors (sanity-check)
    """
    # 1) Load decoder matrices (F, d_model)
    Wa_cpu = _load_decoder(layer_a)
    Wb_cpu = _load_decoder(layer_b)
    assert Wa_cpu.shape[0] == Wb_cpu.shape[0], "Mismatch in number of latents"
    n_latents = Wa_cpu.shape[0]

    # 2) Build sparse k-NN graph in chosen metric (функция сама перенесёт на device)
    rows, cols, vals = _knn_edges(Wa_cpu, Wb_cpu, metric)

    # 3) Solve exact assignment using the F-parameterized solver
    perm = _solve_sparse_assignment_F(rows, cols, vals, metric, n_latents)

    # 4) Sanity-check: sample up to 1000 indices and compute mean cosine
    device = cfg.device_match
    Wa = Wa_cpu.to(device)
    Wb = Wb_cpu.to(device)

    rng = np.random.default_rng(cfg.seed + 1000 * layer_a + layer_b)
    sample_size = min(1_000, n_latents)
    idx_np = rng.choice(n_latents, sample_size, replace=False)
    idx = torch.from_numpy(idx_np).to(device)

    Wa_n = F.normalize(Wa[idx], dim=1)
    perm_t = torch.from_numpy(perm).to(device)
    Wb_n = F.normalize(Wb[perm_t[idx]], dim=1)

    mc = (Wa_n * Wb_n).sum(1).mean().item()

    return perm, mc

# ─────────────────────────────────────────────────────────────────────────────
# NEW: helper to detect duplicates and add tiny jitter
# ─────────────────────────────────────────────────────────────────────────────
def _jitter_duplicates(E: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """
    Add small Gaussian noise to exact-duplicate rows to break ties.
    Assumes E is (F, d) float32. Returns a *copy* with jitter applied.
    """
    rng = np.random.default_rng(seed)
    # We detect exact duplicates using np.unique on rows.
    # NOTE: this is OK if your embeddings for duplicates are bitwise-equal.
    # If not, you'll need a more tolerant hashing / bucketing.
    uniq, inv, counts = np.unique(E, axis=0, return_inverse=True, return_counts=True)
    if counts.max() == 1:
        return E  # no duplicates at all

    E_out = E.copy()
    for gid, cnt in enumerate(counts):
        if cnt <= 1:
            continue
        idx = np.where(inv == gid)[0]
        noise = rng.normal(0.0, sigma, size=(cnt, E.shape[1])).astype(np.float32)
        E_out[idx] += noise
    return E_out


# ─────────────────────────────────────────────────────────────────────────────
# NEW: load text embeddings for a given layer
# ─────────────────────────────────────────────────────────────────────────────
def _load_text_embeddings(layer: int) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """
    Build a full (F, d_text) embedding matrix for `layer` from the *unique*
    embeddings produced by `clustering.py`, by broadcasting them to all F
    latents using the `rep_idx` mapping stored in
    results/clusters_all_layers/layerXX_full_labels.csv.

    Returns:
        E_torch      : (F, d_text) float32 tensor on CPU
        no_desc_mask : (F,) bool tensor (True if that latent had no description)
    """

    F = cfg.f_latents

    # ---- Paths we need ----
    clu_dir = cfg.clusters_dir
    # files produced by clustering.py / save_results(...)
    df_unique_path = clu_dir / f"layer{layer:02d}_clusters.csv"
    df_full_path   = clu_dir / f"layer{layer:02d}_full_labels.csv"

    if not df_unique_path.exists():
        raise FileNotFoundError(f"Unique clusters CSV not found: {df_unique_path}")
    if not df_full_path.exists():
        raise FileNotFoundError(f"Full labels CSV not found: {df_full_path}")

    # ---- Load dataframes ----
    df_unique = pd.read_csv(df_unique_path)
    df_full   = pd.read_csv(df_full_path)

    # Sanity checks
    if "processed_text" not in df_unique.columns or "original_index" not in df_unique.columns:
        raise ValueError(f"{df_unique_path} does not contain required columns "
                         f"['processed_text', 'original_index'].")
    if "rep_idx" not in df_full.columns or "processed_text" not in df_full.columns:
        raise ValueError(f"{df_full_path} does not contain required columns "
                         f"['rep_idx', 'processed_text'].")

    if len(df_full) != F:
        raise ValueError(
            f"Expected df_full to have F={F} rows, got {len(df_full)}. "
            "Check cfg.f_latents or the saved CSV."
        )

    # ---- Recompute the same hash that clustering.embed_texts_cached used ----
    def _hash_texts(txts: List[str]) -> str:
        h = hashlib.sha1()
        for t in txts:
            h.update(t.encode("utf-8"))
        return h.hexdigest()[:10]

    unique_texts = df_unique["processed_text"].astype(str).tolist()
    emb_hash = _hash_texts(unique_texts)
    emb_path = clu_dir / f"layer{layer:02d}_emb_{emb_hash}.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Cached unique embeddings not found: {emb_path}\n"
            "Re-run clustering.py or adjust cfg.clusters_dir."
        )

    # ---- Load unique embeddings (U, d_text) and broadcast to (F, d_text) ----
    E_unique = np.load(emb_path).astype(np.float32)
    if E_unique.shape[0] != len(df_unique):
        raise ValueError(
            f"Unique embeddings row-count ({E_unique.shape[0]}) "
            f"!= df_unique rows ({len(df_unique)})."
        )

    d_text = E_unique.shape[1]
    E_full = np.zeros((F, d_text), dtype=np.float32)
    no_desc_mask = np.zeros(F, dtype=bool)

    # Map representative original_index (from df_unique) -> row in E_unique
    # df_unique['original_index'] are the first occurrences (representatives)
    rep_to_row = {
        int(rep_idx): int(row_idx)
        for row_idx, rep_idx in enumerate(df_unique["original_index"].astype(int).tolist())
    }

    # Broadcast
    rep_idx_arr = df_full["rep_idx"].to_numpy(dtype=np.int64)
    for i in range(F):
        rep = rep_idx_arr[i]
        if rep == -1:  # no description
            no_desc_mask[i] = True
            continue
        row = rep_to_row.get(rep, None)
        if row is None:
            # Fallback: if something went wrong, mark as no-desc
            no_desc_mask[i] = True
        else:
            E_full[i] = E_unique[row]

    # ---- Add tiny jitter to exact duplicates (for the described ones only) ----
    E_full = _jitter_duplicates(E_full, sigma=cfg.sigma_jitter, seed=cfg.seed + layer)

    # ---- Replace "no description" rows with random unit vectors ----
    if no_desc_mask.any():
        rng = np.random.default_rng(cfg.seed + 10_000 + layer)
        rand = rng.normal(size=(no_desc_mask.sum(), d_text)).astype(np.float32)
        rand /= (np.linalg.norm(rand, axis=1, keepdims=True) + 1e-12)
        E_full[no_desc_mask] = rand

    # ---- Final L2-normalization ----
    norms = np.linalg.norm(E_full, axis=1, keepdims=True) + 1e-12
    E_full = E_full / norms

    # ---- Return torch tensors on CPU ----
    E_torch = torch.from_numpy(E_full).float()
    no_desc_mask_t = torch.from_numpy(no_desc_mask)
    return E_torch, no_desc_mask_t


# ─────────────────────────────────────────────────────────────────────────────
# NEW: a version of the solver that takes arbitrary F
# (so we don't rely on cfg.f_latents being exact)
# ─────────────────────────────────────────────────────────────────────────────
def _solve_sparse_assignment_F(rows, cols, vals, metric: str, F: int) -> np.ndarray:
    """Same as _solve_sparse_assignment, but parameterized by F."""
    r = _to_np(rows).astype(np.int32)
    c = _to_np(cols).astype(np.int32)
    d = _to_np(vals).astype(np.float32)

    scale = 10_000 if metric == "cos" else 1_000
    cost = np.round(d * scale).astype(np.int32)
    # safe INF in int32
    INF  = min(np.iinfo(np.int32).max // 4, int(cost.max()) * 10 + 1)

    mcf = min_cost_flow.SimpleMinCostFlow()

    # supplies/demands
    for i in range(F):
        mcf.set_node_supply(i, 1)
        mcf.set_node_supply(F + i, -1)

    # k-NN arcs
    for u, v, cst in zip(r, c, cost):
        mcf.add_arc_with_capacity_and_unit_cost(int(u), F + int(v), 1, int(cst))

    # dummy arcs
    for i in range(F):
        mcf.add_arc_with_capacity_and_unit_cost(i, F + i, 1, INF)

    status = mcf.solve()
    if status != mcf.OPTIMAL:
        raise RuntimeError(f"OR-Tools MinCostFlow failed with status = {status}")

    perm = np.empty(F, np.int32)
    for arc in range(mcf.num_arcs()):
        if mcf.flow(arc):
            u = mcf.tail(arc)
            v = mcf.head(arc) - F
            if 0 <= u < F and 0 <= v < F:
                perm[u] = v

    return perm


# ─────────────────────────────────────────────────────────────────────────────
# NEW: text-embedding based sparse matching
# ─────────────────────────────────────────────────────────────────────────────
def sae_match_text(layer_a: int, layer_b: int):
    """
    Sparse matching in the space of text-embeddings (cosine distance),
    with a penalty for pairs involving 'no description' features.
    """
    # Load per-layer text embeddings
    Ea, no_a = _load_text_embeddings(layer_a)  # (F, d_text), (F,)
    Eb, no_b = _load_text_embeddings(layer_b)

    F = Ea.size(0)
    assert Eb.size(0) == F, "Text embedding rows mismatch across layers."

    device = cfg.device_match
    Ea, Eb = Ea.to(device), Eb.to(device)
    no_a_t = no_a.to(device)
    no_b_t = no_b.to(device)

    # Build sparse k-NN in text space (cosine)
    # vals = 1 - cosine_similarity (since Ea/Eb are L2-normalized)
    rows, cols, vals = _knn_edges(Ea, Eb, metric="cos")

    # Add penalty if at least one endpoint has no description
    penalty_mask = (no_a_t[rows] | no_b_t[cols]).float()
    vals = vals + cfg.lambda_no_desc * penalty_mask

    # Solve assignment (parameterized by F, not cfg.f_latents)
    perm = _solve_sparse_assignment_F(rows, cols, vals, metric="cos", F=F)

    # Mean cosine sanity check (pure cosine, without penalty)
    perm_t = torch.from_numpy(perm).to(device)
    rng = np.random.default_rng(cfg.seed + 2000 * layer_a + layer_b)
    sample = min(1_000, F)
    idx_np = rng.choice(F, sample, replace=False)
    idx = torch.from_numpy(idx_np).to(device)

    mc = (Ea[idx] * Eb[perm_t[idx]]).sum(1).mean().item()

    return perm, mc

# ─────────────────────────────────────────────────────────────────────────────
# EV: Explained-Variance utilities
# ─────────────────────────────────────────────────────────────────────────────

def ev_ratio(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Var(pred - target) / Var(target). Lower is better."""
    return (torch.var(pred - target, unbiased=False) /
            torch.var(target,         unbiased=False)).item()

def ev_1minus(pred: torch.Tensor, target: torch.Tensor) -> float:
    """1 - Var(residual) / Var(target). Higher is better."""
    return 1.0 - ev_ratio(pred, target)

def _sample_starts(seq_len: int, ctx_len: int, k: int, seed: int) -> np.ndarray:
    """Pick k non-overlapping windows of length ctx_len."""
    rng = np.random.default_rng(seed)
    all_starts = np.arange(0, seq_len - ctx_len + 1, ctx_len)
    k = min(k, len(all_starts))
    return np.sort(rng.choice(all_starts, size=k, replace=False))

def _collect_pair_activations(model: tl.HookedTransformer,
                              layer: int,
                              starts: np.ndarray,
                              seq_np: np.ndarray,
                              ctx_len: int,
                              batch: int,
                              device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect (on CPU) residual streams for layers l and l+1
    at the last token position of each window.
    Returns:
        h_l   : (N, d_model) on CPU
        h_lp1 : (N, d_model) on CPU
    """
    hook_l   = f"blocks.{layer}.hook_resid_post"
    hook_lp1 = f"blocks.{layer+1}.hook_resid_post"

    h_l_list, h_lp1_list = [], []

    with torch.inference_mode():
        for i in range(0, len(starts), batch):
            idx0 = starts[i:i+batch]
            idx_mat = idx0[:, None] + np.arange(ctx_len)
            toks = torch.as_tensor(seq_np[idx_mat], dtype=torch.long, device=device)

            buf_l, buf_lp1 = [], []

            def cap(buf):
                def _hook(out, *, hook=None):
                    # append last-token hidden state, move to CPU
                    buf.append(out[:, -1].cpu())
                return _hook

            model.run_with_hooks(
                toks,
                fwd_hooks=[(hook_l,   cap(buf_l)),
                           (hook_lp1, cap(buf_lp1))],
                reset_hooks_end=True,
            )

            h_l_list.append(buf_l[0])
            h_lp1_list.append(buf_lp1[0])

            del toks, buf_l, buf_lp1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return torch.cat(h_l_list), torch.cat(h_lp1_list)

def _forward_through_sae_pair(
    sae_l,
    sae_lp1,
    perm: torch.Tensor,
    h_l: torch.Tensor,
    mean_lp1: torch.Tensor,
    std_lp1: torch.Tensor,
) -> torch.Tensor:
    """
    Encode with SAE_l, permute latents by `perm`, then decode with SAE_{l+1},
    passing mean/std computed on h_{l+1} so that decode() properly de-normalises
    into the raw resid space of layer l+1.

    All tensors are on CPU.
    """
    with torch.inference_mode():
        z, _ = sae_l.encode(h_l)  # info from layer l is NOT used
        return sae_lp1.decode(z[:, perm], info={"mean": mean_lp1, "std": std_lp1})


def compute_ev_for_perm_pair(
    layer: int,
    perm: np.ndarray,
    ctx_len:   int | None = None,
    windows_k: int | None = None,
    tokens_path: Path | None = None,
    batch: int | None = None,
    seed:  int | None = None,
    device_gpu: str | None = None,
) -> float:
    """
    Compute EV_ratio = Var(pred-target) / Var(target) for one layer pair (l → l+1),
    in raw-residual space of layer l+1.  
    All diagnostic prints from model loading are silenced.
    """

    # ---------- fall-back to global config ----------
    ctx_len     = ctx_len     or cfg.ctx_len
    windows_k   = windows_k   or cfg.windows_k
    tokens_path = tokens_path or cfg.tokens_path
    batch       = batch       or cfg.batch_gpu
    seed        = seed        or cfg.seed
    device_gpu  = device_gpu  or cfg.device_match

    # ---------- prepare text windows ----------
    seq = np.load(tokens_path)
    seq = seq[: (len(seq) // ctx_len) * ctx_len]
    starts = _sample_starts(len(seq), ctx_len, windows_k, seed)

    # ---------- load GPT-2 once, suppressing stdout ----------
    import io, contextlib
    silent_buf = io.StringIO()
    with contextlib.redirect_stdout(silent_buf):            # <── no console noise
        model = tl.HookedTransformer.from_pretrained(
            "gpt2-small",
            center_writing_weights=False,
            device=device_gpu,
        ).eval()

    # ---------- load SAE checkpoints (CPU) ----------
    from sae_v5_32k import load_sae_layer
    sae_l   = load_sae_layer(cfg.checkpoints_dir / f"layer{layer}.pt",     device="cpu")
    sae_lp1 = load_sae_layer(cfg.checkpoints_dir / f"layer{layer+1}.pt",   device="cpu")

    # ---------- collect residual activations ----------
    h_l, h_lp1 = _collect_pair_activations(
        model, layer, starts, seq, ctx_len, batch, device_gpu
    )

    # ---------- stats of layer l+1 for de-normalisation ----------
    _, mean_lp1, std_lp1 = sae_lp1._layer_norm(h_lp1)

    # ---------- encode-permute-decode ----------
    perm_t = torch.from_numpy(perm).long()           # CPU tensor
    h_hat  = _forward_through_sae_pair(
        sae_l, sae_lp1, perm_t, h_l, mean_lp1, std_lp1
    )

    # ---------- explained-variance ratio ----------
    return ev_ratio(h_hat, h_lp1)

def compute_ev_for_perm_dir(perm_dir: Path,
                            metric: str,
                            ctx_len:   int = None,
                            windows_k: int = None,
                            tokens_path: Path = None,
                            batch: int = None,
                            seed:  int = None,
                            device_gpu: str = None) -> Dict[str, float]:
    """
    Compute EV (as ev_ratio) for all 0->1, 1->2, ..., L-2->L-1 pairs,
    reading permutations from perm_dir/P_??_??.npy,
    **in the raw resid space of layer l+1 **.
    """
    ctx_len     = ctx_len     or cfg.ctx_len
    windows_k   = windows_k   or cfg.windows_k
    tokens_path = tokens_path or cfg.tokens_path
    batch       = batch       or cfg.batch_gpu
    seed        = seed        or cfg.seed
    device_gpu  = device_gpu  or cfg.device_match

    # 1) tokens + window starts
    seq = np.load(tokens_path)
    seq = seq[: (len(seq)//ctx_len)*ctx_len]
    starts = _sample_starts(len(seq), ctx_len, windows_k, seed)
    print(f"• {len(starts)} windows × {ctx_len} tokens (≈ {len(starts)*ctx_len:,} tokens)")

    # 2) GPT-2 small (FP32)
    model = tl.HookedTransformer.from_pretrained(
        "gpt2-small",
        center_writing_weights=False,
        device=device_gpu
    ).eval()

    # 3) SAE checkpoints (CPU)
    from sae_v5_32k import load_sae_layer
    sae = {
        l: load_sae_layer(cfg.checkpoints_dir / f"layer{l}.pt", device="cpu")
        for l in range(cfg.num_layers)
    }

    ev_dict: Dict[str, float] = {}
    for l in range(cfg.num_layers - 1):
        # 4) activations
        h_l, h_lp1 = _collect_pair_activations(
            model, l, starts, seq, ctx_len, batch, device_gpu
        )

        # 4.1) stats of layer l+1
        _, mean_lp1, std_lp1 = sae[l+1]._layer_norm(h_lp1)

        perm_path = perm_dir / f"P_{l:02d}_{l+1:02d}.npy"
        if not perm_path.exists():
            raise FileNotFoundError(f"Missing permutation: {perm_path}")
        perm = torch.from_numpy(np.load(perm_path)).long()  # CPU

        # 5) forward (raw space)
        h_hat = _forward_through_sae_pair(sae[l], sae[l+1], perm, h_l, mean_lp1, std_lp1)

        # 6) EV
        ev = ev_ratio(h_hat, h_lp1)
        ev_dict[f"{l}->{l+1}"] = ev
        print(f"{perm_dir.name:>6}  {l}->{l+1} : EV_ratio = {ev:.4f}")

        del h_l, h_lp1, h_hat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return ev_dict

# ─────────────────────────────────────────────────────────────────────────────
# CLI: matching runners
# ─────────────────────────────────────────────────────────────────────────────
def cli_match(metric: str):
    # for "text" metric delegate to the text-based matcher
    if metric == "text":
        return cli_match_text()

    # all other metrics ("cos", "mse")
    out_dir = cfg.perm_root / metric
    out_dir.mkdir(parents=True, exist_ok=True)

    for l in tqdm(range(cfg.num_layers - 1), desc=f"{metric.upper()} matching", ascii=True, ncols=80, file=sys.stdout):
        tgt = out_dir / f"P_{l:02d}_{l+1:02d}.npy"
        if tgt.exists():
            continue
        t0 = time.time()
        perm, mc = sae_match_sparse(l, l + 1, metric)
        np.save(tgt, perm)
        dt = time.time() - t0
        print(f"✓ {metric.upper()} {l}->{l+1} mean‑cos={mc:.4f} [{dt/60:.1f} min] → {tgt}")


def cli_match_text():
    out_dir = cfg.perm_root / "text"
    out_dir.mkdir(parents=True, exist_ok=True)

    for l in tqdm(range(cfg.num_layers - 1), desc="TEXT matching", ascii=True, ncols=80, file=sys.stdout):
        tgt = out_dir / f"P_{l:02d}_{l+1:02d}.npy"
        if tgt.exists():
            continue
        t0 = time.time()
        perm, mc = sae_match_text(l, l + 1)
        np.save(tgt, perm)
        dt = time.time() - t0
        print(f"✓ TEXT {l}->{l+1} mean‑cos={mc:.4f} [{dt/60:.1f} min] → {tgt}")


# ─────────────────────────────────────────────────────────────────────────────
# main entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse, json
    from pathlib import Path

    p = argparse.ArgumentParser(description="Sparse SAE‑matching & EV diagnostics")
    sub = p.add_subparsers(dest="cmd", required=True)

    # match subcommand
    m = sub.add_parser("match", help="Build sparse permutations")
    m.add_argument(
        "--metric",
        choices=cfg.metric_choices,   # ("cos", "mse", "text")
        default="cos",
        help="Which space to match in: cos (decoder), mse (decoder‑L2), text (text embeddings)",
    )

    # ev subcommand
    e = sub.add_parser("ev", help="Compute explained‑variance for a metric")
    e.add_argument(
        "--metric",
        choices=cfg.metric_choices,   # ("cos", "mse", "text")
        required=True,
        help="Metric to load permutations from when computing EV",
    )
    e.add_argument("--json-out", type=Path, help="Optional output JSON file")

    args = p.parse_args()

    if args.cmd == "match":
        cli_match(args.metric)
    else:  # args.cmd == "ev"
        # Use the local compute_ev_for_perm_dir defined in this file
        perm_dir = cfg.perm_root / args.metric
        ev = compute_ev_for_perm_dir(perm_dir, args.metric)
        if args.json_out:
            args.json_out.write_text(json.dumps(ev, indent=2))
            print(f"✓ EV metrics → {args.json_out}")
        else:
            print(json.dumps(ev, indent=2))


if __name__ == "__main__":
    main()