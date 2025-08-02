# clustering.py
# ------------------------------------------------------------
# Two‑stage clustering of Neuronpedia descriptions
# 1) UMAP dimensionality reduction
# 2) HDBSCAN clustering  (primary sweep + noise‑refinement pass)


from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hdbscan
import numpy as np
import pandas as pd
import torch
import umap
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.std import tqdm
from collections import Counter

# --------------------------------------------------------------------------- #
# GLOBAL SEED — reproducibility                                               #
# --------------------------------------------------------------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
_NGRAM_RE = re.compile(r"\b\w+(?:'\w+)?\b")  # lowercase + apostrophes

SRC_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SRC_DIR.parent

DEFAULT_DESC_DIR = PROJECT_ROOT / "data" / "gpt2-small-res_post_32k-oai"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "clusters_all_layers"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "results" / "labels"

# --------------------------------------------------------------------------- #
# Config dataclass                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class ClusterConfig:
    # paths
    desc_dir: Path = DEFAULT_DESC_DIR
    out_dir: Path = DEFAULT_OUT_DIR
    labels_dir: Path = DEFAULT_LABELS_DIR
    ckpt_dir: Path = PROJECT_ROOT / "checkpoints"

    # model
    model_name: str = "all-mpnet-base-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # constants
    F: int = 32_768

    # primary sweep
    n_neighbors_grid: List[int] = field(default_factory=lambda: [10, 20])
    min_cluster_size_grid: List[int] = field(default_factory=lambda: [8, 12, 20])
    umap_base: Dict[str, Any] = field(
        default_factory=lambda: dict(
            min_dist=0.1, n_components=15, metric="cosine"
        )
    )
    hdb_base: Dict[str, Any] = field(default_factory=lambda: dict(metric="euclidean"))
    primary_score: str = "mean_coherence"

    # noise‑refine sweep
    n_neighbors_fine: List[int] = field(default_factory=lambda: [5, 10, 15])
    min_cluster_size_fine: List[int] = field(default_factory=lambda: [3, 4, 6])
    umap_fine: Dict[str, Any] = field(
        default_factory=lambda: dict(
            min_dist=0.05, n_components=15, metric="cosine"
        )
    )
    hdb_fine: Dict[str, Any] = field(
        default_factory=lambda: dict(metric="euclidean", min_samples=1)
    )
    t_fine_score_filter: float = 0.20


cfg = ClusterConfig()
cfg.out_dir.mkdir(parents=True, exist_ok=True)
cfg.labels_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Text utils                                                                  #
# --------------------------------------------------------------------------- #
LEADING = re.compile(
    r"""^(references?\s+to|mentions?\s+of|instances?\s+of(?:\s+the)?|
          occurrences?\s+of(?:\s+the)?|
          (?:[A-Za-z]+\s+){0,3}(?:related|relating|associated)\s+to|
          statements?\s+about|technical\s+terms\s+and\s+keywords\s+related\s+to|
          letter\s+'?[A-Za-z]'\s+in\s+various\s+forms(?:\s+or\s+contexts)?|
          prepositions?\s+indicating|specific\s+terms\s+and\s+numerical\s+data\s+related\s+to|
          variations?\s+of\s+the\s+suffix|phrases?\s+indicating|words?\s+associated\s+with|
          names?\s+related\s+to|details?\s+related\s+to|content\s+related\s+to|
          patterns?\s+related\s+to|the\s+presence\s+of|various\s+types\s+of|
          markers?\s+of|the\s+concept\s+of)\b[\s:,-]*""",
    flags=re.I | re.VERBOSE,
)
TRAILING = re.compile(
    r"(?:\s+(?:in|of|across|within)\s+(?:a|the|various|different|multiple|diverse)\s+"
    r"(?:context|contexts|formats|forms|settings|situations))\s*$",
    flags=re.I,
)


def clean_text(text: str) -> str:
    s = unicodedata.normalize("NFKC", text.strip())
    while True:
        new_s = LEADING.sub("", s).lstrip()
        if new_s == s:
            break
        s = new_s
    s = TRAILING.sub("", s).strip()
    return re.sub(r"\s+", " ", s)


# --------------------------------------------------------------------------- #
# I/O helpers                                                                 #
# --------------------------------------------------------------------------- #
def load_and_preprocess(layer_idx: int) -> pd.DataFrame:
    """Return DF with columns: original_index, raw_text, processed_text."""
    fp = cfg.desc_dir / f"gpt2-small_{layer_idx}-res_post_32k-oai.json"
    if not fp.exists():
        raise FileNotFoundError(fp)

    with fp.open(encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(
        {"original_index": np.arange(cfg.F), "raw_text": ["" for _ in range(cfg.F)]}
    )
    for e in data:
        try:
            idx = int(e["index"])
            if 0 <= idx < cfg.F:
                df.at[idx, "raw_text"] = e["description"].strip()
        except (KeyError, ValueError):
            continue

    df = df[df["raw_text"].str.strip() != ""].copy()
    df["processed_text"] = df["raw_text"].apply(clean_text)
    df = (
        df[df["processed_text"] != ""]
        .drop_duplicates("processed_text", keep="first")
        .reset_index(drop=True)
    )
    return df


def _hash_texts(txts: List[str]) -> str:
    h = hashlib.sha1()
    for t in txts:
        h.update(t.encode())
    return h.hexdigest()[:10]


def embed_texts_cached(
    df: pd.DataFrame, layer_idx: int, model: SentenceTransformer, batch_size: int = 128
) -> np.ndarray:
    texts = df["processed_text"].tolist()
    cache = cfg.out_dir / f"layer{layer_idx:02d}_emb_{_hash_texts(texts)}.npy"
    if cache.exists():
        emb = np.load(cache, allow_pickle=False)
        # ── sanity‑check cache
        valid = (
            isinstance(emb, np.ndarray)
            and emb.ndim == 2
            and emb.shape[0] == len(texts)
            and emb.shape[1] > 0
            and np.isfinite(emb).all()
        )
        if valid:
            tqdm.write(f"  • cached embeddings loaded ({cache.name})")
            return emb
        try:
            cache.unlink()
        except OSError:
            pass

    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.save(cache, emb)
    tqdm.write(f"  • embeddings cached ({cache.name})")
    return emb

# --------------------------------------------------------------------------- #
# ML helpers                                                                  #
# --------------------------------------------------------------------------- #
def reduce_umap(emb: np.ndarray, *, n_neighbors: int) -> np.ndarray:
    X = np.ascontiguousarray(emb, dtype=np.float32)
    # ── sanity checks before UMAP
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"reduce_umap: expected non‑empty 2D array, got shape={emb.shape}")
    if not np.isfinite(X).all():
        raise ValueError("reduce_umap: embeddings contain NaN or Inf — cache may be corrupted")

    params = dict(cfg.umap_base, n_neighbors=n_neighbors)
    reducer = umap.UMAP(**params, random_state=SEED, verbose=False)
    red = reducer.fit_transform(X)
    if red.shape[0] != X.shape[0]:
        raise RuntimeError(f"reduce_umap: unexpected number of rows in output: {red.shape[0]} != {X.shape[0]}")
    return red

def cluster_hdb(reduced: np.ndarray, *, min_cluster_size: int) -> np.ndarray:
    params = dict(cfg.hdb_base, min_cluster_size=min_cluster_size)
    try:
        return hdbscan.HDBSCAN(
            **params,
            cluster_selection_method="eom",
            core_dist_n_jobs=-1,
            approx_min_span_tree=True
        ).fit_predict(reduced)
    except ValueError:
        # On failure, return "all noise" labels silently
        n = reduced.shape[0]
        return np.full(n, -1, dtype=int)

def compute_metrics(
    reduced: np.ndarray, base_emb: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    mask = labels >= 0
    uniq = np.unique(labels[mask])
    m: Dict[str, float] = {"noise_ratio": float((labels == -1).mean())}

    if len(uniq) >= 2:
        m["silhouette"] = float(
            silhouette_score(reduced[mask], labels[mask], metric="euclidean")
        )
        m["davies_bouldin"] = float(davies_bouldin_score(reduced[mask], labels[mask]))
        m["calinski_harabasz"] = float(
            calinski_harabasz_score(reduced[mask], labels[mask])
        )
    else:
        m.update(
            silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan
        )

    coherences = []
    for c in uniq:
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        sim = cosine_similarity(base_emb[idx])
        coherences.append(sim[np.triu_indices_from(sim, k=1)].mean())
    m["mean_coherence"] = float(np.mean(coherences)) if coherences else np.nan
    return m


# --------------------------------------------------------------------------- #
# Noise‑refinement                                                            #
# --------------------------------------------------------------------------- #
def _cluster_noise(
    emb_noise: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int], Optional[int]]:
    # return best labels, best score, and the (n_neighbors, min_cluster_size) that achieved it
    best_score, best_lbls = None, None
    best_n_nb, best_mcs = None, None
    for n_nb in cfg.n_neighbors_fine:
        red = umap.UMAP(
            **cfg.umap_fine,
            n_neighbors=n_nb,
            random_state=SEED,
            verbose=False
        ).fit_transform(emb_noise)

        for mcs in cfg.min_cluster_size_fine:
            lbls = hdbscan.HDBSCAN(
                **cfg.hdb_fine,
                min_cluster_size=mcs,
                cluster_selection_method="eom",
                core_dist_n_jobs=-1
            ).fit_predict(red)

            if (lbls >= 0).sum() < 5:
                continue

            score = compute_metrics(red, emb_noise, lbls)[cfg.primary_score]
            if best_score is None or score > best_score:
                best_score, best_lbls = score, lbls
                best_n_nb, best_mcs = n_nb, mcs

    return best_lbls, best_score, best_n_nb, best_mcs

def _refine_noise(
    df_unique: pd.DataFrame, emb_unique: np.ndarray, primary_K: int
) -> Tuple[pd.DataFrame, Optional[int], Optional[int], int]:
    # returns (updated df_unique, refine_n_nb, refine_mcs, tiny_added)
    noise_mask = df_unique["cluster"] == -1
    if not noise_mask.any():
        # log to stderr to avoid breaking the tqdm bar
        tqdm.write("  • refine_noise: no noise points")
        return df_unique, None, None, 0

    emb_noise = emb_unique[noise_mask.values]
    if emb_noise.shape[0] < 5:
        return df_unique, None, None, 0

    lbls, score, ref_n_nb, ref_mcs = _cluster_noise(emb_noise)
    if (
        lbls is None
        or (lbls >= 0).sum() < 5
        or score < cfg.t_fine_score_filter
    ):
        # log to stderr to avoid breaking the tqdm bar
        tqdm.write("  • refine_noise: nothing accepted")
        return df_unique, None, None, 0

    uniq_new = sorted({c for c in lbls if c >= 0})
    remap = {old: primary_K + i for i, old in enumerate(uniq_new)}
    new_ids = np.array([remap.get(c, -1) for c in lbls], dtype=int)

    # assign cluster ids back to the noise rows
    df_unique.loc[noise_mask, "cluster"] = new_ids
    tiny_added = len(uniq_new)

    # the caller (run_layer_auto) will print a single consolidated line
    return df_unique, ref_n_nb, ref_mcs, tiny_added

# --------------------------------------------------------------------------- #
# Build full‐labels vector and DataFrame                                      #
# --------------------------------------------------------------------------- #
def _build_full_labels(
    layer_idx: int,
    df_unique: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
      •  label  = cluster id   for every latent that has *some* description;
                 duplicates inherit the cluster of their representative text.
      •  label  = –2           if the latent has no description at all.
      •  rep_idx column        points to the row that survived de‑duplication.

    Columns in the returned DataFrame:
        original_index | processed_text | rep_idx | cluster
    """
    F = cfg.F

    # ── 1.  Build helper maps from the unique‑rows DF ──────────────────────
    # map processed_text  -> representative latent index (first occurrence)
    map_pt_to_rep: Dict[str, int] = {}
    # map processed_text  -> final cluster id (after refine_noise)
    map_pt_to_cluster: Dict[str, int] = {}

    for _, row in df_unique.iterrows():
        pt = row["processed_text"]
        map_pt_to_rep.setdefault(pt, row["original_index"])
        map_pt_to_cluster[pt] = int(row["cluster"])

    # ── 2.  Load *all* processed descriptions for this layer ───────────────
    proc_texts = [""] * F
    json_path = cfg.desc_dir / f"gpt2-small_{layer_idx}-res_post_32k-oai.json"
    with json_path.open(encoding="utf‑8") as f:
        for entry in json.load(f):
            try:
                idx = int(entry["index"])
                if 0 <= idx < F:
                    proc_texts[idx] = clean_text(entry.get("description", ""))
            except Exception:
                continue

    # ── 3.  Build labels_vec, rep_idx, DataFrame ───────────────────────────
    labels_vec = np.full(F, -2, dtype=np.int16)   # –2 → no description
    rep_idx    = np.full(F, -1, dtype=np.int32)

    for i, pt in enumerate(proc_texts):
        if not pt:
            continue                              # label stays –2
        rep        = map_pt_to_rep.get(pt, i)
        rep_idx[i] = rep
        labels_vec[i] = map_pt_to_cluster.get(pt, -1)   # –1 → noise

    full_df = pd.DataFrame(
        dict(
            original_index=np.arange(F, dtype=np.int32),
            processed_text=proc_texts,
            rep_idx=rep_idx,
            cluster=labels_vec,
        )
    )
    return full_df, labels_vec

# --------------------------------------------------------------------------- #
# End‑to‑end per‑layer pipeline                                               #
# --------------------------------------------------------------------------- #
def run_layer_auto(
    layer_idx: int, model: SentenceTransformer
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Dict[str, float], Dict[str, float]]:
    tqdm.write(f"\n=== Layer {layer_idx:02d} ===")
    df_unique = load_and_preprocess(layer_idx)
    tqdm.write(f"  • {len(df_unique)} unique non-empty descriptions")

    emb_unique = embed_texts_cached(df_unique, layer_idx, model)

    # Primary grid search: keep the best result
    best_score = None
    best_res: Dict[str, Any] = {}
    best_red = None
    any_success = False

    for n_nb in cfg.n_neighbors_grid:
        red = reduce_umap(emb_unique, n_neighbors=n_nb)
        for mcs in cfg.min_cluster_size_grid:
            labels = cluster_hdb(red, min_cluster_size=mcs)
            # skip this combination if it produced only noise
            if (labels >= 0).sum() == 0:
                continue
            any_success = True

            metrics = compute_metrics(red, emb_unique, labels)
            score = metrics[cfg.primary_score]
            if best_score is None or score > best_score:
                best_score = score
                best_res = {
                    "labels": labels,
                    "n_neighbors": n_nb,
                    "min_cluster_size": mcs,
                    "metrics": metrics,
                }
                best_red = red

    if not any_success:
        # warn once if all combinations failed
        tqdm.write("⚠️ primary grid search: no valid clusters found, marking all as noise")
        labels = np.full(len(emb_unique), -1, dtype=int)
        metrics = {"mean_coherence": float("nan"), "noise_ratio": 1.0}
        best_res = {
            "labels": labels,
            "n_neighbors": cfg.n_neighbors_grid[0],
            "min_cluster_size": cfg.min_cluster_size_grid[0],
            "metrics": metrics,
        }
        best_red = reduce_umap(emb_unique, n_neighbors=cfg.n_neighbors_grid[0])

    # unpack the best result
    primary_labels = best_res["labels"]
    n_clusters = len(np.unique(primary_labels[primary_labels >= 0]))
    pm = best_res["metrics"]
    tqdm.write(
        f"  • primary: n_nb={best_res['n_neighbors']}, "
        f"mcs={best_res['min_cluster_size']} → "
        f"{cfg.primary_score}={pm[cfg.primary_score]:.3f}, "
        f"noise={pm['noise_ratio']:.3f}, "
        f"{n_clusters} clusters"
    )

    df_unique["cluster"] = primary_labels
    df_unique["n_neighbors"] = best_res["n_neighbors"]
    df_unique["min_cluster_size"] = best_res["min_cluster_size"]
    metrics_primary = pm

    # Noise refinement
    k_before = df_unique["cluster"].loc[lambda s: s >= 0].nunique()
    df_unique, ref_n_nb, ref_mcs, added = _refine_noise(df_unique, emb_unique, k_before)

    final_labels = df_unique["cluster"].to_numpy(dtype=int)
    metrics_refined = compute_metrics(best_red, emb_unique, final_labels)
    n_clusters_refined = len(np.unique(final_labels[final_labels >= 0]))

    rn = ref_n_nb if ref_n_nb is not None else "–"
    rm = ref_mcs if ref_mcs is not None else "–"
    tqdm.write(
        f"  • refine_noise: n_nb={rn}, mcs={rm} → "
        f"{cfg.primary_score}={metrics_refined[cfg.primary_score]:.3f}, "
        f"noise={metrics_refined['noise_ratio']:.3f}, "
        f"+{added} clusters"
    )

    total_noise = int((final_labels == -1).sum())
    total_points = len(df_unique)
    tqdm.write(
        f"  • total: {n_clusters_refined} clusters, "
        f"mean_coherence={metrics_refined['mean_coherence']:.3f}, "
        f"noise={metrics_refined['noise_ratio']:.3f} "
        f"({total_noise}/{total_points})"
    )

    full_df, labels_vec = _build_full_labels(layer_idx, df_unique)
    return df_unique, full_df, labels_vec, metrics_primary, metrics_refined

# --------------------------------------------------------------------------- #
# Persist                                                                     #
# --------------------------------------------------------------------------- #
def save_results(
    layer_idx: int,
    df_unique: pd.DataFrame,
    df_full: pd.DataFrame,
    labels_vec: np.ndarray,
    metrics_primary: Dict[str, float],
    metrics_refined: Dict[str, float],
) -> None:
    """
    Persist clustering outputs exactly as in the notebook:
      • unique descriptions  →  layerXX_clusters.csv
      • all F latents        →  layerXX_full_labels.csv (with processed_text & rep_idx)
      • final labels vector  →  labels/cluster_labels_LXX.npy
      • metrics (JSON)       →  layerXX_metrics.json
    """
    uni_csv  = cfg.out_dir   / f"layer{layer_idx:02d}_clusters.csv"
    full_csv = cfg.out_dir   / f"layer{layer_idx:02d}_full_labels.csv"
    lbl_npy  = cfg.labels_dir / f"cluster_labels_L{layer_idx:02d}.npy"
    meta_js  = cfg.out_dir   / f"layer{layer_idx:02d}_metrics.json"

    df_unique.to_csv(uni_csv, index=False)
    df_full.to_csv(full_csv, index=False)          # now includes processed_text & rep_idx
    np.save(lbl_npy, labels_vec)

    with meta_js.open("w", encoding="utf‑8") as f:
        json.dump(
            {"primary_metrics": metrics_primary, "refined_metrics": metrics_refined},
            f,
            indent=2,
        )

    tqdm.write(
        f"✔ layer {layer_idx:02d}: "
        f"unique={len(df_unique):5d}, full={len(df_full):5d}, "
        f"primary_{cfg.primary_score}={metrics_primary[cfg.primary_score]:.3f}, "
        f"refined_{cfg.primary_score}={metrics_refined[cfg.primary_score]:.3f}"
    )

def describe_cluster(layer_idx: int, cluster_id: int) -> None:
    """
    Print all raw_text descriptions for the specified cluster,
    preceded by layer, cluster id, item count, and coherence.
    """
    csv_path = cfg.out_dir / f"layer{layer_idx:02d}_clusters.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Clusters file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    sel = df[df["cluster"] == cluster_id]
    count = len(sel)
    if count == 0:
        print(f"No items in layer {layer_idx}, cluster {cluster_id}")
        return

    texts = sel["processed_text"].tolist()
    model = SentenceTransformer(cfg.model_name, device=cfg.device)
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    sims = cosine_similarity(emb)
    if count > 1:
        tri = np.triu_indices(count, k=1)
        coherence = float(sims[tri].mean())
    else:
        coherence = float("nan")

    print(f"Layer {layer_idx}, Cluster {cluster_id}, Items: {count}, Coherence: {coherence:.3f}")
    for raw in sel["raw_text"].tolist():
        print(f"- {raw}")

# --------------------------------------------------------------------------- #
# Decoder-based clustering helpers                                            #
# --------------------------------------------------------------------------- #
def load_decoder_embeddings(layer_idx: int) -> np.ndarray:
    """
    return matrice (F, 768) — columns of decoder SAE for layer `layer_idx`,
    L2-normed.
    """
    ckpt_path = cfg.ckpt_dir / f"layer{layer_idx}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    W = ckpt["decoder.weight"].detach().cpu().numpy().astype(np.float32)  # (768, F)
    emb = W.T                       # (F, 768)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb /= np.maximum(norms, 1e-9)
    return emb


def run_decoder_layer_auto(
    layer_idx: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], int, int]:
    """
    full pipeline for one layer:
      • load decoder.weight
      • do primary grid-search UMAP+HDBSCAN
      • return (labels, reduced, metrics, best_k, best_mcs)
    """
    emb = load_decoder_embeddings(layer_idx)
    best_score, best_metrics = None, None
    best_labels, best_red = None, None
    best_k, best_mcs = None, None

    for k in cfg.n_neighbors_grid:
        red = reduce_umap(emb, n_neighbors=k)
        for mcs in cfg.min_cluster_size_grid:
            labels = cluster_hdb(red, min_cluster_size=mcs)
            if (labels >= 0).sum() == 0:
                continue
            m = compute_metrics(red, emb, labels)
            score = m[cfg.primary_score]
            if best_score is None or score > best_score:
                best_score   = score
                best_metrics = m
                best_labels  = labels
                best_red     = red
                best_k, best_mcs = k, mcs

    # if all is noise
    if best_labels is None:
        k0, mcs0 = cfg.n_neighbors_grid[0], cfg.min_cluster_size_grid[0]
        best_red = reduce_umap(emb, n_neighbors=k0)
        best_labels = np.full(emb.shape[0], -1, dtype=int)
        best_metrics = compute_metrics(best_red, emb, best_labels)
        best_k, best_mcs = k0, mcs0

    tqdm.write(
        f"Decoder L{layer_idx:02d}: k={best_k}, mcs={best_mcs} → "
        f"{cfg.primary_score}={best_metrics[cfg.primary_score]:.3f}, "
        f"noise={best_metrics['noise_ratio']:.3f}"
    )
    return best_labels, best_red, best_metrics, best_k, best_mcs


def save_decoder_results(
    layer_idx: int,
    labels: np.ndarray,
    reduced: np.ndarray,
    metrics: Dict[str, float],
    n_neighbors: int,
    min_cluster_size: int,
) -> None:
    """
    save results to:
      • labels/decoder_cluster_labels_LXX.npy
      • results/decoder_clusters/layerXX_reduced.npy
      • results/decoder_clusters/layerXX_metrics.json
    """
    out_root = cfg.out_dir / "decoder_clusters"
    out_root.mkdir(parents=True, exist_ok=True)

    np.save(cfg.labels_dir / f"decoder_cluster_labels_L{layer_idx:02d}.npy", labels)
    np.save(out_root / f"layer{layer_idx:02d}_reduced.npy", reduced)

    meta_path = out_root / f"layer{layer_idx:02d}_metrics.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            dict(
                metrics=metrics,
                n_neighbors=n_neighbors,
                min_cluster_size=min_cluster_size,
            ),
            f,
            indent=2,
        )
    rel = meta_path.relative_to(PROJECT_ROOT)
    tqdm.write(f"✔ saved decoder layer {layer_idx:02d} to {rel}")

def describe_decoder_cluster(layer_idx: int, decoder_cluster_id: int) -> None:
    import pandas as pd
    import numpy as np
    from clustering import cfg

    full_csv = cfg.out_dir / f"layer{layer_idx:02d}_full_labels.csv"
    if not full_csv.exists():
        raise FileNotFoundError(full_csv)
    df_full = pd.read_csv(full_csv)

    lbl_path = cfg.labels_dir / f"decoder_cluster_labels_L{layer_idx:02d}.npy"
    if not lbl_path.exists():
        raise FileNotFoundError(lbl_path)
    labels_dec = np.load(lbl_path)

    sel_idx = np.where(labels_dec == decoder_cluster_id)[0]
    if len(sel_idx) == 0:
        print(f"No items in decoder cluster {decoder_cluster_id}")
        return

    sel = df_full[df_full["original_index"].isin(sel_idx)]
    print(f"Layer {layer_idx}, Decoder cluster {decoder_cluster_id}, Items: {len(sel)}")
    for txt in sel["processed_text"].tolist():
        print("-", txt)

# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two‑stage clustering of Neuronpedia descriptions"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--layer", type=int, help="Cluster a single layer 0–11")
    grp.add_argument("--all", action="store_true", help="Cluster all layers 0–11")
    parser.add_argument(
        "--ngrams", action="store_true", help="Show top‑20 n‑grams before clustering"
    )
    args = parser.parse_args()

    layers = range(12) if args.all else [args.layer]
    model = SentenceTransformer(cfg.model_name, device=cfg.device)

    for ℓ in tqdm(layers, desc="Layers"):
        if args.ngrams:
            print_ngrams(ℓ, n=2, top_k=20)
            print_ngrams(ℓ, n=3, top_k=20)

        df_u, df_f, lbl_vec, m_pri, m_ref = run_layer_auto(ℓ, model)
        save_results(ℓ, df_u, df_f, lbl_vec, m_pri, m_ref)


if __name__ == "__main__":
    main()