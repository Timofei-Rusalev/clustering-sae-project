# ─────────────────────────────────────────────────────────────────────────────
# File: src/visualization/additional_pictures.py
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 1. Generic dataclass and helpers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FlowConfig:
    project_root: Path
    layer_src: int
    layer_tgt: int
    src_clusters: List[int]
    tgt_clusters: List[int]
    titles_src: Mapping[int, str]
    titles_tgt: Mapping[int, str]
    lift_labels: Mapping[Tuple[int, int], float] = field(default_factory=dict)
    palette: Tuple[str, ...] = (
        "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
        "#a6d854", "#ffd92f"
    )
    figsize: Tuple[float, float] = (11.0, 6.0)
    clusters_dir: Path = Path("results/clusters_all_layers")
    transitions_dir: Path = Path("results/transitions_2step_ot/text")
    node_size_base: int = 200
    node_size_coef: int = 8

    def __post_init__(self) -> None:
        self.clusters_dir    = (self.project_root / self.clusters_dir).resolve()
        self.transitions_dir = (self.project_root / self.transitions_dir).resolve()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Generic bipartite-OT plotter
# ─────────────────────────────────────────────────────────────────────────────
def plot_bipartite_flow(cfg: FlowConfig) -> plt.Axes:
    labels_src = _load_layer_labels(cfg.clusters_dir, cfg.layer_src)
    labels_tgt = _load_layer_labels(cfg.clusters_dir, cfg.layer_tgt)
    mass_src   = _cluster_masses(labels_src, cfg.src_clusters)
    mass_tgt   = _cluster_masses(labels_tgt, cfg.tgt_clusters)

    rows, cols, vals = _load_ot_matrix(cfg.transitions_dir,
                                       cfg.layer_src, cfg.layer_tgt)
    row_index = _build_row_index(labels_src)
    all_tgt   = sorted(set(labels_tgt) - {-1})              # col-index → cluster-id

    G = _build_graph(rows, cols, vals, row_index, all_tgt,
                     cfg.src_clusters, cfg.tgt_clusters)
    pos        = _compute_positions(G, cfg.src_clusters, cfg.tgt_clusters)
    src_colors = {cid: cfg.palette[i % len(cfg.palette)]
                  for i, cid in enumerate(cfg.src_clusters)}
    tgt_colors = _mix_target_colors(G, src_colors, cfg.tgt_clusters)

    fig, ax = plt.subplots(figsize=cfg.figsize)
    _draw_nodes(ax, G, pos, mass_src, mass_tgt,
                src_colors, tgt_colors,
                cfg.node_size_base, cfg.node_size_coef)
    _draw_edges(ax, G, pos, src_colors)
    _draw_labels(ax, G, pos, cfg, src_colors)
    _finalize_axes(ax, cfg.layer_src, cfg.layer_tgt)

    # notebook-style slight shrink
    fig.set_size_inches(fig.get_figwidth()*0.95,
                        fig.get_figheight()*0.75, forward=True)
    plt.tight_layout()
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 3. I/O helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_layer_labels(base: Path, layer: int) -> pd.Series:
    return pd.read_csv(base / f"layer{layer:02d}_full_labels.csv")["cluster"]

def _cluster_masses(labels: pd.Series, ids: List[int]) -> Dict[int, int]:
    return {cid: int((labels == cid).sum()) for cid in ids}

def _load_ot_matrix(base: Path, l_src: int, l_tgt: int):
    data = np.load(base / f"trans_L{l_src:02d}{l_tgt:02d}.npz")
    return data["rows"], data["cols"], data["vals"].astype(int)

def _build_row_index(labels: pd.Series) -> Dict[int, int]:
    return {cid: i for i, cid in enumerate(sorted(set(labels) - {-1}))}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Graph, layout & drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def _build_graph(rows, cols, vals, row_index, all_tgt,
                 src_ids, tgt_ids) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from([("Lsrc", c) for c in src_ids])
    G.add_nodes_from([("Ltgt", c) for c in tgt_ids])

    for s in src_ids:
        mask  = rows == row_index[s]
        total = vals[mask].sum() or 1
        for col, v in zip(cols[mask], vals[mask]):
            tgt = all_tgt[col]
            if tgt in tgt_ids and v > 0:
                G.add_edge(("Lsrc", s), ("Ltgt", tgt),
                           weight=v, frac=v/total)
    return G


def _compute_positions(G, src_ids, tgt_ids):
    import numpy as np
    def lin(items, x, ymin=0.1, ymax=0.9):
        ys = np.linspace(ymax, ymin, len(items))
        pref = "Lsrc" if x == 0 else "Ltgt"
        return {(pref, cid): (x, y) for cid, y in zip(items, ys)}

    def edge_max(node, mode):
        edges = G.out_edges(node, data=True) if mode == "out" else G.in_edges(node, data=True)
        return max((d["weight"] for *_, d in edges), default=0)

    src_sorted = sorted(src_ids, key=lambda c: edge_max(("Lsrc", c), "out"), reverse=True)
    tgt_sorted = sorted(tgt_ids, key=lambda c: edge_max(("Ltgt", c), "in"),  reverse=True)
    return {**lin(src_sorted, 0.0), **lin(tgt_sorted, 1.0)}


def _mix_target_colors(G, src_colors, tgt_ids):
    out = {}
    for t in tgt_ids:
        inc = [(u, d["weight"]) for u, _, d in G.in_edges(("Ltgt", t), data=True)]
        if not inc:
            out[t] = "#cccccc"; continue
        tot = sum(w for _, w in inc)
        rgb = sum(np.array(mcolors.to_rgb(src_colors[u[1]]))*w for u,w in inc)/tot
        out[t] = mcolors.to_hex(rgb)
    return out


def _draw_nodes(ax, G, pos, m_src, m_tgt, c_src, c_tgt, base, coef):
    import networkx as nx
    size = lambda m: base + coef*m
    nx.draw_networkx_nodes(G, pos,
        nodelist=[n for n in G if n[0] == "Lsrc"],
        node_size=[size(m_src[n[1]]) for n in G if n[0] == "Lsrc"],
        node_color=[c_src[n[1]] for n in G if n[0] == "Lsrc"],
        edgecolors="black", ax=ax)
    nx.draw_networkx_nodes(G, pos,
        nodelist=[n for n in G if n[0] == "Ltgt"],
        node_size=[size(m_tgt[n[1]]) for n in G if n[0] == "Ltgt"],
        node_color=[c_tgt[n[1]] for n in G if n[0] == "Ltgt"],
        edgecolors="black", ax=ax)


def _draw_edges(ax, G, pos, c_src):
    mx = max(d["frac"] for *_, d in G.edges(data=True))
    for u, v, d in G.edges(data=True):
        x1,y1 = pos[u]; x2,y2 = pos[v]
        lw = 2 + 6*d["frac"]/mx
        ax.annotate("",
            xy=(x2,y2), xytext=(x1,y1),
            arrowprops=dict(arrowstyle="-|>", lw=lw,
                            color=c_src[u[1]],
                            shrinkA=15, shrinkB=15,
                            connectionstyle="arc3,rad=0.03"))


def _draw_labels(ax, G, pos, cfg, c_src):
    for (layer,cid),(x,y) in pos.items():
        title = (cfg.titles_src if layer=="Lsrc" else cfg.titles_tgt).get(cid,"")
        dx    = -0.04 if layer=="Lsrc" else 0.03
        ha    = "right" if layer=="Lsrc" else "left"
        ax.text(x+dx, y, f"cluster {cid}\n[{title}]",
                ha=ha, va="center", fontsize=10, fontweight="bold")

    for u,v,d in G.edges(data=True):
        x1,y1 = pos[u]; x2,y2 = pos[v]
        xl,yl = x1+0.3*(x2-x1), y1+0.3*(y2-y1)
        yl   += cfg.lift_labels.get((u[1],v[1]), 0.0)
        ax.text(xl, yl, f"{d['frac']:.0%}",
                color=c_src[u[1]], fontsize=9, ha="center", va="center")


def _finalize_axes(ax, l_src, l_tgt):
    ax.set_title(f"Evolution of few clusters by optimal transport ({l_src} → {l_tgt} layers)",
                 fontsize=14)
    ax.axis("off")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Shortcut 1 – L5 → L6 OT graph
# ─────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

def plot_l5_l6_flow() -> plt.Axes:
    cfg = FlowConfig(
        project_root = _PROJECT_ROOT,
        layer_src    = 5,
        layer_tgt    = 6,
        src_clusters = [725,244,152,243,254,722],
        tgt_clusters = [368,1347,1457,1516],
        titles_src   = {
            725:"scientific reports",244:"reported statements",152:"importance emphasis",
            243:"quoted dialogue",254:"reporting & docs",722:"commentary & opinions"
        },
        titles_tgt   = {
            368:"reported speech",1347:"tech company names",
            1457:"technical operations",1516:"confession & suspicion"
        },
        lift_labels  = {(244,368):0.015, (254,1516):0.015}
    )
    return plot_bipartite_flow(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Shortcut 2 – custom SAE-Match graph L8 → L9
# ─────────────────────────────────────────────────────────────────────────────
def plot_l8_l9_flow() -> plt.Axes:
    """
    Render the two-column SAE-Match flow (clusters 711/702 → 998/630,
    layers 8 → 9) exactly as in the final notebook cell.
    """
    import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.markers import MarkerStyle
    from matplotlib.transforms import Affine2D
    from pathlib import Path

    # — configuration —
    LAYER_SRC, LAYER_TGT = 8, 9
    SRC_CLUSTERS = [711, 702]
    TGT_CLUSTERS = [998, 630]
    METRIC = "text"
    TEXT_COL = "processed_text"
    cluster_headings = {
        711: "cluster 711 [AI & NN]",
        702: "cluster 702 [Neuroanatomy]",
        998: "cluster 998 [AI]",
        630: "cluster 630 [Neuroscience]"
    }

    project_root = Path(__file__).parent.parent.parent.resolve()
    clusters_dir = project_root / "results" / "clusters_all_layers"
    perm_dir     = project_root / "results" / "permutations"

    df_src = pd.read_csv(clusters_dir / f"layer{LAYER_SRC:02d}_full_labels.csv")
    df_tgt = pd.read_csv(clusters_dir / f"layer{LAYER_TGT:02d}_full_labels.csv")
    P_text = np.load(perm_dir / METRIC / f"P_{LAYER_SRC:02d}_{LAYER_TGT:02d}.npy")

    if TEXT_COL not in df_src.columns: TEXT_COL = "description"
    if TEXT_COL not in df_tgt.columns: TEXT_COL = "description"

    def lighten(c, amt=0.8):
        rgb, white = np.array(to_rgb(c)), np.ones(3)
        return tuple(rgb*(1-amt) + white*amt)

    # — source nodes with swaps —
    src_nodes = [{"idx": int(i), "text": df_src.at[i, TEXT_COL], "cluster": sc}
                 for sc in SRC_CLUSTERS
                 for i in df_src.index[df_src["cluster"] == sc]]
    i711 = [k for k,n in enumerate(src_nodes) if n["cluster"] == 711]
    if len(i711) >= 4:
        src_nodes[i711[0]], src_nodes[i711[3]] = src_nodes[i711[3]], src_nodes[i711[0]]
        src_nodes[i711[2]], src_nodes[i711[3]] = src_nodes[i711[3]], src_nodes[i711[2]]
    i702 = [k for k,n in enumerate(src_nodes) if n["cluster"] == 702]
    if len(i702) >= 6:
        src_nodes[i702[2]], src_nodes[i702[5]] = src_nodes[i702[5]], src_nodes[i702[2]]

    links = [(n["idx"], int(P_text[n["idx"]])) for n in src_nodes]

    # — target nodes with swaps —
    linked630 = {j for _,j in links if df_tgt.at[j,"cluster"] == 630}
    tgt_nodes, seen = [], set()
    for tc in TGT_CLUSTERS:
        for j in df_tgt.index[df_tgt["cluster"] == tc]:
            txt = df_tgt.at[j, TEXT_COL]
            if txt in seen: continue
            if tc == 998 or (tc == 630 and j in linked630):
                seen.add(txt)
                tgt_nodes.append({"idx": int(j), "text": txt, "cluster": tc})
    for j in df_tgt.index[df_tgt["cluster"] == 630]:
        if j not in linked630:
            txt = df_tgt.at[j, TEXT_COL]
            if txt not in seen:
                tgt_nodes.append({"idx": int(j), "text": txt, "cluster": 630})
                break
    i998 = [k for k,t in enumerate(tgt_nodes) if t["cluster"] == 998]
    if len(i998) >= 3:
        tgt_nodes[i998[1]], tgt_nodes[i998[2]] = tgt_nodes[i998[2]], tgt_nodes[i998[1]]

    # — positions —
    N_src, N_tgt = len(src_nodes), len(tgt_nodes)
    y_src = np.linspace(1,0,N_src)
    y_tgt = np.linspace(1,0,N_tgt)
    x_src, x_tgt = 0.0, 0.98
    pos_src = {n["idx"]:(x_src, y) for n,y in zip(src_nodes, y_src)}
    pos_tgt = {n["idx"]:(x_tgt, y) for n,y in zip(tgt_nodes, y_tgt)}
    gap = 0.1
    for n in src_nodes:
        if n["cluster"] == 702:
            x,y = pos_src[n["idx"]]; pos_src[n["idx"]] = (x, y-gap)
    for n in tgt_nodes:
        if n["cluster"] == 630:
            x,y = pos_tgt[n["idx"]]; pos_tgt[n["idx"]] = (x, y-gap)
    shift = 0.03
    for d in (pos_src, pos_tgt):
        for idx in d:
            x,y = d[idx]; d[idx] = (x, y+shift)

    # — colours —
    src_palette = sns.color_palette("Set2", len(SRC_CLUSTERS))
    src_c = {sc: src_palette[i] for i,sc in enumerate(SRC_CLUSTERS)}
    tgt_c = {998:"#FF6666", 630:sns.color_palette("Set1",2)[1]}
    src_l = {sc: lighten(src_c[sc]) for sc in SRC_CLUSTERS}
    tgt_l = {tc: lighten(tgt_c[tc]) for tc in TGT_CLUSTERS}

    # — figure —
    fig, ax = plt.subplots(figsize=(12, max(N_src,N_tgt)*0.3 + 2.3))
    ax.axis("off")
    fig.suptitle("Evolution of few clusters by SAE Match (8 → 9 layers)",
                 fontsize=14, y=0.92)

    # — link geometry tweaks —
    cell_shift       = 0.05
    left_offset      = 0.25
    right711_offset  = 0.20
    right702_offset  = 0.20
    src_711  = sorted([n["idx"] for n in src_nodes if n["cluster"] == 711],
                      key=lambda i: pos_src[i][1], reverse=True)
    second_711 = src_711[1] if len(src_711)>1 else None
    src_702  = sorted([n["idx"] for n in src_nodes if n["cluster"] == 702],
                      key=lambda i: pos_src[i][1], reverse=True)
    third_702, fifth_702 = (src_702[2] if len(src_702)>2 else None,
                            src_702[4] if len(src_702)>4 else None)
    tgt_630  = sorted([n["idx"] for n in tgt_nodes if n["cluster"] == 630],
                      key=lambda i: pos_tgt[i][1], reverse=True)
    first_630, fifth_630 = (tgt_630[0] if tgt_630 else None,
                            tgt_630[4] if len(tgt_630)>4 else None)

    for i,j in links:
        if j not in pos_tgt:
            continue
        x0,y0 = pos_src[i]
        if i == second_711: x0 += right711_offset
        if i == third_702:  x0 += right702_offset
        x1,y1 = pos_tgt[j]
        if df_tgt.at[j,"cluster"] == 630: x1 -= left_offset
        y0s,y1s = y0-cell_shift, y1-cell_shift
        ax.plot([x0,x1],[y0s,y1s], color="black", linewidth=1, zorder=1)

        t = 0.5
        if (i,j) == (second_711, first_630):   t = 0.82
        elif (i,j) == (fifth_702, fifth_630): t = 0.70

        xm = x0 + t*(x1-x0)
        ym = y0s + t*(y1s-y0s)
        angle = np.degrees(np.arctan2(y1s-y0s, x1-x0))
        m = MarkerStyle(">")
        m._transform = Affine2D().rotate_deg(angle) + m.get_transform()
        ax.plot(xm, ym, marker=m, color="black", markersize=6, zorder=2)

    # — text cells —
    for n in src_nodes:
        x,y = pos_src[n["idx"]]
        ax.text(x, y-cell_shift, n["text"], ha="left", va="center",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=src_l[n["cluster"]],
                          edgecolor=src_c[n["cluster"]], linewidth=1.2),
                zorder=2)
    for n in tgt_nodes:
        x,y = pos_tgt[n["idx"]]
        ax.text(x, y-cell_shift, n["text"], ha="right", va="center",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=tgt_l[n["cluster"]],
                          edgecolor=tgt_c[n["cluster"]], linewidth=1.2),
                zorder=2)

    # — headings —
    header_off = 0.05
    for sc in SRC_CLUSTERS:
        ys = [pos_src[n["idx"]][1]-cell_shift for n in src_nodes if n["cluster"] == sc]
        if ys:
            topy = max(ys)
            xh = x_src + 0.1 + (0.06 if sc in (711,702) else 0)
            ax.text(xh, topy+header_off, cluster_headings[sc],
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    for sc in TGT_CLUSTERS:
        ys = [pos_tgt[n["idx"]][1]-cell_shift for n in tgt_nodes if n["cluster"] == sc]
        if ys:
            topy = max(ys)
            xh = x_tgt - 0.1 - (0.03 if sc == 630 else 0)
            ax.text(xh, topy+header_off, cluster_headings[sc],
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    return ax