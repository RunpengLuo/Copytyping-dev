"""Diagnostic plots for CNP-HMM results:

1. Cell x segment raw BAF / RDR heatmap, rows ordered by hierarchical
   clustering of the learned per-cell CN paths.
2. Learned transition rates along the genome as a state trellis (edge width /
   alpha encode the per-segment transition probability).
"""

import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm, to_rgba
from matplotlib.patches import Rectangle
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram

from copytyping.utils import read_whitelist_segments
from copytyping.plot.plot_common import build_wl_coords
from copytyping.plot.plot_heatmap import plot_heatmap
from copytyping.cnphmm_inference.cnt_clustering import (
    compute_cnt_dist,
    build_cnt_tree,
)

# discrete blue->red BAF palette (matches plot_cnv_heatmap)
_BAF_COLORS = [
    "#1f77b4",
    "#3b8bc6",
    "#67a9cf",
    "#90c4d6",
    "#b8d6da",
    "#d9d9d9",
    "#fddbc7",
    "#f4a582",
    "#d6604d",
    "#b2182b",
]

# allele-specific CN palette for the decoded CN-state heatmap. Each imbalance
# family gets its own hue so the state's character reads from color: balanced =
# grayscale (recede), pure-LOH A>B = blues / A<B = golds, non-LOH A>B = purples /
# A<B = greens, deepening with total CN.
_ASCN_PALETTE = {
    # --- 1. Balanced States (Strict Grayscale) ---
    (0, 0): "whitesmoke",
    (1, 1): "lightgray",
    (2, 2): "darkgray",
    (3, 3): "dimgray",
    (4, 4): "black",
    (5, 5): "black",
    # --- 2. Pure LOH A > B (Strictly Blues) ---
    (1, 0): "lightskyblue",
    (2, 0): "dodgerblue",
    (3, 0): "blue",
    (4, 0): "mediumblue",
    (5, 0): "darkblue",
    (6, 0): "navy",
    # --- 3. Pure LOH A < B (Strictly Yellows/Golds) ---
    (0, 1): "lightyellow",
    (0, 2): "yellow",
    (0, 3): "gold",
    (0, 4): "goldenrod",
    (0, 5): "darkgoldenrod",
    (0, 6): "#8B6508",  # Dark Ochre/Gold (strictly avoiding red)
    # --- 4. Non-LOH A > B (Strictly Purples) ---
    (2, 1): "plum",
    (3, 1): "mediumorchid",
    (3, 2): "mediumpurple",  # Closer to balanced
    (4, 1): "darkviolet",
    (4, 2): "purple",
    (5, 1): "indigo",
    # --- 5. Non-LOH A < B (Strictly Greens) ---
    (1, 2): "palegreen",
    (1, 3): "mediumseagreen",
    (2, 3): "darkseagreen",  # Closer to balanced
    (1, 4): "forestgreen",
    (2, 4): "green",
    (1, 5): "darkgreen",
}
_ASCN_DEFAULT = "magenta"  # states outside the palette (e.g. very high CN)


# Distance-based clustering (CNT matrix) is O(U^2) memory / O(U^2 G) to build;
# beyond this many unique profiles we fall back to lexsort.
_MAX_CLUSTER_TAXA = 2000


def _dendro_polylines(
    link: np.ndarray, block_sizes: np.ndarray, row_start: int, n_total: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Dendrogram link polylines whose leaves are cell *blocks* (one per unique
    profile). Each leaf is centered on its profile's block of cells (block sizes
    respected), and merge heights are normalized to ``x in [-1, 0]`` (root at
    left). Returns ``[(x, y_frac), ...]`` with ``y_frac`` the global row fraction
    in ``[0, 1]`` (so the dendrogram aligns to the heatmap rows)."""
    dd = dendrogram(link, no_plot=True)
    leaves = dd["leaves"]  # profile ids in plot (tree) order
    bs = np.asarray(block_sizes, dtype=np.float64)[leaves]  # block sizes in plot order
    cum = np.concatenate([[0.0], np.cumsum(bs)])
    centers = (cum[:-1] + cum[1:]) / 2.0  # within-rep row center per leaf
    scipy_x = 5.0 + 10.0 * np.arange(len(leaves))  # scipy leaf x-positions (plot order)
    heights = np.array([h for dc in dd["dcoord"] for h in dc])
    hmax = heights.max() if heights.size else 1.0
    polys = []
    for xs, ys in zip(dd["icoord"], dd["dcoord"]):
        yrow = np.interp(xs, scipy_x, centers)  # within-rep row positions
        y_frac = (row_start + yrow) / n_total
        x = -(np.asarray(ys) / hmax)
        polys.append((x, y_frac))
    return polys


def _order_within(
    Z_sub: np.ndarray,
    cn_states: np.ndarray,
    chrom: np.ndarray,
    method: str,
    max_taxa: int,
) -> tuple[np.ndarray, str | None, tuple | None]:
    """Row permutation of ``Z_sub`` grouping similar cells, the Newick tree (or
    ``None``), and a ``(linkage, block_sizes)`` dendrogram bundle (or ``None``).
    ``method='lexsort'`` orders by the decoded path directly (no tree). The
    distance methods (``cnt_nj`` / ``cnt_upgma`` / ``cnt_complete``) build a tree on the
    per-chromosome CNT distance between the *unique* decoded profiles and order
    cells by its leaf order — only when ``U <= max_taxa`` (else lexsort). The
    dendrogram bundle is present only for the agglomerative methods (which have a
    linkage); ``cnt_nj`` has a Newick tree but no dendrogram."""
    if method == "lexsort":
        return np.lexsort(Z_sub.T[::-1]), None, None
    unique_paths, inverse = np.unique(Z_sub, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    U = unique_paths.shape[0]
    if U <= 2:
        order = np.concatenate([np.where(inverse == u)[0] for u in range(U)])
        return order, None, None
    if U <= max_taxa:
        A = cn_states[unique_paths, 0]  # (U, G)
        B = cn_states[unique_paths, 1]
        dist = compute_cnt_dist(A, B, chrom)
        newick, leaf, link = build_cnt_tree(dist, method)
        order = np.concatenate([np.where(inverse == u)[0] for u in leaf])
        dendro = None if link is None else (link, np.bincount(inverse, minlength=U))
        return order, newick, dendro
    return np.lexsort(Z_sub.T[::-1]), None, None


def compute_cell_order(
    Z: np.ndarray,
    rep_id: np.ndarray,
    cn_states: np.ndarray,
    chrom: np.ndarray,
    method: str = "cnt_nj",
    max_taxa: int = _MAX_CLUSTER_TAXA,
) -> tuple[np.ndarray, np.ndarray, dict[str, str], list]:
    """Row order for the cell heatmaps: **partition by REP_ID first**, then order
    cells within each rep by ``method`` (``lexsort`` | ``cnt_nj`` | ``cnt_upgma`` |
    ``cnt_complete``) over the unique decoded profiles, falling back to lexsort when there
    are too many unique profiles. ``chrom`` is the ``(G,)`` per-segment chromosome
    label. Returns ``(order, cell_labels, trees, dendro_polys)``: ``cell_labels``
    the per-cell REP_ID aligned to ``order`` (contiguous blocks per rep → REP_ID
    y-labels); ``trees`` mapping REP_ID → Newick; and ``dendro_polys`` the
    block-leaf dendrogram link polylines (global row fractions) for the
    agglomerative methods. Computed once and shared across all cell heatmaps so
    the BAF / RDR / decoded panels use an identical row order.
    """
    rep_id = np.asarray(rep_id)
    n_total = rep_id.shape[0]
    parts, used, trees, dendro_polys = [], [], {}, []
    row = 0
    for rep in pd.unique(rep_id):
        idx = np.flatnonzero(rep_id == rep)
        sub_order, newick, dendro = _order_within(
            Z[idx], cn_states, chrom, method, max_taxa
        )
        parts.append(idx[sub_order])
        u = np.unique(Z[idx], axis=0).shape[0]
        used.append(f"{rep}:{method if newick is not None else 'lexsort'}({u}u)")
        if newick is not None:
            trees[str(rep)] = newick
        if dendro is not None:
            link, bs = dendro
            dendro_polys.extend(_dendro_polylines(link, bs, row, n_total))
        row += idx.size
    order = np.concatenate(parts)
    logging.info(f"plot: cell order partitioned by REP_ID -> {', '.join(used)}")
    return order, rep_id[order], trees, dendro_polys


def _draw_dendrogram(ax: plt.Axes, polys: list, height: float) -> None:
    """Draw block-leaf dendrogram link polylines on a left axis (rows on y,
    normalized merge height on x), aligned to a heatmap of total ``height``. All
    links are drawn as a single LineCollection (cheap even for thousands)."""
    segs = [np.column_stack([x, height * np.asarray(y_frac)]) for x, y_frac in polys]
    ax.add_collection(LineCollection(segs, colors="black", linewidths=0.4))
    ax.set_xlim(-1.05, 0.05)
    ax.set_ylim(0, height)
    ax.axis("off")


def _draw_cn_legend_bottom(
    ax: plt.Axes,
    cn_states: np.ndarray,
    c_max: int | None = None,
    yc: float = -0.12,
    box_w: float = 0.018,
    fontsize: float = 8.0,
) -> None:
    """CN-state color key below the heatmap, in three left-to-right groups:

      1. **Balanced** (``(0,0),(1,1),(2,2),...``) — one centered row.
      2. **Pure LOH** — a 2-row block: top row A>B ``(1,0),(2,0),...``, bottom row
         A<B ``(0,1),(0,2),...`` (column ``k`` pairs ``(k,0)`` over ``(0,k)``).
      3. **Non-LOH** — a 2-row block: top row A>B ``(2,1),(3,1),(3,2),(4,1),...``,
         bottom row the mirror A<B ``(1,2),(1,3),(2,3),(1,4),...``.

    Swatches **touch** within a group (a gap separates groups); top-row labels sit
    above and bottom-/center-row labels below, like x-ticks. Shows every palette
    state plus any decoded state outside it (default color), capped at total CN
    ``c_max``. ``yc`` is the vertical center axes-fraction (negative = below the
    axis); ``box_w`` the swatch width in axes-x fraction."""
    states = set(_ASCN_PALETTE) | {(int(a), int(b)) for a, b in cn_states}
    if c_max is not None:
        states = {s for s in states if sum(s) <= c_max}

    balanced = sorted((s for s in states if s[0] == s[1]), key=sum)
    loh_top = sorted((s for s in states if s[1] == 0 and s[0] >= 1), key=lambda s: s[0])
    non_top = sorted(
        (s for s in states if s[0] > s[1] >= 1), key=lambda s: (sum(s), s[0])
    )
    # columns per group: (top_state, bottom_state, center_state), any may be None
    g1 = [(None, None, s) for s in balanced]
    g2 = [((k, 0), (0, k) if (0, k) in states else None, None) for k, _ in loh_top]
    g3 = [((a, b), (b, a) if (b, a) in states else None, None) for a, b in non_top]
    groups = [g for g in (g1, g2, g3) if g]

    # square swatches: convert box_w (axes-x) to an equal-size box_h (axes-y)
    fig = ax.figure
    bbox = ax.get_position()
    fw, fh = fig.get_size_inches()
    box_h = box_w * (bbox.width * fw) / (bbox.height * fh)
    gap = box_w * 0.8  # between-group gap; touching within a group
    pad = 0.010  # label gap from swatch edge

    # x left edges: touching within a group, gap between groups
    cols, cursor = [], 0.0
    for gi, g in enumerate(groups):
        if gi > 0:
            cursor += gap
        for top, bot, ctr in g:
            cols.append((cursor, top, bot, ctr))
            cursor += box_w
    shift = 0.5 - cursor / 2.0  # center the whole key at x=0.5

    def swatch(left: float, ybot: float, ab: tuple[int, int]) -> None:
        col = _ASCN_PALETTE.get(ab, _ASCN_DEFAULT)
        ax.add_patch(
            Rectangle(
                (left, ybot),
                box_w,
                box_h,
                transform=ax.transAxes,
                facecolor=col,
                edgecolor="gray",
                linewidth=0.3,
                clip_on=False,
                zorder=5,
            )
        )

    def label(xc: float, yt: float, ab: tuple[int, int], va: str) -> None:
        ax.text(
            xc,
            yt,
            f"{ab[0]}|{ab[1]}",
            transform=ax.transAxes,
            ha="center",
            va=va,
            fontsize=fontsize,
            zorder=6,
            clip_on=False,
        )

    ax.text(
        shift - gap,
        yc,
        "A|B",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=fontsize + 2,
        fontweight="bold",
        clip_on=False,
    )
    for left, top, bot, ctr in cols:
        x = left + shift
        xc = x + box_w / 2.0
        if ctr is not None:  # single centered row (balanced)
            swatch(x, yc - box_h / 2.0, ctr)
            label(xc, yc - box_h / 2.0 - pad, ctr, "top")
        if top is not None:  # top row sits above yc; label above it
            swatch(x, yc, top)
            label(xc, yc + box_h + pad, top, "bottom")
        if bot is not None:  # bottom row sits below yc; label below it
            swatch(x, yc - box_h, bot)
            label(xc, yc - box_h - pad, bot, "top")


def _heatmap_page(
    pdf: PdfPages,
    genome_coords: pd.DataFrame,
    wl_segments: pd.DataFrame,
    mat: np.ndarray,
    cell_labels: np.ndarray,
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
    title: str,
    dendro_polys: list | None,
    dpi: int,
    subtitle: str = "",
    colorbar: bool = False,
    cbar_label: str = "",
    legend_fn=None,
) -> None:
    """Render one cells x segments heatmap as a single PDF page. A fixed axes
    rectangle (identical across pages, with a reserved right strip for the
    legend) keeps all panels the same width. Continuous panels get a horizontal
    colorbar inset (labeled ``cbar_label``); categorical panels get ``legend_fn``
    drawn in the right strip; a left block-leaf dendrogram is added when
    ``dendro_polys`` is given. The REP_ID/proportion y-labels sit on the right,
    rotated 45 deg like the chromosome labels."""
    fig = plt.figure(figsize=(24, 9))
    if dendro_polys:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 20], wspace=0.005)
        _draw_dendrogram(fig.add_subplot(gs[0, 0]), dendro_polys, height=10)
        ax = fig.add_subplot(gs[0, 1])
    else:
        ax = fig.add_subplot(1, 1, 1)
    # fixed margins -> identical heatmap width on every page; the top margin
    # leaves room for the (top) chromosome labels below the suptitle, the bottom
    # margin for the colorbar / CN legend.
    fig.subplots_adjust(left=0.035, right=0.92, top=0.88, bottom=0.16)
    plot_heatmap(
        ax,
        cell_labels,
        genome_coords,
        mat,
        wl_segments,
        height=10,
        cmap=cmap,
        norm=norm,
        title=None,
        ylabel=None,
    )
    fig.suptitle(title, y=0.985, fontsize=14, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=10)
    ax.yaxis.tick_right()  # REP_ID/proportion labels on the right (dendrogram is left)
    # rotate 45 deg but anchor each label at its block center (rotation_mode=anchor
    # keeps the tick point as the alignment origin, so the label stays on its block)
    for lbl in ax.get_yticklabels():
        lbl.set_rotation(45)
        lbl.set_rotation_mode("anchor")
        lbl.set_horizontalalignment("left")
        lbl.set_verticalalignment("center")
    if colorbar:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cax = ax.inset_axes([0.0, -0.085, 0.30, 0.022])  # inset -> ax width unchanged
        fig.colorbar(sm, cax=cax, orientation="horizontal", label=cbar_label)
    if legend_fn is not None:
        legend_fn(ax)
    pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_cell_seg_heatmaps(
    genome_coords: pd.DataFrame,
    B: sparse.csr_matrix,
    C: sparse.csr_matrix,
    X: sparse.csr_matrix,
    T: np.ndarray,
    H: np.ndarray,
    base_props: np.ndarray,
    Z: np.ndarray,
    cn_states: np.ndarray,
    order: np.ndarray,
    cell_labels: np.ndarray,
    region_bed: str,
    sample: str,
    out_dir: str,
    out_prefix: str,
    dpi: int = 200,
    dendro_polys: list | None = None,
    c_max: int | None = None,
    seg_info: str = "",
) -> None:
    """Three-page cells x segments heatmap PDF: (1) phased BAF, (2) raw log2RDR,
    (3) decoded CN state. All three share the same row ``order`` (REP_ID-
    partitioned, then path-clustered) and ``cell_labels`` (REP_ID block labels);
    a left block-leaf dendrogram is drawn when ``dendro_polys`` is given. The BAF
    and RDR pages carry a horizontal colorbar; the decoded page carries the
    allele-specific CN pyramid legend.

    The decoded global phasing ``H`` is applied to the BAF before plotting:
    segments with ``H_g = 0`` are flipped (``BAF -> 1 - BAF``), so the panel
    shows the BAF in the model's resolved orientation rather than the raw
    bulk-phased ``B / C``.
    """
    wl_segments = read_whitelist_segments(region_bed)
    Bd = np.asarray(B.toarray(), dtype=np.float64)
    Cd = np.asarray(C.toarray(), dtype=np.float64)
    Xd = np.asarray(X.toarray(), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        baf = np.where(Cd > 0, Bd / Cd, np.nan).T  # (N, G)
        rdr = (
            Xd
            / np.clip(T[None, :], 1e-9, None)
            / np.clip(base_props[:, None], 1e-12, None)
        )
        log2rdr = np.log2(np.clip(rdr, 1e-6, None)).T  # (N, G)
    # apply decoded global phasing: flip BAF at segments with H_g = 0 (NaN-safe)
    flip = np.asarray(H) == 0
    baf[:, flip] = 1.0 - baf[:, flip]
    baf = baf[order]
    log2rdr = log2rdr[order]

    baf_cmap = mcolors.ListedColormap(_BAF_COLORS, name="baf_disc")
    baf_cmap.set_bad("white")
    baf_norm = mcolors.BoundaryNorm(np.linspace(0, 1, 11), baf_cmap.N, clip=True)
    rdr_cmap = plt.get_cmap("coolwarm").copy()
    rdr_cmap.set_bad("white")
    rdr_norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    K = cn_states.shape[0]
    cn_colors = [
        _ASCN_PALETTE.get((int(a), int(b)), _ASCN_DEFAULT) for a, b in cn_states
    ]
    cn_cmap = mcolors.ListedColormap(cn_colors, name="cn_states")
    cn_cmap.set_bad("white")
    cn_norm = mcolors.BoundaryNorm(np.arange(K + 1) - 0.5, K)

    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"{out_prefix}.cnphmm.cell_baf_rdr.pdf")
    with PdfPages(out_path) as pdf:
        _heatmap_page(
            pdf,
            genome_coords,
            wl_segments,
            baf,
            cell_labels,
            baf_cmap,
            baf_norm,
            f"{sample} BAF Heatmap",
            dendro_polys,
            dpi,
            subtitle=seg_info,
            colorbar=True,
            cbar_label="BAF",
        )
        _heatmap_page(
            pdf,
            genome_coords,
            wl_segments,
            log2rdr,
            cell_labels,
            rdr_cmap,
            rdr_norm,
            f"{sample} log2RDR Heatmap",
            dendro_polys,
            dpi,
            subtitle=seg_info,
            colorbar=True,
            cbar_label="log2RDR",
        )
        _heatmap_page(
            pdf,
            genome_coords,
            wl_segments,
            Z[order].astype(float),
            cell_labels,
            cn_cmap,
            cn_norm,
            f"{sample} decoded CNA state Heatmap",
            dendro_polys,
            dpi,
            subtitle=seg_info,
            legend_fn=lambda ax: _draw_cn_legend_bottom(ax, cn_states, c_max=c_max),
        )
    logging.info(f"saved cell BAF/RDR/decoded heatmaps (3 pages) to {out_path}")


def plot_transition_trellis(
    genome_coords: pd.DataFrame,
    A: np.ndarray,
    cn_states: np.ndarray,
    region_bed: str,
    sample: str,
    out_dir: str,
    out_prefix: str,
    dpi: int = 200,
    prob_thresh: float = 0.02,
    out_name: str = "transition_trellis",
    title: str | None = None,
    linewidth: float = 2.5,
    node_size: float = 8.0,
    gamma_exp: float = 2.5,
) -> None:
    """Plot per-segment transition rates as a state trellis, **one chromosome per
    PDF page**: x = that chromosome's segments, y = CN states, one edge per
    ``c -> c'`` within-chromosome transition. Probability is encoded by color
    (Reds colorbar, white at p->0) and, to keep the many low-probability edges
    from summing into dense blocks, by a **non-linear** opacity and width:
    ``alpha = p**gamma_exp`` and ``lw = linewidth * p**gamma_exp`` (``gamma_exp``
    in [2, 3]). Strongest edges are drawn on top; transitions below ``prob_thresh``
    are dropped. Used for both the learned ``A`` and the pre-EM prior mean (via
    ``out_name`` / ``title``).
    """
    chrom = genome_coords["#CHR"].to_numpy()
    starts = genome_coords["START"].to_numpy()
    ends = genome_coords["END"].to_numpy()
    K = cn_states.shape[0]
    y = np.arange(K)
    cmap = plt.get_cmap("Reds")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    ytick_labels = [f"{int(a)}|{int(b)}" for a, b in cn_states]

    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"{out_prefix}.cnphmm.{out_name}.pdf")
    n_pages = 0
    with PdfPages(out_path) as pdf:
        for ch in pd.unique(chrom):
            seg_idx = np.flatnonzero(chrom == ch)
            # use real per-segment midpoints within the chromosome for x
            xseg = (starts[seg_idx] + ends[seg_idx]) / 2.0

            # within-chromosome transitions (segments are contiguous in index)
            edges, probs = [], []
            for j in range(seg_idx.size - 1):
                g = seg_idx[j]
                x0, x1 = xseg[j], xseg[j + 1]
                Ag = A[g]
                for c in range(K):
                    for cp in range(K):
                        p = float(Ag[c, cp])
                        if p < prob_thresh:
                            continue
                        edges.append([(x0, y[c]), (x1, y[cp])])
                        probs.append(p)
            order = np.argsort(probs)
            edges = [edges[i] for i in order]
            probs = np.asarray(probs)[order] if len(probs) else np.zeros(0)
            scaled = np.clip(probs, 0.0, 1.0) ** gamma_exp
            colors = [to_rgba(cmap(norm(p)), float(s)) for p, s in zip(probs, scaled)]

            fig, ax = plt.subplots(
                figsize=(max(8, min(22, seg_idx.size * 0.4)), 0.4 * K + 2)
            )
            ax.add_collection(
                LineCollection(edges, linewidths=linewidth * scaled, colors=colors)
            )
            node_x = np.repeat(xseg, K)
            node_y = np.tile(y, seg_idx.size)
            ax.scatter(
                node_x, node_y, s=node_size, c="black", zorder=3, edgecolors="none"
            )
            ax.set_xlim(xseg.min(), xseg.max())
            ax.set_ylim(-1, K)
            ax.set_yticks(y)
            ax.set_yticklabels(ytick_labels, fontsize=8)
            ax.set_ylabel("CN state (A|B)")
            ax.set_xlabel(f"{ch} position (bp)")
            base = title or f"{sample} learned CN-state transition rates"
            ax.set_title(f"{base} — {ch} ({seg_idx.size} segments)")
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            fig.colorbar(
                sm, ax=ax, fraction=0.02, pad=0.01, label="transition probability"
            )
            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            n_pages += 1
    logging.info(f"saved transition trellis ({n_pages} chrom pages) to {out_path}")


def plot_transition_entropy(
    genome_coords: pd.DataFrame,
    seg_entropy_prior: np.ndarray,
    seg_entropy_learned: np.ndarray,
    entropy_hist: np.ndarray,
    region_bed: str,
    sample: str,
    out_dir: str,
    out_prefix: str,
    dpi: int = 200,
) -> None:
    """Transition-entropy diagnostics (both occupancy-weighted, normalized to
    [0, 1]):

      * top — per-segment transition entropy along the genome, prior vs learned.
        Where the learned curve drops below the prior, the data has resolved
        (converged) that segment's transition; segments staying at the prior
        level are uninformative. High learned entropy flags boundaries where
        cells disagree (subclonal / uncertain).
      * bottom — genome-mean transition entropy per EM iteration; the plateau is
        the convergence signal.
    """
    wl_segments = read_whitelist_segments(region_bed)
    wl = build_wl_coords(genome_coords, wl_segments)
    x = wl["positions"][:-1]  # one point per segment->segment transition

    fig, axes = plt.subplots(
        2, 1, figsize=(20, 7), gridspec_kw={"height_ratios": [3, 1]}
    )
    ax = axes[0]
    ax.plot(x, seg_entropy_prior, color="gray", lw=0.8, label="prior")
    ax.plot(x, seg_entropy_learned, color="#b2182b", lw=0.9, label="learned")
    for v in wl["chr_vlines"]:
        ax.axvline(v, color="black", linewidth=0.5)
    ax.set_xlim(0, wl["chr_end"])
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("norm. transition entropy")
    ax.set_xticks(wl["xtick_chrs"])
    ax.set_xticklabels(wl["xlab_chrs"], rotation=60, fontsize=9, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"{sample} per-segment transition entropy (prior vs learned)")

    ax2 = axes[1]
    it = np.arange(len(entropy_hist))
    ax2.plot(it, entropy_hist, marker="o", ms=3, color="#1f77b4")
    ax2.set_xlabel("EM iteration")
    ax2.set_ylabel("mean entropy")
    ax2.set_title("convergence")
    fig.tight_layout()

    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"{out_prefix}.cnphmm.transition_entropy.pdf")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"saved transition entropy diagnostics to {out_path}")


def _state_lines(
    ax: plt.Axes,
    hist: np.ndarray,
    cn_states: np.ndarray,
    ylabel: str,
    title: str,
) -> None:
    """Plot one per-iteration trajectory per CN state (``hist`` is ``(n_iter, K)``),
    each line colored by the allele-specific palette and labeled ``A|B``."""
    it = np.arange(hist.shape[0])
    for k in range(hist.shape[1]):
        a, b = int(cn_states[k, 0]), int(cn_states[k, 1])
        color = _ASCN_PALETTE.get((a, b), _ASCN_DEFAULT)
        ax.plot(
            it, hist[:, k], lw=1.0, marker="o", ms=2.5, color=color, label=f"{a}|{b}"
        )
    ax.set_xlabel("EM iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=6,
        ncol=2,
        handlelength=1.0,
        title="A|B",
    )


def plot_training_diagnostics(
    ll_hist: np.ndarray,
    entropy_hist: np.ndarray,
    pi_hist: np.ndarray,
    tau_hist: np.ndarray,
    invphi_hist: np.ndarray,
    phase_frac_hist: np.ndarray,
    cn_states: np.ndarray,
    obj_label: str,
    sample: str,
    out_dir: str,
    out_prefix: str,
    dpi: int = 200,
) -> None:
    """Per-iteration training-diagnostics PDF, one page per quantity, so you can
    see whether the model trained: (1) the training objective + its per-iteration
    change (the convergence slope -> 0 = converged); (2) mean transition entropy
    and phasing stability; (3) pi, (4) BB tau, (5) NB inv-phi per CN state."""
    ll = np.asarray(ll_hist, dtype=np.float64)
    it = np.arange(ll.shape[0])
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"{out_prefix}.cnphmm.training.pdf")

    with PdfPages(out_path) as pdf:
        # page 1 — objective + per-iteration change (convergence slope)
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        axes[0].plot(it, ll, marker="o", ms=4, color="#1f77b4")
        axes[0].set_ylabel(obj_label)
        axes[0].set_title(f"{sample} training objective")
        d = np.abs(np.diff(ll))
        axes[1].semilogy(
            it[1:], np.clip(d, 1e-12, None), marker="o", ms=4, color="#b2182b"
        )
        axes[1].set_xlabel("EM iteration")
        axes[1].set_ylabel("|change| per iter (log)")
        axes[1].set_title("convergence slope (-> 0 when trained)")
        fig.tight_layout()
        pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # page 2 — transition entropy + phasing stability
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        axes[0].plot(it, np.asarray(entropy_hist), marker="o", ms=4, color="#1f77b4")
        axes[0].set_ylabel("mean transition entropy")
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].set_title(f"{sample} transition entropy")
        axes[1].plot(it, np.asarray(phase_frac_hist), marker="o", ms=4, color="#1f77b4")
        axes[1].set_xlabel("EM iteration")
        axes[1].set_ylabel("frac. segments H=1")
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].set_title("phasing stability (share kept in bulk orientation)")
        fig.tight_layout()
        pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # pages 3-5 — per-state parameter trajectories
        for hist, ylabel, title in (
            (np.asarray(pi_hist), "pi (initial-state prob)", f"{sample} pi per state"),
            (np.asarray(tau_hist), "tau (BB dispersion)", f"{sample} BB tau per state"),
            (
                np.asarray(invphi_hist),
                "inv-phi (NB dispersion)",
                f"{sample} NB inv-phi per state",
            ),
        ):
            fig, ax = plt.subplots(figsize=(9, 5))
            _state_lines(ax, hist, cn_states, ylabel, title)
            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
    logging.info(f"saved training diagnostics ({5} pages) to {out_path}")
