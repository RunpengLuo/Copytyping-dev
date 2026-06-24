import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from copytyping.plot.plot_common import (
    FigureSaver,
    NA_COLOR,
    _clone_order_key,
    build_label_colors,
)
from copytyping.utils import NA_CELLTYPE, is_tumor_label

# perceptually-uniform sequential cmap for the [0,1] posterior probability
POSTERIOR_CMAP = "viridis"


def plot_scatter_2d_per_cell(
    read_counts: np.ndarray,
    ballele_counts: np.ndarray,
    total_allele_counts: np.ndarray,
    cn_A: np.ndarray,
    cn_B: np.ndarray,
    cn_C: np.ndarray,
    imbalanced: np.ndarray,
    aneuploid: np.ndarray,
    clones: list[str],
    cnprofile: pd.DataFrame,
    anns: pd.DataFrame,
    sample: str,
    outfile: str,
    col_label: str,
    row_label: str | None = None,
    base_props: np.ndarray | None = None,
    markersize: int = 4,
    rasterized: bool = True,
    dpi: int = 100,
    img_type: str = "pdf",
    transparent: bool = False,
    color_map: dict[str, str] | None = None,
):
    """Per-cluster 2D BAF-vs-log2RDR cross-tab grid, one page per cluster.

    Each page is a grid with ``row_label`` (e.g. cell type) as rows and
    ``col_label`` (e.g. inferred clone) as columns; each ax shows the cells in
    that (row, col) intersection, colored by the column (``color_map``). Only the
    column's own clone CN-state landmark (expected BAF/log2RDR) is drawn. All axes
    share the same scale.

    Args:
        read_counts: (G, N) read depth / feature counts.
        ballele_counts: (G, N) B-allele counts.
        total_allele_counts: (G, N) total-allele counts (A + B).
        cn_A/cn_B/cn_C: (G, K) per-clone copy numbers.
        clones: clone names, length K.
        imbalanced/aneuploid: (G,) boolean masks selecting informative rows
            (from Count_Data.allele_mask["IMBALANCED"] / total_mask["ANEUPLOID"]).
        cnprofile: per-row CNP table (CNP/LENGTH columns for titles).
        anns: per-cell annotations; holds ``col_label`` and (optional) ``row_label``.
        col_label: column label column (clone); also selects each cell's color.
        row_label: row label column (cell type); None -> single row.
        color_map: {value: color} for ``col_label`` (shared across plots).
        base_props: (G,) RDR baseline; defaults to the global read-count fraction.
    """
    cluster_indices = np.where(imbalanced | aneuploid)[0]
    col_labels_all = anns[col_label].to_numpy()
    row_labels_all = anns[row_label].to_numpy() if row_label is not None else None
    # per-cell assignment confidence (colors the points); None -> clone color
    post_all = (
        anns["max_posterior"].to_numpy() if "max_posterior" in anns.columns else None
    )

    num_clones = cn_A.shape[1]
    library_size = read_counts.sum(axis=0).astype(float)
    if base_props is None:
        base_props = read_counts.sum(axis=1) / max(library_size.sum(), 1)

    out_base = (
        outfile[:-4] if outfile.lower().endswith((".pdf", ".png", ".svg")) else outfile
    )
    with FigureSaver(out_base, img_type, dpi, transparent) as pdf:
        for g in cluster_indices:
            row = cnprofile.iloc[g]
            length_mb = row["LENGTH"] / 1e6 if "LENGTH" in row.index else np.nan
            n_bbc = int(row["#BBC"]) if "#BBC" in row.index else 0

            tag = []
            if imbalanced[g]:
                tag.append("IMB")
            if aneuploid[g]:
                tag.append("ANE")

            total_allele_g = total_allele_counts[g].astype(float)
            ballele_g = ballele_counts[g].astype(float)
            read_g = read_counts[g].astype(float)
            lam_g = base_props[g]

            valid = (total_allele_g > 0) & (library_size * lam_g > 0)
            if valid.sum() < 10:
                continue

            baf = ballele_g[valid] / total_allele_g[valid]
            rdr = read_g[valid] / (library_size[valid] * lam_g)
            log2rdr = np.log2(np.clip(rdr, 1e-6, None))
            col_cells = col_labels_all[valid]
            row_cells = row_labels_all[valid] if row_labels_all is not None else None
            post_cells = post_all[valid] if post_all is not None else None

            col_vals = sorted(set(col_cells), key=_clone_order_key)
            row_vals = (
                sorted(set(row_cells), key=_clone_order_key)
                if row_cells is not None
                else [None]
            )
            # marginal cell counts (column-sum per clone, row-sum per cell type)
            col_total = {cv: int((col_cells == cv).sum()) for cv in col_vals}
            row_total = (
                {rv: int((row_cells == rv).sum()) for rv in row_vals}
                if row_cells is not None
                else {}
            )

            def col_color(cv):
                if color_map is not None:
                    return color_map.get(str(cv), NA_COLOR)
                return build_label_colors([cv], clone_indexed=True)[0]

            # Expected (BAF, log2RDR) landmark per clone: BAF=B/(A+B),
            # log2RDR=log2(C_{g,k} / sum_g(lambda_g * C_{g,k})).
            exp_by_clone = {}
            for k in range(num_clones):
                C_k = cn_C[g, k]
                exp_baf_k = cn_B[g, k] / C_k if C_k > 0 else 0.5
                denom_k = float(np.sum(base_props * cn_C[:, k]))
                exp_log2rdr_k = (
                    np.log2(C_k / denom_k) if (denom_k > 0 and C_k > 0) else 0.0
                )
                exp_by_clone[clones[k]] = (exp_baf_k, exp_log2rdr_k)

            xlim = (-0.05, 1.05)
            # Adaptive ylim: tighten to (-2, 2) if all obs + expected fit, else (-5, 5)
            obs_finite = log2rdr[np.isfinite(log2rdr)]
            exp_finite = np.array([y for _, y in exp_by_clone.values()], dtype=float)
            all_y = np.concatenate([obs_finite, exp_finite])
            if all_y.size > 0 and all_y.min() >= -2 and all_y.max() <= 2:
                ylim = (-2, 2)
            else:
                ylim = (-5, 5)

            # Grid: rows = row_label (cell type), cols = col_label (clone). Each ax
            # holds the coinciding cells, colored by the column (clone).
            nrows, ncols = len(row_vals), len(col_vals)
            fig_h = 3.0 * nrows
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(3.0 * ncols, fig_h),
                sharex=True,
                sharey=True,
                squeeze=False,
            )
            for r, row_v in enumerate(row_vals):
                for c, col_v in enumerate(col_vals):
                    ax = axes[r][c]
                    m = col_cells == col_v
                    if row_v is not None:
                        m = m & (row_cells == row_v)
                    n_cells = int(m.sum())
                    if n_cells == 0:
                        ax.text(
                            0.5,
                            0.5,
                            "N/A",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=24,
                            fontweight="bold",
                            color="lightgray",
                        )
                    else:
                        if post_cells is not None:
                            ax.scatter(
                                baf[m],
                                log2rdr[m],
                                s=markersize * 2,
                                c=post_cells[m],
                                cmap=POSTERIOR_CMAP,
                                vmin=0.0,
                                vmax=1.0,
                                edgecolors="none",
                                rasterized=rasterized,
                                antialiased=False,
                            )
                        else:
                            ax.scatter(
                                baf[m],
                                log2rdr[m],
                                s=markersize * 2,
                                c=col_color(col_v),
                                edgecolors="none",
                                rasterized=rasterized,
                                antialiased=False,
                            )
                        ax.axvline(0.5, color="grey", linewidth=0.5)
                        ax.axhline(0.0, color="grey", linewidth=0.5)
                        if col_v in exp_by_clone:  # only this column's clone landmark
                            ax.plot(
                                *exp_by_clone[col_v],
                                marker="x",
                                color="black",
                                markersize=8,
                                markeredgewidth=2.0,
                                alpha=0.8,
                                zorder=10,
                            )
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.set_title(f"n={n_cells}", fontsize=9)
                    # FP/FN intersection (tumor cell <-> normal clone, or
                    # non-tumor cell <-> tumor clone) with cells -> dark-red border
                    row_known = row_v is not None and str(row_v) not in NA_CELLTYPE
                    if (
                        n_cells > 0
                        and row_known
                        and is_tumor_label(row_v) != is_tumor_label(col_v)
                    ):
                        for spine in ax.spines.values():
                            spine.set_edgecolor("darkred")
                            spine.set_linewidth(2.0)
                    if c == 0 and row_v is not None:
                        ax.set_ylabel(
                            f"{row_v} (n={row_total[row_v]})",
                            fontsize=11,
                            fontweight="bold",
                        )
                    if r == nrows - 1:
                        k = clones.index(col_v) if col_v in clones else None
                        cn_txt = (
                            f"\nCN={cn_A[g, k]}|{cn_B[g, k]}" if k is not None else ""
                        )
                        ax.set_xlabel(
                            f"{col_v} (n={col_total[col_v]}){cn_txt}",
                            fontsize=11,
                            fontweight="bold",
                        )

            fig.suptitle(
                f"{sample} — cluster {g} ({length_mb:.1f}Mb, {n_bbc} BBCs) — "
                f"{'/'.join(tag)}  (x=BAF, y=log2RDR)",
                fontsize=11,
                fontweight="bold",
                x=0.01,
                y=1 - 0.03 / fig_h,
                ha="left",
                va="top",
            )
            # fixed-inch top (title) / bottom (colorbar) reservations so the gaps
            # stay tight regardless of how tall the grid is
            bottom = 0.55 / fig_h if post_all is not None else 0.0
            fig.tight_layout(rect=[0, bottom, 1, 1 - 0.28 / fig_h])
            if post_all is not None:
                # shared horizontal posterior colorbar, centered below the last row
                cax = fig.add_axes([0.35, 0.26 / fig_h, 0.30, 0.10 / fig_h])
                cb = fig.colorbar(
                    ScalarMappable(Normalize(0.0, 1.0), cmap=POSTERIOR_CMAP),
                    cax=cax,
                    orientation="horizontal",
                )
                cb.set_label("max posterior", fontsize=8)
                cax.tick_params(labelsize=7)
            pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
