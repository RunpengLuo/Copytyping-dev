import os
import sys
import shutil
import logging

import numpy as np
import pandas as pd

import scanpy as sc

from copytyping.utils import *

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from copytyping.sx_data.sx_data import SX_Data
from copytyping.io_utils import load_visium_path_annotation
from copytyping.inference.validation import evaluate_malignant_accuracy


def set_normal_gray(adata, col, gray="#b0b0b0"):
    adata.obs[col] = adata.obs[col].astype("category")
    cats = list(adata.obs[col].cat.categories)

    base = sns.color_palette("tab10", n_colors=max(len(cats) + 1, 10)).as_hex()
    colors = []
    j = 0
    for c in cats:
        if c == "normal":
            colors.append(gray)
        else:
            colors.append(base[j])
            j += 1

    adata.uns[f"{col}_colors"] = colors


def plot_visium_panel(
    sample: str,
    slices: list,
    out_dir: str,
    spot_label="spot_label",
    path_label="Microregion_annotation",
    dpi=300,
    size=1.5,
    title_info="",
):
    """Generate a single-page PDF with all slices side by side.

    Args:
        slices: list of (rep_id, anns_df, adata) tuples.
    """
    import squidpy as sq

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    has_purity = "tumor_purity" in slices[0][1].columns
    has_path = path_label in slices[0][1].columns
    row_labels = ["H&E"]
    if has_path:
        row_labels.append(path_label)
    if has_purity:
        purity_label = f"tumor purity\n{title_info}" if title_info else "tumor purity"
        row_labels.append(purity_label)
    row_labels.append("copytyping")
    row_labels.append("max posterior")
    nrows = len(row_labels)
    ncols = len(slices)

    col_w = 6
    row_h = 6
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(col_w * ncols, row_h * nrows),
        squeeze=False,
    )

    for ci, (rep_id, anns_vis, vis_adata) in enumerate(slices):
        vis_adata.obs[spot_label] = anns_vis[spot_label].astype("category")
        if has_path:
            vis_adata.obs[path_label] = anns_vis[path_label].astype("category")
        if has_purity:
            vis_adata.obs["tumor_purity"] = anns_vis["tumor_purity"].values
        vis_adata.obs["max_posterior"] = anns_vis["max_posterior"].values
        set_normal_gray(vis_adata, spot_label)
        if has_path:
            set_normal_gray(vis_adata, path_label)

        ri = 0
        sq.pl.spatial_scatter(
            vis_adata,
            color=None,
            library_id=rep_id,
            ax=axes[ri, ci],
        )

        if has_path:
            ri += 1
            sq.pl.spatial_scatter(
                vis_adata,
                color=path_label,
                size=size,
                library_id=rep_id,
                ax=axes[ri, ci],
                edgecolors="none",
            )

        if has_purity:
            ri += 1
            sq.pl.spatial_scatter(
                vis_adata,
                color="tumor_purity",
                size=size,
                library_id=rep_id,
                cmap="magma_r",
                vmin=0,
                vmax=1,
                ax=axes[ri, ci],
                edgecolors="none",
            )

        ri += 1
        sq.pl.spatial_scatter(
            vis_adata,
            color=spot_label,
            size=size,
            library_id=rep_id,
            ax=axes[ri, ci],
            edgecolors="none",
        )

        ri += 1
        sq.pl.spatial_scatter(
            vis_adata,
            color="max_posterior",
            size=size,
            library_id=rep_id,
            cmap="magma_r",
            vmin=0,
            vmax=1,
            ax=axes[ri, ci],
            edgecolors="none",
        )

    # only show legend on the last column
    for ri in range(nrows):
        for ci in range(ncols - 1):
            legend = axes[ri, ci].get_legend()
            if legend is not None:
                legend.remove()

    # set every axes title to sample_rep_id, overriding squidpy defaults
    for ri in range(nrows):
        for ci, (rep_id, _, _) in enumerate(slices):
            axes[ri, ci].set_title(f"{sample}_{rep_id}", fontsize=10)

    # row-level suptitles via figure text, centered on each row
    fig.subplots_adjust(wspace=0.3, hspace=0.25, top=0.95)
    fig.canvas.draw()
    for ri, rlabel in enumerate(row_labels):
        row_top = axes[ri, 0].get_position().y1
        row_bot = axes[ri, 0].get_position().y0
        y_mid = (row_top + row_bot) / 2
        fig.text(
            0.02,
            y_mid,
            rlabel,
            fontsize=13,
            fontweight="bold",
            ha="center",
            va="center",
            rotation=90,
        )

    fig.suptitle(sample, fontsize=14, fontweight="bold")

    out_file = os.path.join(out_dir, f"{sample}.visium.pdf")
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"saved visium panel to {out_file}")


def plot_visium_debug(
    sample: str,
    slices: list,
    param_trace: list,
    barcodes: pd.DataFrame,
    data_type: str,
    out_dir: str,
    ref_label=None,
    dpi=200,
    size=1.5,
):
    """Debug PDF: one row per EM iteration showing tumor purity per slice.

    Args:
        slices: list of (rep_id, anns_df, adata) tuples.
        param_trace: list of param dicts, one per iteration.
        barcodes: full barcodes DataFrame (index aligned with theta arrays).
        data_type: e.g. "gex".
        ref_label: if provided, compute per-iteration AUC against this column.
    """
    import squidpy as sq
    from sklearn.metrics import roc_auc_score
    from copytyping.inference.validation import TUMOR_LABELS, UNKNOWN_LABELS

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    theta_key = f"{data_type}-theta"
    niters = len(param_trace)
    ncols = len(slices)

    # map barcode -> position in theta array
    barcode_to_idx = {bc: i for i, bc in enumerate(barcodes["BARCODE"].values)}

    # compute per-iteration AUC if ref_label available
    iter_aucs = []
    if ref_label and ref_label in barcodes.columns:
        known = ~barcodes[ref_label].isin(UNKNOWN_LABELS)
        y_true = barcodes.loc[known, ref_label].isin(TUMOR_LABELS).to_numpy(dtype=int)
        known_idx = known.to_numpy()
        for params_t in param_trace:
            theta_t = params_t[theta_key]
            scores = theta_t[known_idx]
            if np.any(np.isnan(scores)):
                iter_aucs.append(np.nan)
            else:
                iter_aucs.append(roc_auc_score(y_true, scores))

    col_w = 6
    row_h = 5
    fig, axes = plt.subplots(
        nrows=niters,
        ncols=ncols,
        figsize=(col_w * ncols, row_h * niters),
        squeeze=False,
    )

    for ri, params_t in enumerate(param_trace):
        theta_t = params_t[theta_key]
        for ci, (rep_id, anns_vis, vis_adata) in enumerate(slices):
            idx = [barcode_to_idx[bc] for bc in vis_adata.obs_names]
            vis_adata.obs["tumor_purity"] = theta_t[idx]
            sq.pl.spatial_scatter(
                vis_adata,
                color="tumor_purity",
                size=size,
                library_id=rep_id,
                cmap="magma_r",
                vmin=0,
                vmax=1,
                ax=axes[ri, ci],
                edgecolors="none",
            )
            axes[ri, ci].set_title(f"{sample}_{rep_id}")

    # row labels
    fig.subplots_adjust(wspace=0.3, hspace=0.25)
    fig.canvas.draw()
    for ri in range(niters):
        row_top = axes[ri, 0].get_position().y1
        row_bot = axes[ri, 0].get_position().y0
        y_mid = (row_top + row_bot) / 2
        rlabel = f"iter {ri}"
        if iter_aucs:
            rlabel += f"\nAUC={iter_aucs[ri]:.3f}"
        fig.text(
            0.02,
            y_mid,
            rlabel,
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            rotation=90,
        )

    # only keep legend on last column
    for ri in range(niters):
        for ci in range(ncols - 1):
            legend = axes[ri, ci].get_legend()
            if legend is not None:
                legend.remove()

    fig.suptitle(
        f"{sample} — tumor purity per iteration", fontsize=14, fontweight="bold"
    )

    out_file = os.path.join(out_dir, f"{sample}.visium_debug.pdf")
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"saved debug visium panel to {out_file}")
