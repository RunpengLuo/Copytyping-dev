import logging
import os
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# Silence anndata "storing X as categorical" messages
logging.getLogger("anndata").setLevel(logging.WARNING)


def set_clone_colors(adata, col, gray="#b0b0b0"):
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


def blend_clone_color_by_purity(
    adata, label_col, purity_col, out_col="clone_purity_color"
):
    """Blend clone color with gray based on tumor purity per spot.

    For each spot: color = purity * clone_color + (1 - purity) * gray.
    This shows both clone identity and tumor content in a single panel.
    """
    gray = np.array([0.69, 0.69, 0.69])  # #b0b0b0
    labels = adata.obs[label_col].astype("category")
    cats = list(labels.cat.categories)

    base = sns.color_palette("tab10", n_colors=max(len(cats) + 1, 10))
    clone_rgb = {}
    j = 0
    for c in cats:
        if c == "normal":
            clone_rgb[c] = gray
        else:
            clone_rgb[c] = np.array(base[j][:3])
            j += 1

    purity = adata.obs[purity_col].to_numpy(dtype=float)
    n = len(adata)
    rgba = np.ones((n, 4), dtype=float)
    for i in range(n):
        lab = labels.iloc[i]
        c = clone_rgb.get(lab, gray)
        rgba[i, :3] = purity[i] * c + (1 - purity[i]) * gray

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*categorical.*")
        adata.obs[out_col] = [mcolors.to_hex(rgba[i, :3]) for i in range(n)]
    return rgba


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
        set_clone_colors(vis_adata, spot_label)
        if has_path:
            set_clone_colors(vis_adata, path_label)

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
                alpha=0.5,
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
        if has_purity:
            rgba = blend_clone_color_by_purity(vis_adata, spot_label, "tumor_purity")
            ax = axes[ri, ci]
            # override PatchCollection facecolors with blended colors
            from matplotlib.collections import PatchCollection as _PC

            for child in ax.get_children():
                if isinstance(child, _PC) and len(child.get_paths()) == len(rgba):
                    child.set_facecolors(rgba)
                    break
            # replace legend with clean clone labels
            old_legend = ax.get_legend()
            if old_legend:
                old_legend.remove()
            cats = list(vis_adata.obs[spot_label].cat.categories)
            base_pal = sns.color_palette("tab10", n_colors=max(len(cats) + 1, 10))
            handles = []
            j = 0
            for c in cats:
                if c == "normal":
                    continue
                handles.append(mpatches.Patch(color=base_pal[j], label=c))
                j += 1
            ax.legend(
                handles=handles,
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                fontsize=7,
                framealpha=0.8,
            )

    # only show legend on the last column, positioned outside the plot
    for ri in range(nrows):
        for ci in range(ncols):
            legend = axes[ri, ci].get_legend()
            if legend is None:
                continue
            if ci < ncols - 1:
                legend.remove()
            else:
                legend.set_bbox_to_anchor((1.0, 0.5))
                legend._loc = 6  # center left

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


def plot_visium_iters(
    sample: str,
    slices: list,
    iter_anns: list,
    out_dir: str,
    clones: list = None,
    spot_label: str = "copytyping-label",
    ref_label=None,
    dpi=150,
    size=1.5,
):
    """Animated GIF: one frame per EM iteration, all slices side by side.

    Row 0: clone x purity blend (MAP label colored, blended with gray by theta).
    Row 1: max_posterior heatmap (confidence of MAP assignment).
    """
    import squidpy as sq
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    from copytyping.utils import is_tumor_label, NA_CELLTYPE

    niters = len(iter_anns)
    ncols = len(slices)

    # per-iteration AUC
    iter_aucs = []
    if ref_label:
        for anns_t in iter_anns:
            if ref_label not in anns_t.columns:
                continue
            known = ~anns_t[ref_label].isin(NA_CELLTYPE)
            y_true = (
                anns_t.loc[known, ref_label].apply(is_tumor_label).to_numpy(dtype=int)
            )
            if 0 < y_true.sum() < len(y_true):
                scores = anns_t.loc[known, "tumor_purity"].to_numpy()
                iter_aucs.append(roc_auc_score(y_true, scores))
            else:
                iter_aucs.append(np.nan)

    # clone colors matching blend_clone_color_by_purity
    gray = np.array([0.69, 0.69, 0.69])
    base_palette = sns.color_palette("tab10", n_colors=max(len(clones or []) + 1, 10))
    clone_rgb = {}
    j = 0
    for c in clones or []:
        if c == "normal":
            clone_rgb[c] = gray
        else:
            clone_rgb[c] = np.array(base_palette[j][:3])
            j += 1
    legend_handles = [
        mpatches.Patch(color=clone_rgb.get(c, gray), label=c) for c in (clones or [])
    ]

    col_w = 6
    row_h = 5
    label_col = "_iter_clone"
    frames = []

    for ri, anns_t in enumerate(iter_anns):
        anns_indexed = anns_t.set_index("BARCODE")
        fig, axes = plt.subplots(
            nrows=2,
            ncols=ncols,
            figsize=(col_w * ncols, row_h * 2),
            squeeze=False,
        )

        for ci, (rep_id, _anns_vis, vis_adata) in enumerate(slices):
            anns_sub = anns_indexed.reindex(vis_adata.obs_names)
            vis_adata.obs["tumor_purity"] = anns_sub["tumor_purity"].values
            vis_adata.obs["max_posterior"] = anns_sub["max_posterior"].values
            vis_adata.obs[label_col] = pd.Categorical(
                anns_sub[spot_label].values, categories=clones
            )

            # Row 0: clone x purity
            rgba = blend_clone_color_by_purity(vis_adata, label_col, "tumor_purity")
            ax0 = axes[0, ci]
            sq.pl.spatial_scatter(
                vis_adata,
                color=label_col,
                size=size,
                library_id=rep_id,
                ax=ax0,
                edgecolors="none",
            )
            from matplotlib.collections import PatchCollection as _PC

            for child in ax0.get_children():
                if isinstance(child, _PC) and len(child.get_paths()) == len(rgba):
                    child.set_facecolors(rgba)
                    break
            ax0.set_title(f"{rep_id} — clone x purity", fontsize=10)
            legend = ax0.get_legend()
            if legend is not None:
                legend.remove()

            # Row 1: max_posterior
            ax1 = axes[1, ci]
            sq.pl.spatial_scatter(
                vis_adata,
                color="max_posterior",
                size=size,
                library_id=rep_id,
                ax=ax1,
                edgecolors="none",
                cmap="magma_r",
                vmin=0,
                vmax=1,
            )
            ax1.set_title(f"{rep_id} — max posterior", fontsize=10)
            # keep colorbar only on last column
            if ci < ncols - 1:
                cb = ax1.collections[-1].colorbar if ax1.collections else None
                if cb is not None:
                    cb.remove()

        # legend on last axis of row 0
        axes[0, -1].legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=8,
        )

        title = f"{sample}  iter {ri + 1}/{niters}"
        if iter_aucs and ri < len(iter_aucs):
            title += f"  AUC={iter_aucs[ri]:.3f}"
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout()

        tmp_path = os.path.join(out_dir, f"_frame_{ri}.png")
        fig.savefig(tmp_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        frames.append(Image.open(tmp_path).convert("RGB"))
        os.remove(tmp_path)

    out_file = os.path.join(out_dir, f"{sample}.visium_iters.gif")
    frames[0].save(
        out_file,
        save_all=True,
        append_images=frames[1:],
        duration=800,
        loop=0,
    )
    logging.info(f"saved visium iterations GIF to {out_file}")


def plot_purity_histogram(
    anns: pd.DataFrame,
    sample: str,
    out_dir: str,
    spot_label: str = "copytyping-label",
    clones: list = None,
    bins=50,
    dpi=200,
):
    """Histogram of tumor_purity per clone label."""
    if "tumor_purity" not in anns.columns:
        return

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    gray = "#b0b0b0"
    base = sns.color_palette("tab10", n_colors=10).as_hex()
    clone_colors = {}
    j = 0
    for c in clones or []:
        if c == "normal":
            clone_colors[c] = gray
        else:
            clone_colors[c] = base[j]
            j += 1

    labels = anns[spot_label].values
    purity = anns["tumor_purity"].values
    unique_labels = sorted(set(labels), key=lambda x: (x != "normal", x))

    fig, ax = plt.subplots(figsize=(8, 4))
    for lab in unique_labels:
        mask = labels == lab
        vals = purity[mask]
        if len(vals) == 0:
            continue
        ax.hist(
            vals,
            bins=bins,
            range=(0, 1),
            alpha=0.6,
            label=f"{lab} (n={len(vals)})",
            color=clone_colors.get(lab, gray),
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_xlabel("Tumor purity (θ)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{sample} — purity distribution by clone", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_file = os.path.join(out_dir, f"{sample}.purity_histogram.pdf")
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"saved purity histogram to {out_file}")
