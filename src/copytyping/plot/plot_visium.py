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
    labeling_trace: list,
    barcodes: pd.DataFrame,
    out_dir: str,
    clones: list = None,
    ref_label=None,
    dpi=100,
    size=1.5,
):
    """PDF with one page per EM iteration from labeling_trace.

    Page 0: normal init from allele-only sub-EM (1 row).
    Pages 1+: Row 0 clone x purity, Row 1 max_posterior, Row 2 purity histogram.
    """
    import squidpy as sq
    from matplotlib.backends.backend_pdf import PdfPages
    from sklearn.metrics import roc_auc_score
    from copytyping.utils import is_tumor_label, NA_CELLTYPE

    niters = len(labeling_trace)
    ncols = len(slices)
    bc_to_idx = {bc: i for i, bc in enumerate(barcodes["BARCODE"].values)}

    # clone colors
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

    # per-iteration AUC
    iter_aucs = []
    if ref_label and ref_label in barcodes.columns:
        known = ~barcodes[ref_label].isin(NA_CELLTYPE)
        y_true = (
            barcodes.loc[known, ref_label].apply(is_tumor_label).to_numpy(dtype=int)
        )
        known_idx = known.to_numpy()
        for lt in labeling_trace:
            purity = lt.get("tumor_purity")
            if purity is not None and 0 < y_true.sum() < len(y_true):
                iter_aucs.append(roc_auc_score(y_true, purity[known_idx]))
            else:
                iter_aucs.append(np.nan)

    col_w = 6
    row_h = 5
    label_col = "_iter_clone"
    spatial_file = os.path.join(out_dir, f"{sample}.visium_iters.pdf")
    hist_file = os.path.join(out_dir, f"{sample}.iter_histograms.pdf")
    from matplotlib.collections import PatchCollection as _PC

    def _make_title(ri, labels):
        title = f"{sample}  iter {ri}/{niters - 1}"
        if ri == 0 and ref_label and ref_label in barcodes.columns:
            is_normal = labels == "normal"
            ref_normal = (
                barcodes[ref_label]
                .apply(lambda x: not is_tumor_label(x) and x not in NA_CELLTYPE)
                .to_numpy()
            )
            tp = int((is_normal & ref_normal).sum())
            fp = int((is_normal & ~ref_normal).sum())
            fn = int((~is_normal & ref_normal).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-10)
            title += f"  (init normal: prec={prec:.3f} rec={rec:.3f} f1={f1:.3f})"
        if iter_aucs and ri < len(iter_aucs):
            title += f"  AUC={iter_aucs[ri]:.3f}"
        return title

    # --- Spatial PDF (H&E + labels) ---
    with PdfPages(spatial_file) as pdf:
        for ri, lt in enumerate(labeling_trace):
            labels = lt["labels"]
            max_post = lt["max_posterior"]
            purity = lt.get("tumor_purity")
            has_purity = purity is not None

            unique_labels = sorted(set(labels))
            if has_purity:
                frame_cats = [
                    c for c in (clones or unique_labels) if c in unique_labels
                ]
                if not frame_cats:
                    frame_cats = unique_labels
            else:
                frame_cats = unique_labels

            fig, axes = plt.subplots(
                nrows=1,
                ncols=ncols,
                figsize=(col_w * ncols, row_h),
                squeeze=False,
            )
            for ci, (rep_id, _anns_vis, vis_adata_orig) in enumerate(slices):
                vis_adata = vis_adata_orig.copy()
                idx = [bc_to_idx[bc] for bc in vis_adata.obs_names]
                vis_adata.obs[label_col] = pd.Categorical(
                    labels[idx], categories=frame_cats
                )
                if has_purity:
                    vis_adata.obs["tumor_purity"] = purity[idx]

                ax0 = axes[0, ci]
                set_clone_colors(vis_adata, label_col)
                if has_purity:
                    sq.pl.spatial_scatter(
                        vis_adata,
                        color=label_col,
                        size=size,
                        library_id=rep_id,
                        ax=ax0,
                        edgecolors="none",
                    )
                    rgba = blend_clone_color_by_purity(
                        vis_adata, label_col, "tumor_purity"
                    )
                    for child in ax0.get_children():
                        if isinstance(child, _PC) and len(child.get_paths()) == len(
                            rgba
                        ):
                            child.set_facecolors(rgba)
                            break
                else:
                    sq.pl.spatial_scatter(
                        vis_adata,
                        color=label_col,
                        size=size,
                        library_id=rep_id,
                        ax=ax0,
                        alpha=0.5,
                        edgecolors="none",
                    )
                ax0.set_title(f"{rep_id}", fontsize=10)
                leg = ax0.get_legend()
                if leg is not None:
                    leg.remove()

            frame_legend = [
                mpatches.Patch(color=clone_rgb.get(c, gray), label=c)
                for c in frame_cats
            ]
            axes[0, -1].legend(
                handles=frame_legend,
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                fontsize=8,
            )
            fig.suptitle(_make_title(ri, labels), fontsize=12, fontweight="bold")
            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
    logging.info(f"saved visium iterations to {spatial_file}")

    # --- Histogram PDF (posterior + purity per iter) ---
    with PdfPages(hist_file) as pdf:
        for ri, lt in enumerate(labeling_trace):
            labels = lt["labels"]
            max_post = lt["max_posterior"]
            purity = lt.get("tumor_purity")
            if purity is None:
                continue

            unique_labels = sorted(set(labels))
            frame_cats = [c for c in (clones or unique_labels) if c in unique_labels]
            if not frame_cats:
                frame_cats = unique_labels
            active = [lab for lab in frame_cats if (labels == lab).any()]
            hist_labels = [f"{lab} (n={int((labels == lab).sum())})" for lab in active]
            hist_colors = [clone_rgb.get(lab, gray) for lab in active]

            fig, (ax_post, ax_pur) = plt.subplots(1, 2, figsize=(14, 4))

            ax_post.hist(
                [max_post[labels == lab] for lab in active],
                bins=50,
                range=(0, 1),
                stacked=True,
                label=hist_labels,
                color=hist_colors,
                edgecolor="black",
                linewidth=0.3,
            )
            ax_post.set_xlabel("max posterior", fontsize=10)
            ax_post.set_ylabel("count", fontsize=10)
            ax_post.set_title("posterior distribution", fontsize=10)
            ax_post.legend(fontsize=8)

            ax_pur.hist(
                [purity[labels == lab] for lab in active],
                bins=50,
                range=(0, 1),
                stacked=True,
                label=hist_labels,
                color=hist_colors,
                edgecolor="black",
                linewidth=0.3,
            )
            ax_pur.set_xlabel("tumor purity (θ)", fontsize=10)
            ax_pur.set_ylabel("count", fontsize=10)
            ax_pur.set_title("purity distribution", fontsize=10)
            ax_pur.legend(fontsize=8)

            fig.suptitle(_make_title(ri, labels), fontsize=12, fontweight="bold")
            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
    logging.info(f"saved iteration histograms to {hist_file}")
