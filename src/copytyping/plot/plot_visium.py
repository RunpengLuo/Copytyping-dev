import logging
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from copytyping.utils import is_tumor_label, INVALID_LABELS, NA_CELLTYPE

logging.getLogger("anndata").setLevel(logging.WARNING)

# ── Unified color palette ──
# normal=gray, NA=tab10[0], tumor clones=tab10[1:]
NORMAL_COLOR = "lightgray"
NA_COLOR = "darkgray"
_TUMOR_COLORS = sns.color_palette("tab10", 10).as_hex()


def _label_color_index(label):
    """Stable color index for a label. clone1→0, clone2→1, etc. Others by sorted name."""
    import re

    m = re.match(r"clone(\d+)", label)
    if m:
        return int(m.group(1)) - 1
    return hash(label) % len(_TUMOR_COLORS)


def build_label_colors(categories: list, clone_indexed=True) -> list:
    """Return color list for categories. Consistent regardless of subset/order.

    - INVALID_LABELS → NA_COLOR
    - "normal" → NORMAL_COLOR
    - clone_indexed=True: "cloneN" → stable tab10 index by N, others by hash
    - clone_indexed=False: non-normal labels get sequential distinct colors
    """
    colors = []
    tumor_i = 0
    for c in categories:
        if c in INVALID_LABELS:
            colors.append(NA_COLOR)
        elif c == "normal":
            colors.append(NORMAL_COLOR)
        elif clone_indexed:
            colors.append(_TUMOR_COLORS[_label_color_index(c) % len(_TUMOR_COLORS)])
        else:
            colors.append(_TUMOR_COLORS[tumor_i % len(_TUMOR_COLORS)])
            tumor_i += 1
    return colors


def set_label_colors(adata, col, clone_indexed=True):
    """Set adata.uns colors using unified scheme."""
    adata.obs[col] = adata.obs[col].astype("category")
    adata.uns[f"{col}_colors"] = build_label_colors(
        list(adata.obs[col].cat.categories), clone_indexed=clone_indexed
    )


def build_legend(categories: list) -> list:
    """Build legend handles using unified colors."""
    colors = build_label_colors(categories)
    return [mpatches.Patch(color=colors[i], label=c) for i, c in enumerate(categories)]


def blend_purity_rgba(adata, label_col, purity_col):
    """Blend label color with gray by purity. Returns (N, 4) RGBA."""
    gray = np.array(mcolors.to_rgb(NORMAL_COLOR))
    labels = adata.obs[label_col].astype("category")
    cats = list(labels.cat.categories)
    colors = build_label_colors(cats)
    cat_rgb = {c: np.array(mcolors.to_rgb(colors[i])) for i, c in enumerate(cats)}

    raw_purity = adata.obs[purity_col].to_numpy(dtype=float)
    na_mask = np.isnan(raw_purity)
    purity = np.clip(np.nan_to_num(raw_purity, nan=0.0), 0.0, 1.0)
    n = len(adata)
    rgba = np.ones((n, 4), dtype=float)
    for i in range(n):
        c = cat_rgb.get(labels.iloc[i], gray)
        rgba[i, :3] = purity[i] * c + (1 - purity[i]) * gray
    rgba[na_mask, 3] = 0.0  # NA purity → transparent
    # NA labels → transparent
    na_labels = labels.isin(["NA", "Unknown"])
    rgba[na_labels.values, 3] = 0.0
    return rgba


# ── Plot functions ──


def plot_visium_panel(
    sample: str,
    slices: list,
    out_dir: str,
    spot_label="spot_label",
    path_label="Microregion_annotation",
    best_cutoff_label=None,
    best_cutoff_metrics=None,
    dpi=300,
    size=1.5,
    alpha=0.8,
    title_info="",
):
    """Single-page PDF visium panel per slice.

    Rows: H&E, path_label, ref purity, inferred purity, clone x purity,
          [best cutoff clone x purity], CQ.

    best_cutoff_label: column name in anns for best (pcut, post) cutoff labels.
    best_cutoff_metrics: dict with ARI_clone, f1 for the row title.
    """
    import squidpy as sq
    from matplotlib.collections import PatchCollection as _PC

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    has_path = path_label in slices[0][1].columns
    ref_purity_col = f"{path_label}-tumor_purity"
    has_ref_purity = ref_purity_col in slices[0][1].columns
    has_cq = "CQ" in slices[0][1].columns

    row_labels = ["H&E"]
    if has_path:
        row_labels.append(path_label)
    if has_ref_purity:
        row_labels.append("ref purity")
    purity_label = f"tumor purity\n{title_info}" if title_info else "tumor purity"
    row_labels.append(purity_label)
    clone_purity_ri = len(row_labels)
    row_labels.append("clone x purity")
    has_best_cutoff = (
        best_cutoff_label is not None
        and best_cutoff_label in slices[0][1].columns
        and best_cutoff_label != spot_label
    )
    best_cutoff_ri = None
    if has_best_cutoff:
        best_cutoff_ri = len(row_labels)
        m = best_cutoff_metrics or {}
        ari = m.get("ARI_clone", float("nan"))
        f1 = m.get("f1", float("nan"))
        # Extract cutoff values from label name (e.g. "..._pcut0.5_post0.8")
        cutoff_parts = []
        if "_pcut" in best_cutoff_label:
            pc = best_cutoff_label.split("_pcut")[1].split("_")[0]
            cutoff_parts.append(f"pcut={pc}")
        if "_post" in best_cutoff_label:
            pt = best_cutoff_label.split("_post")[1].split("_")[0]
            cutoff_parts.append(f"post={pt}")
        cutoff_str = ", ".join(cutoff_parts) if cutoff_parts else best_cutoff_label
        row_labels.append(f"{cutoff_str}\nARI={ari:.3f} F1={f1:.3f}")
    if has_cq:
        row_labels.append("CQ score")

    nrows = len(row_labels)
    ncols = len(slices)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 6 * nrows),
        squeeze=False,
    )

    for ci, (rep_id, anns_vis, vis_adata) in enumerate(slices):
        vis_adata.obs[spot_label] = anns_vis[spot_label].astype("category")
        set_label_colors(vis_adata, spot_label)
        if has_path:
            vis_adata.obs[path_label] = anns_vis[path_label].astype("category")
            set_label_colors(vis_adata, path_label, clone_indexed=False)
        if has_ref_purity:
            vis_adata.obs[ref_purity_col] = anns_vis[ref_purity_col].values
        vis_adata.obs["tumor_purity"] = anns_vis["tumor_purity"].values

        ri = 0
        # 1. H&E
        sq.pl.spatial_scatter(vis_adata, color=None, library_id=rep_id, ax=axes[ri, ci])

        # 2. path_label (no H&E)
        if has_path:
            ri += 1
            sq.pl.spatial_scatter(
                vis_adata,
                color=path_label,
                size=size,
                library_id=rep_id,
                ax=axes[ri, ci],
                img=False,
                alpha=alpha,
                edgecolors="none",
            )

        # 3. ref purity (H&E)
        if has_ref_purity:
            ri += 1
            sq.pl.spatial_scatter(
                vis_adata,
                color=ref_purity_col,
                size=size,
                library_id=rep_id,
                ax=axes[ri, ci],
                cmap="magma_r",
                vmin=0,
                vmax=1,
                edgecolors="none",
            )

        # 4. inferred purity (H&E)
        ri += 1
        sq.pl.spatial_scatter(
            vis_adata,
            color="tumor_purity",
            size=size,
            library_id=rep_id,
            ax=axes[ri, ci],
            cmap="magma_r",
            vmin=0,
            vmax=1,
            edgecolors="none",
        )

        # 5. clone x purity (no H&E) — exclude NA spots
        ri += 1
        keep5 = vis_adata.obs[spot_label] != "NA"
        sub5 = vis_adata[keep5].copy()
        sub5.obs[spot_label] = sub5.obs[spot_label].cat.remove_unused_categories()
        set_label_colors(sub5, spot_label)
        sq.pl.spatial_scatter(
            sub5,
            color=spot_label,
            size=size,
            library_id=rep_id,
            ax=axes[ri, ci],
            img=False,
            edgecolors="none",
        )
        rgba = blend_purity_rgba(sub5, spot_label, "tumor_purity")
        for child in axes[ri, ci].get_children():
            if isinstance(child, _PC) and len(child.get_paths()) == len(rgba):
                child.set_facecolors(rgba)
                break
        old_legend = axes[ri, ci].get_legend()
        if old_legend:
            old_legend.remove()

        # 6. Best cutoff clone x purity (no H&E) — exclude NA spots
        if has_best_cutoff:
            ri += 1
            vis_adata.obs[best_cutoff_label] = anns_vis[best_cutoff_label].astype(
                "category"
            )
            keep6 = vis_adata.obs[best_cutoff_label] != "NA"
            sub6 = vis_adata[keep6].copy()
            sub6.obs[best_cutoff_label] = sub6.obs[
                best_cutoff_label
            ].cat.remove_unused_categories()
            set_label_colors(sub6, best_cutoff_label)
            sq.pl.spatial_scatter(
                sub6,
                color=best_cutoff_label,
                size=size,
                library_id=rep_id,
                ax=axes[ri, ci],
                img=False,
                edgecolors="none",
            )
            rgba = blend_purity_rgba(sub6, best_cutoff_label, "tumor_purity")
            for child in axes[ri, ci].get_children():
                if isinstance(child, _PC) and len(child.get_paths()) == len(rgba):
                    child.set_facecolors(rgba)
                    break
            old_legend = axes[ri, ci].get_legend()
            if old_legend:
                old_legend.remove()

        # 7. CQ score (no H&E)
        if has_cq:
            ri += 1
            cq_vals = anns_vis["CQ"].values.copy().astype(float)
            cq_vals = np.clip(cq_vals, 0, 30)
            vis_adata.obs["CQ_clipped"] = cq_vals
            sq.pl.spatial_scatter(
                vis_adata,
                color="CQ_clipped",
                size=size,
                library_id=rep_id,
                ax=axes[ri, ci],
                img=False,
                alpha=alpha,
                edgecolors="none",
                cmap="RdYlGn",
                vmin=0,
                vmax=30,
            )

    # Shared legend on clone x purity row
    all_cats = set()
    for _, anns_vis, vis_adata in slices:
        all_cats.update(vis_adata.obs[spot_label].cat.categories)
    all_cats = sorted(all_cats, key=lambda x: (x != "normal", x == "NA", x))
    axes[clone_purity_ri, -1].legend(
        handles=build_legend(all_cats),
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=7,
        framealpha=0.8,
    )

    for ri in range(nrows):
        for ci in range(ncols):
            legend = axes[ri, ci].get_legend()
            if legend is None:
                continue
            if ci < ncols - 1:
                legend.remove()
            else:
                legend.set_bbox_to_anchor((1.0, 0.5))
                legend._loc = 6

    for ri in range(nrows):
        for ci, (rep_id, _, _) in enumerate(slices):
            axes[ri, ci].set_title(f"{sample}_{rep_id}", fontsize=10)
    for ri, rlabel in enumerate(row_labels):
        axes[ri, 0].set_ylabel(rlabel, fontsize=13, fontweight="bold")

    fig.tight_layout()
    out_file = os.path.join(out_dir, f"{sample}.visium.pdf")
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"saved visium panel to {out_file}")


def plot_visium_loh_baf(
    sample: str,
    slices: list,
    raw_clust,
    loh_baf,
    loh_info,
    out_file: str,
    dpi=200,
    size=1.5,
    alpha=0.8,
):
    """PDF with visium LOH BAF spatial plots.

    Args:
        raw_clust: cluster-level SX_Data-like object (from seg_sx.to_cluster_level()).
        loh_baf: float (N, K_tumor) from compute_loh_baf — aggregated BAF per clone.
        loh_info: list of (clone_name, entries) from compute_loh_baf.

    Pages:
    - One page per CNP cluster with LOH in at least one tumor clone,
      showing per-spot BAF. Title = CN states + segments.
    - One page per tumor clone, showing aggregated BAF across its LOH clusters.
    """
    import squidpy as sq
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    ncols = len(slices)
    bc_to_idx = {bc: i for i, bc in enumerate(raw_clust.barcodes["BARCODE"])}

    # Find clusters with LOH in at least one tumor clone
    loh_clusters = []
    for gi in range(raw_clust.G):
        loh_ks = [
            k
            for k in range(1, raw_clust.K)
            if raw_clust.B[gi, k] == 0 and raw_clust.A[gi, k] > 0
        ]
        if loh_ks:
            loh_clusters.append(gi)

    if not loh_clusters and not loh_info:
        logging.info("no LOH clusters found, skipping LOH BAF visium")
        return

    def _spot_vals(data, vis_adata):
        """Map (N,) array to per-vis-spot values via barcode index."""
        vals = np.full(len(vis_adata), np.nan)
        for si, bc in enumerate(vis_adata.obs_names):
            idx = bc_to_idx.get(bc, -1)
            if idx >= 0:
                vals[si] = data[idx]
        return vals

    # BAF colormap matching plot_heatmap: NaN=white, 0.5=gray
    baf_colors = [
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
    baf_cmap = mcolors.ListedColormap(baf_colors, name="baf_disc")
    baf_cmap.set_bad("white")
    baf_norm = mcolors.BoundaryNorm(np.linspace(0, 1, 11), baf_cmap.N, clip=True)

    def _plot_page(pdf, col_data, title):
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6), squeeze=False)
        for ci, (rep_id, anns_vis, vis_adata) in enumerate(slices):
            col_name = "_loh_tmp"
            vis_adata.obs[col_name] = _spot_vals(col_data, vis_adata)
            sq.pl.spatial_scatter(
                vis_adata,
                color=col_name,
                size=size,
                library_id=rep_id,
                ax=axes[0, ci],
                img=False,
                alpha=alpha,
                edgecolors="none",
                cmap=baf_cmap,
                norm=baf_norm,
            )
            axes[0, ci].set_title(f"{sample}_{rep_id}", fontsize=10)
        fig.suptitle(title, fontsize=11, fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig, dpi=dpi)
        plt.close(fig)

    with PdfPages(out_file) as pdf:
        # Per-cluster pages
        for gi in loh_clusters:
            row = raw_clust.cnv_blocks.iloc[gi]
            cn_parts = [
                f"{raw_clust.clones[k]}={raw_clust.A[gi, k]}|{raw_clust.B[gi, k]}"
                for k in range(raw_clust.K)
            ]
            length_mb = row.get("LENGTH", 0) / 1e6
            n_bbc = int(row.get("#BBC", 0))
            title = (
                f"cluster {gi} ({length_mb:.1f}Mb, {n_bbc} BBCs): {', '.join(cn_parts)}"
            )

            D_g = raw_clust.D[gi].astype(float)
            Y_g = raw_clust.Y[gi].astype(float)
            baf_g = np.where(D_g > 0, Y_g / D_g, np.nan)
            _plot_page(pdf, baf_g, title)

        # Aggregated LOH BAF per clone
        for ki, (clone, entries) in enumerate(loh_info):
            _plot_page(
                pdf,
                loh_baf[:, ki],
                f"Aggregated LOH BAF — {clone} ({len(entries)} clusters)",
            )

    logging.info(f"saved LOH BAF visium to {out_file}")


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
    """Two PDFs: spatial (H&E + labels) and histograms (posterior + purity)."""
    import squidpy as sq
    from matplotlib.backends.backend_pdf import PdfPages
    from sklearn.metrics import roc_auc_score
    from matplotlib.collections import PatchCollection as _PC

    niters = len(labeling_trace)
    ncols = len(slices)
    bc_to_idx = {bc: i for i, bc in enumerate(barcodes["BARCODE"].values)}

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

    def _frame_cats(labels, has_purity):
        unique = sorted(set(labels))
        if has_purity and clones:
            cats = [c for c in clones if c in unique]
            return cats if cats else unique
        return unique

    col_w, row_h = 6, 5
    label_col = "_iter_clone"
    spatial_file = os.path.join(out_dir, f"{sample}.visium_iters.pdf")
    hist_file = os.path.join(out_dir, f"{sample}.iter_histograms.pdf")

    # --- Spatial PDF ---
    with PdfPages(spatial_file) as pdf:
        for ri, lt in enumerate(labeling_trace):
            labels = lt["labels"]
            max_post = lt["max_posterior"]
            purity = lt.get("tumor_purity")
            has_purity = purity is not None
            cats = _frame_cats(labels, has_purity)

            fig, axes = plt.subplots(
                nrows=2,
                ncols=ncols,
                figsize=(col_w * ncols, row_h * 2),
                squeeze=False,
            )
            for ci, (rep_id, _, vis_adata_orig) in enumerate(slices):
                vis_adata = vis_adata_orig.copy()
                idx = [bc_to_idx[bc] for bc in vis_adata.obs_names]
                vis_adata.obs[label_col] = pd.Categorical(labels[idx], categories=cats)
                vis_adata.obs["max_posterior"] = max_post[idx]
                if has_purity:
                    vis_adata.obs["tumor_purity"] = purity[idx]

                ax0 = axes[0, ci]
                set_label_colors(vis_adata, label_col)
                if has_purity:
                    sq.pl.spatial_scatter(
                        vis_adata,
                        color=label_col,
                        size=size,
                        library_id=rep_id,
                        ax=ax0,
                        edgecolors="none",
                        img=False,
                    )
                    rgba = blend_purity_rgba(vis_adata, label_col, "tumor_purity")
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
                        edgecolors="none",
                        img=False,
                    )
                # per-slice subtitle with normal accuracy on page 0
                subtitle = f"{rep_id}"
                if ri == 0 and ref_label and ref_label in barcodes.columns:
                    spot_labels = labels[idx]
                    ref_vals = barcodes[ref_label].values[idx]
                    pred_normal = spot_labels == "normal"
                    ref_normal = np.array(
                        [
                            not is_tumor_label(x) and x not in NA_CELLTYPE
                            for x in ref_vals
                        ]
                    )
                    tp = int((pred_normal & ref_normal).sum())
                    fp = int((pred_normal & ~ref_normal).sum())
                    fn = int((~pred_normal & ref_normal).sum())
                    p = tp / max(tp + fp, 1)
                    r = tp / max(tp + fn, 1)
                    subtitle += f"\nprec={p:.2f} rec={r:.2f}"
                ax0.set_title(subtitle, fontsize=9)
                leg = ax0.get_legend()
                if leg is not None:
                    leg.remove()

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
                    img=False,
                )
                ax1.set_title(f"{rep_id} — max posterior", fontsize=10)
                if ci < ncols - 1:
                    cb = ax1.collections[-1].colorbar if ax1.collections else None
                    if cb is not None:
                        cb.remove()

            axes[0, -1].legend(
                handles=build_legend(cats),
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                fontsize=8,
            )
            fig.suptitle(_make_title(ri, labels), fontsize=12, fontweight="bold")
            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
    logging.info(f"saved visium iterations to {spatial_file}")

    # --- Histogram PDF ---
    with PdfPages(hist_file) as pdf:
        for ri, lt in enumerate(labeling_trace):
            labels = lt["labels"]
            max_post = lt["max_posterior"]
            purity = lt.get("tumor_purity")
            if purity is None:
                continue

            cats = _frame_cats(labels, True)
            active = [c for c in cats if (labels == c).any()]
            colors = build_label_colors(active)
            hlabels = [f"{c} (n={int((labels == c).sum())})" for c in active]

            fig, (ax_post, ax_pur) = plt.subplots(1, 2, figsize=(14, 4))
            ax_post.hist(
                [max_post[labels == c] for c in active],
                bins=50,
                range=(0, 1),
                stacked=True,
                label=hlabels,
                color=colors,
                edgecolor="black",
                linewidth=0.3,
            )
            ax_post.set_xlabel("max posterior", fontsize=10)
            ax_post.set_ylabel("count", fontsize=10)
            ax_post.set_title("posterior distribution", fontsize=10)
            ax_post.legend(fontsize=8)

            ax_pur.hist(
                [purity[labels == c] for c in active],
                bins=50,
                range=(0, 1),
                stacked=True,
                label=hlabels,
                color=colors,
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
