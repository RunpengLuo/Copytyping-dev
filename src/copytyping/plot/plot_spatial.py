import logging
import os

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PatchCollection

from sklearn.metrics import roc_auc_score

from copytyping.plot.plot_common import (
    NORMAL_COLOR,
    PURITY_CMAP,
    build_label_color_maps,
    build_label_colors,
    make_baf_cmap,
)
from copytyping.utils import is_tumor_label, NA_CELLTYPE

logging.getLogger("anndata").setLevel(logging.WARNING)


def _set_pdf_fonts():
    """Embed TrueType fonts (type 42) so PDF/SVG text stays editable."""
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def compute_loh_baf(
    ballele_counts: np.ndarray,
    total_allele_counts: np.ndarray,
    cn_A: np.ndarray,
    cn_B: np.ndarray,
    clones: list[str],
) -> tuple[np.ndarray, list]:
    """Per-spot aggregated BAF over clone-specific LOH clusters.

    Args:
        ballele_counts: (G, N) cluster-level B-allele counts.
        total_allele_counts: (G, N) cluster-level total-allele counts (A + B).
        cn_A/cn_B: (G, K) per-clone copy numbers.
        clones: clone names, length K.

    Returns (baf_array, loh_info) where:
        baf_array: float (N, K_tumor) — per-spot BAF aggregated over LOH clusters of each tumor clone.
            NaN if no allele coverage or no LOH clusters for that clone.
        loh_info: list of (clone_name, list of "cluster <tab> clone states") per clone with LOH.
    """
    num_clones = len(clones)
    num_cells = ballele_counts.shape[1]
    K_tumor = num_clones - 1
    baf = np.full((num_cells, K_tumor), np.nan)
    loh_info = []

    for ki in range(K_tumor):
        k = ki + 1  # skip normal
        clone = clones[k]
        loh_mask = (cn_B[:, k] == 0) & (cn_A[:, k] > 0)
        if loh_mask.sum() == 0:
            continue

        entries = []
        for gi in np.where(loh_mask)[0]:
            cn_parts = [
                f"{clones[j]}={cn_A[gi, j]}|{cn_B[gi, j]}" for j in range(num_clones)
            ]
            entries.append(f"cluster{gi}\t{', '.join(cn_parts)}")
        loh_info.append((clone, entries))

        count_B_loh = ballele_counts[loh_mask].sum(axis=0).astype(float)
        count_N_loh = total_allele_counts[loh_mask].sum(axis=0).astype(float)
        valid = count_N_loh > 0
        baf[valid, ki] = count_B_loh[valid] / count_N_loh[valid]

        logging.info(f"LOH clusters for {clone} ({int(loh_mask.sum())} clusters):")
        for entry in entries:
            logging.info(f"  {entry}")

    return baf, loh_info


##################################################
# orchestrator
##################################################


def plot_visium_all(
    *,
    sample: str,
    anns: pd.DataFrame,
    h5ad_source: str | sc.AnnData,
    ballele_counts: np.ndarray,
    total_allele_counts: np.ndarray,
    cn_A: np.ndarray,
    cn_B: np.ndarray,
    cluster_barcodes: pd.DataFrame,
    clones: list[str],
    plot_dir: str,
    spot_label: str,
    ref_label: str,
    best_cutoff_label: str | None = None,
    best_cutoff_metrics: dict | None = None,
    labeling_trace: list | None = None,
    barcodes: pd.DataFrame | None = None,
    dpi: int = 200,
):
    """Sample-level visium orchestrator (panel + LOH BAF + iters).

    Visium is single-modality (gex) by definition, so all three plots run
    once per sample, not per-modality.

    Args:
        anns: per-spot DataFrame with BARCODE, REP_ID, label columns.
        h5ad_source: path to .h5ad or AnnData with spatial coords.
        ballele_counts/total_allele_counts: (G, N) cluster-level gex counts (LOH BAF).
        cn_A/cn_B: (G, K) per-clone copy numbers.
        cluster_barcodes: cluster-level gex barcodes DataFrame (BARCODE column).
        clones: clone names list.
        labeling_trace: optional list of EM-iter dicts (from Spot_Model);
            if provided, also runs plot_visium_iters.
        barcodes: union barcodes DataFrame (used by plot_visium_iters).
    """
    slices = build_visium_slices(anns, h5ad_source, ref_label)
    plot_visium_panel(
        sample,
        slices,
        plot_dir,
        spot_label=spot_label,
        path_label=ref_label,
        best_cutoff_label=best_cutoff_label,
        best_cutoff_metrics=best_cutoff_metrics,
        dpi=dpi,
    )
    try:
        loh_baf, loh_info = compute_loh_baf(
            ballele_counts, total_allele_counts, cn_A, cn_B, clones
        )
        plot_visium_loh_baf(
            sample,
            slices,
            cluster_barcodes,
            ballele_counts,
            total_allele_counts,
            cn_A,
            cn_B,
            clones,
            loh_baf,
            loh_info,
            os.path.join(plot_dir, f"{sample}.visium_loh_baf.pdf"),
            dpi=dpi,
        )
    except Exception as e:
        logging.warning(f"visium_loh_baf failed: {e}")
    if labeling_trace is not None:
        plot_visium_iters(
            sample,
            slices,
            labeling_trace,
            barcodes=barcodes,
            out_dir=plot_dir,
            clones=clones,
            ref_label=ref_label,
            dpi=dpi,
        )


##################################################
# slice + color helpers
##################################################


def build_visium_slices(
    anns: pd.DataFrame, h5ad_source: str | sc.AnnData, ref_label: str
):
    """Build per-rep (rep_id, anns_vis, vis_adata) slices for visium plotting.

    h5ad_source can be a path to a .h5ad file or an already-loaded AnnData.
    Each anns_vis is reindexed to vis_adata.obs_names; ref_label column is
    filled with "Unknown" if absent.
    """
    if isinstance(h5ad_source, str):
        h5ad_adata = sc.read_h5ad(h5ad_source)
    else:
        h5ad_adata = h5ad_source
    anns_indexed = anns.set_index("BARCODE")
    slices = []
    for rep_id in sorted(anns["REP_ID"].dropna().unique()):
        anns_rep = anns[anns["REP_ID"] == rep_id]
        vis_adata = h5ad_adata[
            h5ad_adata.obs_names.isin(anns_rep["BARCODE"].values)
        ].copy()
        anns_vis = anns_indexed.reindex(vis_adata.obs_names)
        if ref_label not in anns_vis.columns:
            anns_vis[ref_label] = "Unknown"
        slices.append((rep_id, anns_vis, vis_adata))
    return slices


def set_label_colors(adata: sc.AnnData, col: str, color_map: dict[str, str]):
    """Set adata.uns[col+'_colors'] from an explicit {value: color} map
    (built once via build_label_color_maps so all columns share one palette)."""
    adata.obs[col] = adata.obs[col].astype("category")
    adata.uns[f"{col}_colors"] = [
        color_map.get(str(c), NORMAL_COLOR) for c in adata.obs[col].cat.categories
    ]


def build_legend(categories: list):
    """Build legend handles using unified colors."""
    colors = build_label_colors(categories)
    return [mpatches.Patch(color=colors[i], label=c) for i, c in enumerate(categories)]


def blend_purity_rgba(adata: sc.AnnData, label_col: str, purity_col: str):
    """Blend each spot's label color toward gray by its purity. Returns (N, 4) RGBA.

    NA/Unknown labels and NaN-purity spots are made transparent (alpha 0).
    """
    gray = np.array(mcolors.to_rgb(NORMAL_COLOR))
    labels = adata.obs[label_col].astype("category")
    cats = list(labels.cat.categories)
    cat_rgb = np.array([mcolors.to_rgb(c) for c in build_label_colors(cats)])  # (C, 3)
    codes = labels.cat.codes.to_numpy()  # (N,); -1 for NaN

    base = np.where(codes[:, None] >= 0, cat_rgb[codes.clip(min=0)], gray)  # (N, 3)
    raw_purity = adata.obs[purity_col].to_numpy(dtype=float)
    purity = np.clip(np.nan_to_num(raw_purity, nan=0.0), 0.0, 1.0)[:, None]  # (N, 1)

    rgba = np.ones((len(labels), 4), dtype=float)
    rgba[:, :3] = purity * base + (1.0 - purity) * gray
    transparent = np.isnan(raw_purity) | labels.isin(["NA", "Unknown"]).to_numpy()
    rgba[transparent, 3] = 0.0
    return rgba


def _apply_purity_blend(ax: plt.Axes, adata: sc.AnnData, label_col: str):
    """Recolor the spatial-scatter patches on ``ax`` by purity-blended RGBA."""
    rgba = blend_purity_rgba(adata, label_col, "tumor_purity")
    for child in ax.get_children():
        if isinstance(child, PatchCollection) and len(child.get_paths()) == len(rgba):
            child.set_facecolors(rgba)
            break


# ── Plot functions ──


##################################################
# page plotters
##################################################


def plot_visium_panel(
    sample: str,
    slices: list,
    out_dir: str,
    spot_label: str = "spot_label",
    path_label: str = "Microregion_annotation",
    best_cutoff_label: str | None = None,
    best_cutoff_metrics: dict | None = None,
    dpi: int = 300,
    size: float = 1.5,
    alpha: float = 0.8,
    title_info: str = "",
):
    """Single-page PDF visium panel per slice.

    Rows: H&E, path_label, ref purity, inferred purity, clone x purity,
          [best cutoff clone x purity].

    best_cutoff_label: column name in anns for best (pcut, post) cutoff labels.
    best_cutoff_metrics: dict with ARI_clone, f1 for the row title.
    """
    _set_pdf_fonts()

    has_path = path_label in slices[0][1].columns
    ref_purity_col = f"{path_label}-tumor_purity"
    has_ref_purity = ref_purity_col in slices[0][1].columns

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
    if has_best_cutoff:
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

    nrows = len(row_labels)
    ncols = len(slices)

    # one shared palette across all label columns (clone label first, then path /
    # best-cutoff); a value reused across columns keeps one color
    def _union(col: str):
        vals = set()
        for _, anns_vis, _ in slices:
            if col in anns_vis.columns:
                vals.update(anns_vis[col].astype(str).tolist())
        return sorted(vals)

    label_vals = {spot_label: _union(spot_label)}
    if has_path:
        label_vals[path_label] = _union(path_label)
    if has_best_cutoff:
        label_vals[best_cutoff_label] = _union(best_cutoff_label)
    color_maps = build_label_color_maps(label_vals, primary_label=spot_label)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 6 * nrows),
        squeeze=False,
    )

    for ci, (rep_id, anns_vis, vis_adata) in enumerate(slices):
        vis_adata.obs[spot_label] = anns_vis[spot_label].astype("category")
        set_label_colors(vis_adata, spot_label, color_maps[spot_label])
        if has_path:
            vis_adata.obs[path_label] = anns_vis[path_label].astype("category")
            set_label_colors(vis_adata, path_label, color_maps[path_label])
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
                cmap=PURITY_CMAP,
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
            cmap=PURITY_CMAP,
            vmin=0,
            vmax=1,
            edgecolors="none",
        )

        # 5. clone x purity (no H&E) — exclude NA spots
        ri += 1
        keep5 = vis_adata.obs[spot_label] != "NA"
        sub5 = vis_adata[keep5].copy()
        sub5.obs[spot_label] = sub5.obs[spot_label].cat.remove_unused_categories()
        set_label_colors(sub5, spot_label, color_maps[spot_label])
        sq.pl.spatial_scatter(
            sub5,
            color=spot_label,
            size=size,
            library_id=rep_id,
            ax=axes[ri, ci],
            img=False,
            edgecolors="none",
        )
        _apply_purity_blend(axes[ri, ci], sub5, spot_label)
        old_legend = axes[ri, ci].get_legend()
        if old_legend:
            old_legend.remove()

        # 6. Best cutoff clone x purity (with H&E) — exclude NA spots
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
            set_label_colors(sub6, best_cutoff_label, color_maps[best_cutoff_label])
            sq.pl.spatial_scatter(
                sub6,
                color=best_cutoff_label,
                size=size,
                library_id=rep_id,
                ax=axes[ri, ci],
                img=True,
                edgecolors="none",
            )
            _apply_purity_blend(axes[ri, ci], sub6, best_cutoff_label)
            old_legend = axes[ri, ci].get_legend()
            if old_legend:
                old_legend.remove()

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
    cluster_barcodes: pd.DataFrame,
    ballele_counts: np.ndarray,
    total_allele_counts: np.ndarray,
    cn_A: np.ndarray,
    cn_B: np.ndarray,
    clones: list[str],
    loh_baf: np.ndarray,
    loh_info: list,
    out_file: str,
    dpi: int = 200,
    size: float = 1.5,
    alpha: float = 0.8,
):
    """PDF with visium LOH BAF spatial plots.

    Args:
        cluster_barcodes: cluster-level gex barcodes DataFrame (BARCODE column).
        ballele_counts/total_allele_counts: (G, N) cluster-level gex counts.
        cn_A/cn_B: (G, K) per-clone copy numbers.
        clones: clone names, length K.
        loh_baf: float (N, K_tumor) from compute_loh_baf — aggregated BAF per clone.
        loh_info: list of (clone_name, entries) from compute_loh_baf.

    Pages:
    - One page per CNP cluster with LOH in at least one tumor clone,
      showing per-spot BAF. Title = CN states.
    - One page per tumor clone, showing aggregated BAF across its LOH clusters.
    """
    _set_pdf_fonts()

    ncols = len(slices)
    num_segment = ballele_counts.shape[0]
    num_clones = len(clones)
    bc_to_idx = {bc: i for i, bc in enumerate(cluster_barcodes["BARCODE"])}

    # Find clusters with LOH in at least one tumor clone
    loh_clusters = []
    for gi in range(num_segment):
        loh_ks = [
            k for k in range(1, num_clones) if cn_B[gi, k] == 0 and cn_A[gi, k] > 0
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
    baf_cmap, baf_norm = make_baf_cmap()

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
            cn_parts = [
                f"{clones[k]}={cn_A[gi, k]}|{cn_B[gi, k]}" for k in range(num_clones)
            ]
            title = f"cluster {gi}: {', '.join(cn_parts)}"

            total_allele_g = total_allele_counts[gi].astype(float)
            ballele_g = ballele_counts[gi].astype(float)
            baf_g = np.where(total_allele_g > 0, ballele_g / total_allele_g, np.nan)
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
    clones: list | None = None,
    ref_label: str | None = None,
    dpi: int = 100,
    size: float = 1.5,
):
    """Two PDFs: spatial (H&E + labels) and histograms (posterior + purity)."""
    _set_pdf_fonts()

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

    # shared clone palette across all iterations
    all_iter_labels = sorted({str(x) for lt in labeling_trace for x in lt["labels"]})
    iter_color_map = build_label_color_maps(
        {label_col: all_iter_labels}, primary_label=label_col
    )[label_col]

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
                set_label_colors(vis_adata, label_col, iter_color_map)
                sq.pl.spatial_scatter(
                    vis_adata,
                    color=label_col,
                    size=size,
                    library_id=rep_id,
                    ax=ax0,
                    edgecolors="none",
                    img=False,
                )
                if has_purity:
                    _apply_purity_blend(ax0, vis_adata, label_col)
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
                    cmap=PURITY_CMAP,
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
