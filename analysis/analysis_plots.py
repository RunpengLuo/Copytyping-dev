"""Validation metrics + validate-only plots, relocated out of the copytyping package.

Consolidates the evaluation metrics (formerly copytyping.validation.metrics) and
the validate-only plotting routines (formerly in copytyping.plot.plot_common +
copytyping.plot.plot_modality). Shared plotting/IO helpers are still imported from
the installed copytyping package.
"""

import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from copytyping.inference.count_data import get_cnp_mask
from copytyping.plot.plot_heatmap import plot_cnv_heatmap
from copytyping.plot.plot_scatter_1d import plot_rdr_baf_1d_pseudobulk
from copytyping.plot.plot_scatter_2d import plot_scatter_2d_per_cell
from copytyping.utils import NA_CELLTYPE, is_tumor_label

from sx_data import adaptive_bin_bbc


# ── metrics ──


def _baf_within_var(baf, lab):
    """Compute within-cluster BAF variance."""
    n_total = len(baf)
    within_var = 0.0
    for l in np.unique(lab):
        m = lab == l
        if m.sum() > 1:
            within_var += m.sum() * np.var(baf[m])
    return within_var / max(n_total, 1)


def compute_cluster_baf_metrics(sx_data, labels):
    """Compute per-cluster within-cluster BAF variance.

    For each informative cluster g, computes:
    - within_var: all cells
    - within_var_tumor: tumor cells only (excluding "normal" label)

    Returns dict: cluster_index -> {within_var, within_var_tumor}.
    """
    results = {}
    for g in range(sx_data.G):
        if not (sx_data.MASK["IMBALANCED"][g] or sx_data.MASK["ANEUPLOID"][g]):
            continue
        D_g = sx_data.D[g].astype(float)
        Y_g = sx_data.Y[g].astype(float)
        valid = D_g > 0
        baf = Y_g[valid] / D_g[valid]
        lab = labels[valid]

        wvar = _baf_within_var(baf, lab)

        tumor_mask = lab != "normal"
        wvar_t = (
            _baf_within_var(baf[tumor_mask], lab[tumor_mask])
            if tumor_mask.sum() > 0
            else np.nan
        )

        results[g] = {"within_var": wvar, "within_var_tumor": wvar_t}
    return results


def _eval_subset(anns_sub, qry_label, ref_label, tumor_post):
    """Compute metrics for a subset of annotations.

    All metrics (prec/recall/f1/AUC/ARI_binary/ARI_clone) are computed on cells
    where BOTH ref and pred labels are non-NA. NAs are still preserved upstream
    (e.g., in the crosstab) but excluded from these metric computations.
    """
    ref_known = ~anns_sub[ref_label].isin(NA_CELLTYPE)
    pred_na = anns_sub[qry_label].isin(NA_CELLTYPE)
    na_count = int(pred_na.sum())
    total = len(anns_sub)

    # Binary metrics: both ref-known and pred-known
    both_known = ref_known & ~pred_na
    anns_bin = anns_sub[both_known]
    y_true = anns_bin[ref_label].apply(is_tumor_label).to_numpy(dtype=int)
    has_both = len(y_true) > 0 and 0 < y_true.sum() < len(y_true)

    precision = recall = f1 = accuracy = auc_hard = np.nan
    if has_both:
        y_pred = anns_bin[qry_label].apply(is_tumor_label).to_numpy(dtype=int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0.0
        )
        accuracy = accuracy_score(y_true, y_pred)
        auc_hard = roc_auc_score(y_true, y_pred)

    auc_soft = np.nan
    if has_both and tumor_post in anns_bin:
        soft_vals = anns_bin[tumor_post].values
        soft_valid = ~np.isnan(soft_vals.astype(float))
        if soft_valid.sum() > 0:
            auc_soft = roc_auc_score(y_true[soft_valid], soft_vals[soft_valid])

    # ARI_binary: both ref-known and pred-known
    ari_binary = np.nan
    if has_both:
        ref_binary = np.where(y_true, "tumor", "normal")
        pred_binary = np.where(
            anns_bin[qry_label].apply(is_tumor_label).to_numpy(), "tumor", "normal"
        )
        ari_binary = adjusted_rand_score(ref_binary, pred_binary)

    # ARI_clone: also on both-known
    ari_clone = np.nan
    if len(anns_bin) > 0 and anns_bin[ref_label].nunique() > 1:
        ari_clone = adjusted_rand_score(anns_bin[ref_label], anns_bin[qry_label])

    label_counts = anns_sub[qry_label].value_counts()
    clone_cols = sorted([c for c in label_counts.index if c.startswith("clone")])
    metric = {
        "total": total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "AUC_hard": auc_hard,
        "AUC_soft": auc_soft,
        "ARI_binary": ari_binary,
        "ARI_clone": ari_clone,
        "#normal": int(label_counts.get("normal", 0)),
    }
    for c in clone_cols:
        metric[f"#{c}"] = int(label_counts.get(c, 0))
    metric["#NA"] = na_count

    if "tumor_purity" in anns_sub.columns:
        for grp, sub in anns_sub.groupby(qry_label):
            metric[f"purity_{grp}"] = float(sub["tumor_purity"].mean())
    return metric


def evaluate_malignant_accuracy(
    anns: pd.DataFrame,
    qry_label: str,
    ref_label: str,
    tumor_post: str,
):
    """Evaluate classification accuracy. Returns metric dict."""
    metric = _eval_subset(anns, qry_label, ref_label, tumor_post)

    known_mask = ~anns[ref_label].isin(NA_CELLTYPE)
    anns_known = anns[known_mask]
    ct = pd.crosstab(
        anns_known[ref_label],
        anns_known[qry_label],
        margins=True,
        margins_name="total",
    )

    def _fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "NA"
        return f"{v:.4f}"

    logging.info("evaluation:")
    logging.info(f"  precision = {_fmt(metric['precision'])}")
    logging.info(f"  recall    = {_fmt(metric['recall'])}")
    logging.info(f"  f1        = {_fmt(metric['f1'])}")
    logging.info(f"  accuracy  = {_fmt(metric['accuracy'])}")
    logging.info(f"  AUC_hard  = {_fmt(metric['AUC_hard'])}")
    logging.info(f"  AUC_soft  = {_fmt(metric['AUC_soft'])}")
    logging.info(f"  ARI_binary= {_fmt(metric['ARI_binary'])}")
    logging.info(f"  ARI_clone = {_fmt(metric['ARI_clone'])}")
    logging.info(f"  #NA       = {metric['#NA']}/{metric['total']}")
    logging.info(f"  crosstab (rows=ref {ref_label}, cols=pred {qry_label}):")
    for line in ct.to_string().splitlines():
        logging.info(f"    {line}")

    return metric


def joincount_zscore(labels, W):
    """Per-label joincount z-score for spatial coherence.

    Args:
        labels: (N,) array of clone labels.
        W: (N, N) row-normalized adjacency matrix (e.g. from squidpy).

    Ref: Bouayad Agha & Bellefon, Handbook of Spatial Analysis (2018).
    """
    labels = np.asarray(labels)
    N = len(labels)

    if sparse.issparse(W):
        W = W.toarray()
    W = np.asarray(W, dtype=np.float64)

    W_sum = W.sum()
    W2_sum = (W * W).sum()

    results = {}
    for lab in np.unique(labels):
        mask = (labels == lab).astype(np.float64)
        P = mask.sum() / N
        if P < 1e-6 or P > 1.0 - 1e-6:
            results[lab] = np.nan
            continue

        J = 0.5 * float(mask @ W @ mask)
        E_J = 0.5 * W_sum * P * P
        Var_J = 0.5 * W2_sum * P * P * (1.0 - P * P)
        results[lab] = float((J - E_J) / np.sqrt(Var_J)) if Var_J > 0 else np.nan

    return results


def compute_joincount_zscores(anns, label, spatial_graphs, assay_types):
    """Compute joincount z-scores across all reps and data types.

    Returns dict with keys like JC_{rep_id}_{clone_label}.
    """
    jc_metric = {}
    anns_indexed = anns.set_index("BARCODE")
    for rep_id in anns["REP_ID"].unique():
        for assay_type in assay_types:
            if assay_type not in spatial_graphs:
                continue
            sg_reps = spatial_graphs[assay_type]
            if rep_id not in sg_reps:
                continue
            sg = sg_reps[rep_id]
            sg_keep = np.isin(sg["BARCODE"], anns_indexed.index)
            if sg_keep.sum() == 0:
                continue
            sg_barcodes = sg["BARCODE"][sg_keep]
            W = sg["W"].tocsr() if hasattr(sg["W"], "tocsr") else sg["W"]
            sg_W = W[np.ix_(sg_keep, sg_keep)]
            clone_labels = anns_indexed.loc[sg_barcodes, label].to_numpy()
            jc = joincount_zscore(clone_labels, sg_W)
            logging.info(f"joincount z-score (rep={rep_id}, {assay_type}):")
            for lab, z in sorted(jc.items()):
                if lab == "normal":
                    continue
                logging.info(f"  {lab:8s}: {z:.4f}")
                jc_metric[f"JC_{rep_id}_{lab}"] = z
    return jc_metric


def evaluate_init_normal(init_is_normal, barcodes, ref_label):
    """Log precision/recall/f1 for init normal identification vs ref_label."""
    if ref_label not in barcodes.columns:
        return
    ref_vals = barcodes[ref_label].values
    known = ~np.isin(ref_vals, list(NA_CELLTYPE))
    ref_normal = np.array([not is_tumor_label(x) for x in ref_vals])
    tp = int((init_is_normal[known] & ref_normal[known]).sum())
    fp = int((init_is_normal[known] & ~ref_normal[known]).sum())
    fn = int((~init_is_normal[known] & ref_normal[known]).sum())
    tn = int((~init_is_normal[known] & ~ref_normal[known]).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-10)
    logging.info(
        f"init normal vs ref_label={ref_label}: "
        f"TP={tp} FP={fp} FN={fn} TN={tn} "
        f"prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}"
    )


def refine_labels_by_reference(anns, ref_label, cell_label, out_label):
    num_na_before = (anns[cell_label] == "NA").sum()
    anns[out_label] = anns[cell_label]
    ref_is_tumor = anns[ref_label].apply(is_tumor_label)
    anns.loc[ref_is_tumor & (anns[cell_label] == "normal"), out_label] = "NA"
    anns.loc[~ref_is_tumor & (anns[cell_label] != "normal"), out_label] = "NA"
    num_na_after = (anns[out_label] == "NA").sum()
    logging.info(
        f"#NA before/after refinement={num_na_before}->{num_na_after} / {len(anns)}"
    )
    return anns


# ── validate-only plots ──


def plot_init_baf_histograms(
    data_sources: dict,
    is_normal: np.ndarray,
    sample: str,
    out_dir: str,
    dpi=100,
):
    """Per-LOH-cluster BAF histogram, colored by init normal/tumor label.

    One page per cluster, title shows cluster index and (A,B) CN state.
    Only plots clusters in CLONAL_LOH mask.
    """
    for assay_type, sx in data_sources.items():
        loh_mask = sx.MASK.get("CLONAL_LOH", np.zeros(sx.G, dtype=bool))
        if not loh_mask.any():
            continue
        outfile = os.path.join(out_dir, f"{sample}.{assay_type}.init_baf.pdf")
        loh_idx = np.where(loh_mask)[0]

        with PdfPages(outfile) as pdf:
            for g in loh_idx:
                D_g = sx.D[g]
                valid = D_g > 0
                if valid.sum() == 0:
                    continue
                baf_g = sx.Y[g, valid].astype(float) / D_g[valid].astype(float)
                labels_g = is_normal[valid]

                fig, ax = plt.subplots(figsize=(8, 4))
                baf_normal = baf_g[labels_g]
                baf_tumor = baf_g[~labels_g]
                ax.hist(
                    [baf_normal, baf_tumor],
                    bins=50,
                    range=(0, 1),
                    stacked=True,
                    label=[
                        f"normal (n={len(baf_normal)})",
                        f"tumor (n={len(baf_tumor)})",
                    ],
                    color=["lightgray", "#d62728"],
                    edgecolor="black",
                    linewidth=0.3,
                )
                ax.set_xlabel("BAF", fontsize=10)
                ax.set_ylabel("count", fontsize=10)
                ax.legend(fontsize=8)

                # title with cluster CN state
                a_str = f"{sx.cn_A[g, 1]}|{sx.cn_B[g, 1]}"
                if sx.K > 2:
                    a_str = ";".join(
                        f"{sx.cn_A[g, k]}|{sx.cn_B[g, k]}" for k in range(1, sx.K)
                    )
                fig.suptitle(
                    f"{sample} — {assay_type} cluster {g} — CN: {a_str}",
                    fontsize=11,
                    fontweight="bold",
                )
                fig.tight_layout()
                pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
        logging.info(f"saved init BAF histograms to {outfile}")


def _is_na_label(label):
    """Check if label is uninformative (NA/Unknown, case-insensitive)."""
    return label.lower() in {x.lower() for x in NA_CELLTYPE}


def plot_crosstab(
    assign_df: pd.DataFrame,
    sample: str,
    outfile: str,
    metric: dict,
    acol="copytyping_label",
    bcol="cell_type",
):
    """Plot cross-tabulation heatmap: rows = GT labels (bcol), cols = predicted (acol)."""
    ct = pd.crosstab(assign_df[bcol], assign_df[acol])

    gt_tumor = sorted([r for r in ct.index if is_tumor_label(r)])
    gt_normal = sorted([r for r in ct.index if not is_tumor_label(r)])
    row_order = gt_normal + gt_tumor

    pred_normal = [c for c in ct.columns if c == "normal"]
    pred_tumor = sorted([c for c in ct.columns if is_tumor_label(c)])
    pred_na = [c for c in ct.columns if c in NA_CELLTYPE]
    pred_other = sorted(
        [c for c in ct.columns if c not in pred_normal + pred_tumor + pred_na]
    )
    col_order = pred_normal + pred_tumor + pred_other + pred_na

    row_order = [r for r in row_order if r in ct.index]
    col_order = [c for c in col_order if c in ct.columns]
    ct = ct.loc[row_order, col_order]
    counts = ct.to_numpy(dtype=float)
    num_rows, num_cols = counts.shape

    prec = metric.get("precision", np.nan)
    rec = metric.get("recall", np.nan)
    f1 = metric.get("f1", np.nan)

    error = np.zeros((num_rows, num_cols), dtype=bool)
    for i, gt_lab in enumerate(row_order):
        if _is_na_label(gt_lab):
            continue
        gt_is_tumor = is_tumor_label(gt_lab)
        for j, pred_lab in enumerate(col_order):
            pred_is_tumor = is_tumor_label(pred_lab)
            pred_is_na = pred_lab in NA_CELLTYPE
            pred_is_normal = pred_lab == "normal"
            if pred_is_na:
                error[i, j] = True
            elif gt_is_tumor and pred_is_normal:
                error[i, j] = True
            elif not gt_is_tumor and pred_is_tumor:
                error[i, j] = True

    rgba = np.ones((num_rows, num_cols, 4))
    for i in range(num_rows):
        for j in range(num_cols):
            if error[i, j] and counts[i, j] > 0:
                rgba[i, j] = [1.0, 0.85, 0.85, 1.0]

    cell_w = max(0.5, min(0.7, 5.0 / num_cols))
    cell_h = max(0.35, min(0.5, 4.0 / num_rows))
    fig_w = max(4, cell_w * num_cols + 2.5)
    fig_h = max(2.5, cell_h * num_rows + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(rgba, aspect="auto")

    font_size = max(5, min(8, int(160 / max(num_cols, num_rows))))
    for i in range(num_rows):
        for j in range(num_cols):
            n = int(counts[i, j])
            color = "grey" if n == 0 else "black"
            ax.text(
                j, i, str(n), ha="center", va="center", color=color, fontsize=font_size
            )

    total_n = counts.sum()
    col_sums = counts.sum(axis=0)
    row_sums_1d = counts.sum(axis=1)
    col_labels = [
        f"{c}\n({int(col_sums[j])}, {col_sums[j] / total_n * 100:.1f}%)"
        for j, c in enumerate(col_order)
    ]
    row_labels = [
        f"{r}\n({int(row_sums_1d[i])}, {row_sums_1d[i] / total_n * 100:.1f}%)"
        for i, r in enumerate(row_order)
    ]
    ax.set_xticks(np.arange(num_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(num_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.tick_params(length=0)
    ax.set_xlabel(f"predicted ({acol})", fontsize=10)
    ax.set_ylabel(f"reference ({bcol})", fontsize=10)
    ax.set_title(
        f"{sample}\nprec={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}", fontsize=11
    )

    for i in range(num_rows + 1):
        ax.axhline(i - 0.5, color="grey", linewidth=0.5)
    for j in range(num_cols + 1):
        ax.axvline(j - 0.5, color="grey", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def plot_purity_histograms(
    anns: pd.DataFrame,
    sample: str,
    outfile: str,
    label_col: str,
    purity_col="tumor_purity",
    xlabel="tumor purity",
    dpi=100,
):
    """Per-label histogram of `purity_col`. One page per label in a PDF."""
    if purity_col not in anns.columns:
        return
    labels = sorted(anns[label_col].unique(), key=lambda x: (x != "normal", x))
    with PdfPages(outfile) as pdf:
        for lab in labels:
            mask = anns[label_col] == lab
            vals = anns.loc[mask, purity_col].dropna().values
            if len(vals) == 0:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(vals, bins=50, range=(0, 1), edgecolor="black", linewidth=0.3)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("count", fontsize=10)
            ax.set_title(
                f"{sample} — {lab} (n={len(vals)}, "
                f"median={np.median(vals):.2f}, mean={np.mean(vals):.2f})",
                fontsize=11,
            )
            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)
    logging.info(f"saved {xlabel} histograms to {outfile}")


def plot_cluster_observed_data(
    seg_sx,
    anns: pd.DataFrame,
    sample: str,
    outfile: str,
    label_col: str,
    base_props=None,
    baf_metrics=None,
    dpi=100,
):
    """Per-cluster observed B-allele count, BAF, RDR. One page per cluster, one row per label."""
    labels = sorted(anns[label_col].unique(), key=lambda x: (x != "normal", x))
    n_labels = len(labels)
    label_idx = {lab: np.where(anns[label_col].values == lab)[0] for lab in labels}

    if base_props is None:
        base_props = seg_sx.X.sum(axis=1) / max(seg_sx.T.sum(), 1)

    informative = seg_sx.MASK["IMBALANCED"] | seg_sx.MASK["ANEUPLOID"]
    cluster_indices = np.where(informative)[0]

    with PdfPages(outfile) as pdf:
        for g in cluster_indices:
            row = seg_sx.cnv_blocks.iloc[g]
            cnp_str = row.get("CNP", "")
            length_mb = row["LENGTH"] / 1e6 if "LENGTH" in row.index else np.nan
            n_bbc = int(row["#BBC"]) if "#BBC" in row.index else 0
            is_imb = seg_sx.MASK["IMBALANCED"][g]
            is_ane = seg_sx.MASK["ANEUPLOID"][g]
            tag = []
            if is_imb:
                tag.append("IMB")
            if is_ane:
                tag.append("ANE")

            cn_parts = []
            for k in range(1, seg_sx.K):
                cn_parts.append(
                    f"{seg_sx.clones[k]}={seg_sx.cn_A[g, k]}|{seg_sx.cn_B[g, k]}"
                )
            cn_str = ", ".join(cn_parts)

            fig, axes = plt.subplots(
                n_labels,
                3,
                figsize=(18, 3 * n_labels),
                squeeze=False,
            )
            for ri, lab in enumerate(labels):
                idx = label_idx[lab]
                D_g = seg_sx.D[g, idx].astype(float)
                Y_g = seg_sx.Y[g, idx].astype(float)
                X_g = seg_sx.X[g, idx].astype(float)
                T_i = seg_sx.T[idx].astype(float)
                lam_g = base_props[g]

                valid_d = D_g > 0
                baf = np.divide(Y_g, D_g, out=np.full_like(D_g, np.nan), where=valid_d)
                mu = T_i * lam_g
                valid_x = mu > 0
                rdr = np.divide(X_g, mu, out=np.full_like(X_g, np.nan), where=valid_x)

                # B-allele count
                ax = axes[ri, 0]
                ax.hist(Y_g[valid_d], bins=50, edgecolor="black", linewidth=0.3)
                ax.set_xlabel("B-allele count")
                ax.set_ylabel("count")
                ax.set_title(f"{lab} (n={len(idx)})", fontsize=9)

                # BAF
                ax = axes[ri, 1]
                baf_valid = baf[valid_d]
                d_valid = D_g[valid_d]
                baf_lo = baf_valid[d_valid <= 3]
                baf_hi = baf_valid[d_valid > 3]
                ax.hist(
                    [baf_hi, baf_lo],
                    bins=50,
                    range=(0, 1),
                    stacked=True,
                    label=[f"D>3 (n={len(baf_hi)})", f"D<=3 (n={len(baf_lo)})"],
                    color=["#1f77b4", "#d3d3d3"],
                    edgecolor="black",
                    linewidth=0.3,
                )
                ax.legend(fontsize=6)
                ax.set_xlabel("BAF")
                ax.set_ylabel("count")
                med_baf = np.median(baf_valid) if len(baf_valid) > 0 else np.nan
                ax.set_title(f"BAF median={med_baf:.3f}", fontsize=9)

                # RDR
                ax = axes[ri, 2]
                rdr_valid = rdr[valid_x & np.isfinite(rdr)]
                rdr_clip = np.clip(rdr_valid, 0, 5)
                ax.hist(
                    rdr_clip, bins=50, range=(0, 5), edgecolor="black", linewidth=0.3
                )
                ax.set_xlabel("RDR")
                ax.set_ylabel("count")
                med_rdr = np.median(rdr_valid) if len(rdr_valid) > 0 else np.nan
                ax.set_title(f"RDR median={med_rdr:.3f}", fontsize=9)

            metric_str = ""
            if baf_metrics and g in baf_metrics:
                m = baf_metrics[g]
                metric_str = (
                    f"\nwithin_var={m['within_var']:.4f}"
                    f"  within_var_tumor={m['within_var_tumor']:.4f}"
                )
            fig.suptitle(
                f"{sample} — cluster {g} ({length_mb:.1f}Mb, {n_bbc} BBCs) — "
                f"{'/'.join(tag)}{metric_str}\nCN: {cn_str}",
                fontsize=10,
                fontweight="bold",
            )
            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

        # LOH pages: one page per clone label, aggregated across LOH clusters
        clones = seg_sx.clones
        for lab in labels:
            idx = label_idx[lab]
            if len(idx) == 0:
                continue
            if lab == "normal":
                k = 0
            elif lab in clones:
                k = clones.index(lab)
            else:
                continue

            loh_mask = (seg_sx.cn_B[:, k] == 0) & (seg_sx.cn_A[:, k] > 0)
            n_loh = int(loh_mask.sum())
            if n_loh == 0:
                continue

            loh_length_mb = 0
            if "LENGTH" in seg_sx.cnv_blocks.columns:
                loh_length_mb = seg_sx.cnv_blocks.loc[loh_mask, "LENGTH"].sum() / 1e6

            Y_loh = seg_sx.Y[loh_mask][:, idx].sum(axis=0).astype(float)
            D_loh = seg_sx.D[loh_mask][:, idx].sum(axis=0).astype(float)
            valid = D_loh > 0
            baf_loh = np.divide(
                Y_loh, D_loh, out=np.full_like(Y_loh, np.nan), where=valid
            )

            fig, axes = plt.subplots(1, 3, figsize=(18, 4))
            axes[0].hist(Y_loh, bins=50, edgecolor="black", linewidth=0.3)
            axes[0].set_xlabel("Y_loh (B-allele count)")
            axes[0].set_ylabel("count")
            axes[0].set_title(f"Y_loh  median={np.median(Y_loh):.0f}", fontsize=9)

            axes[1].hist(D_loh, bins=50, edgecolor="black", linewidth=0.3)
            axes[1].set_xlabel("D_loh (total allele count)")
            axes[1].set_ylabel("count")
            axes[1].set_title(f"D_loh  median={np.median(D_loh):.0f}", fontsize=9)

            baf_valid = baf_loh[valid]
            axes[2].hist(
                baf_valid, bins=50, range=(0, 1), edgecolor="black", linewidth=0.3
            )
            axes[2].set_xlabel("BAF_loh")
            axes[2].set_ylabel("count")
            med_baf = np.median(baf_valid) if len(baf_valid) > 0 else np.nan
            axes[2].set_title(f"BAF_loh  median={med_baf:.3f}", fontsize=9)

            fig.suptitle(
                f"{sample} — {lab} (n={len(idx)}) — "
                f"{n_loh} LOH clusters ({loh_length_mb:.1f}Mb)",
                fontsize=11,
                fontweight="bold",
            )
            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

    logging.info(f"saved cluster observed data to {outfile}")


def plot_metrics_barplot(
    summary: pd.DataFrame,
    outfile: str,
    metrics=("precision", "recall", "f1"),
    auc_metrics=("AUC_hard", "AUC_soft"),
    sample_col="SAMPLE",
    assay_types_col="ASSAY_TYPES",
    group_col="cancer_type",
    dpi=200,
):
    """Bar plot of prec/recall/f1 per sample, one page per cancer_type group.

    If AUC columns are available, adds a second row with AUC bars.
    """
    colors = {
        "precision": "#1f77b4",
        "recall": "#ff7f0e",
        "f1": "#2ca02c",
        "AUC_hard": "#d62728",
        "AUC_soft": "#9467bd",
    }

    metric_cols = [m for m in metrics if m in summary.columns]
    if not metric_cols:
        logging.info(f"skipping metrics barplot: no {list(metrics)} columns in summary")
        return
    valid = summary.dropna(subset=metric_cols, how="all").copy()
    if valid.empty:
        return

    auc_cols = [c for c in auc_metrics if c in valid.columns]
    has_auc = len(auc_cols) > 0 and not valid[auc_cols].isna().all().all()

    has_groups = group_col in valid.columns
    groups = sorted(valid[group_col].unique()) if has_groups else [None]

    with PdfPages(outfile) as pdf:
        for grp in groups:
            df = valid[valid[group_col] == grp] if grp is not None else valid
            if df.empty:
                continue

            sample_df = (
                df.sort_values("f1", ascending=False)
                .drop_duplicates(subset=[sample_col], keep="first")
                .sort_values(sample_col)
            )
            samples = sample_df[sample_col].tolist()
            if assay_types_col in sample_df.columns:
                tick_labels = [
                    f"{s}\n({assay})"
                    for s, assay in zip(samples, sample_df[assay_types_col])
                ]
            else:
                tick_labels = samples
            n = len(samples)
            x = np.arange(n)

            nrows = 2 if has_auc else 1
            fig_w = max(6, n * 0.8 + 2)
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=1,
                figsize=(fig_w, 4 * nrows),
                squeeze=False,
            )

            # Row 0: precision / recall / f1
            ax0 = axes[0, 0]
            n_metrics = len(metrics)
            bar_w = 1.0 / (n_metrics + 0.5)
            for mi, m in enumerate(metrics):
                vals = sample_df[m].astype(float).to_numpy()
                bars = ax0.bar(
                    x + mi * bar_w,
                    vals,
                    width=bar_w,
                    color=colors.get(m, f"C{mi}"),
                    label=m,
                )
                for bar, v in zip(bars, vals):
                    if np.isfinite(v):
                        ax0.text(
                            bar.get_x() + bar.get_width() / 2,
                            v + 0.01,
                            f"{v:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=6,
                        )
            ax0.set_xticks(x + bar_w * (n_metrics - 1) / 2)
            ax0.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
            ax0.set_ylim(0, 1.15)
            ax0.set_ylabel("Score")
            ax0.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
            title = grp if grp else "all samples"
            ax0.set_title(title, fontsize=11, fontweight="bold")

            # Row 1: AUC
            if has_auc:
                ax1 = axes[1, 0]
                n_auc = len(auc_cols)
                bar_w_auc = 1.0 / (n_auc + 0.5)
                for mi, m in enumerate(auc_cols):
                    vals = sample_df[m].astype(float).to_numpy()
                    bars = ax1.bar(
                        x + mi * bar_w_auc,
                        vals,
                        width=bar_w_auc,
                        color=colors.get(m, f"C{mi + 3}"),
                        label=m,
                    )
                    for bar, v in zip(bars, vals):
                        if np.isfinite(v):
                            ax1.text(
                                bar.get_x() + bar.get_width() / 2,
                                v + 0.01,
                                f"{v:.2f}",
                                ha="center",
                                va="bottom",
                                fontsize=6,
                            )
                ax1.set_xticks(x + bar_w_auc * (n_auc - 1) / 2)
                ax1.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
                ax1.set_ylim(0, 1.15)
                ax1.set_ylabel("AUC")
                ax1.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))

            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)


def plot_joincount_boxplot(
    summary: pd.DataFrame,
    outfile: str,
    sample_col="SAMPLE",
    group_col="cancer_type",
    rank_metric="f1",
    dpi=200,
):
    """Boxplot of per-rep per-clone joincount z-scores (tumor clones only).

    For each sample, picks the best run (by rank_metric), then each dot
    is one JC_{rep}_{clone} value from that run. One page per group.
    """
    # Collect JC columns (JC_{rep}_{clone} format, exclude old JC_normal)
    jc_cols = [c for c in summary.columns if c.startswith("JC_")]
    if not jc_cols:
        return

    has_groups = group_col in summary.columns
    groups = sorted(summary[group_col].unique()) if has_groups else [None]

    with PdfPages(outfile) as pdf:
        for grp in groups:
            df = summary[summary[group_col] == grp] if grp is not None else summary
            if df.empty:
                continue

            # Pick best run per sample
            best_rows = []
            for sample, sub in df.groupby(sample_col):
                if rank_metric in sub.columns:
                    valid = sub.dropna(subset=[rank_metric])
                    if not valid.empty:
                        best_rows.append(valid.loc[valid[rank_metric].idxmax()])
                        continue
                best_rows.append(sub.iloc[0])
            if not best_rows:
                continue
            best = pd.DataFrame(best_rows)

            # Melt JC columns into long format
            samples = sorted(best[sample_col].unique())
            all_vals = []
            for _, row in best.iterrows():
                sample = row[sample_col]
                for c in jc_cols:
                    v = row.get(c)
                    if pd.notna(v):
                        all_vals.append({"sample": sample, "jc": float(v)})
            if not all_vals:
                continue
            long = pd.DataFrame(all_vals)

            n = len(samples)
            fig_w = max(6, n * 1.2 + 2)
            fig, ax = plt.subplots(figsize=(fig_w, 4))

            box_data = [long[long["sample"] == s]["jc"].values for s in samples]
            bp = ax.boxplot(
                box_data,
                positions=range(n),
                widths=0.5,
                patch_artist=True,
                showfliers=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor("#5b7fb5")
                patch.set_alpha(0.7)

            rng = np.random.default_rng(42)
            for si, sample in enumerate(samples):
                vals = long[long["sample"] == sample]["jc"].values
                jitter = rng.uniform(-0.15, 0.15, size=len(vals))
                ax.scatter(
                    si + jitter,
                    vals,
                    color="#2c4d75",
                    edgecolors="#1a1a1a",
                    s=25,
                    linewidths=0.5,
                    zorder=3,
                    alpha=0.8,
                )

            ax.set_xticks(range(n))
            ax.set_xticklabels(samples, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Spatial coherence (JC z-score)", fontsize=10)
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
            title = grp if grp else "all samples"
            ax.set_title(
                f"{title} — joincount z-score (tumor clones)",
                fontsize=11,
                fontweight="bold",
            )
            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)


# ── per-modality plot orchestrator ──


def plot_modality_panel(
    *,
    sample,
    assay_type,
    prefix,
    plot_dir,
    seg_sx,
    raw_clust,
    bbc_data,
    cnv_blocks,
    anns,
    baseline_fn,
    primary_label,
    plot_labels,
    theta,
    region_bed,
    genome_size,
    dpi,
    heatmap_agg,
    min_snp_count,
    max_bin_length,
    platform_str,
    cluster_base_props=None,
    compute_baf_metrics=False,
    ascn_profile=False,
):
    """Per-modality + per-REP_ID plots: cluster_obs, cluster_2d, heatmap, 1d_scatter.

    Cluster-level plots reuse the model's fitted baseline (``cluster_base_props``,
    i.e. ``model_params["{assay}-lambda"]``); the finer-resolution heatmap (seg) and
    1d_scatter (bbc) derive theirs via ``baseline_fn(sx)``.

    Files written:
        {plot_dir}/{prefix}.{assay_type}.cluster_obs.{label}.pdf  (per plot_label)
        {plot_dir}/{prefix}.{assay_type}.cluster_2d.pdf
        {plot_dir}/{prefix}.{assay_type}.heatmap.agg{agg}.pdf
            (multi-page: rep_id outer × val inner; labels shown as color strips)
        {plot_dir}/{prefix}.{assay_type}.1d_scatter.{label}.pdf
            (multi-page: one page per rep_id)
    """
    raw_lambda = (
        cluster_base_props if cluster_base_props is not None else baseline_fn(raw_clust)
    )
    seg_lambda = baseline_fn(seg_sx)

    for obs_label in plot_labels:
        baf_metrics = None
        if compute_baf_metrics:
            baf_metrics = compute_cluster_baf_metrics(raw_clust, anns[obs_label].values)
            for g, m in sorted(baf_metrics.items()):
                logging.info(
                    f"  cluster {g} ({assay_type}, {obs_label}): "
                    f"within_var={m['within_var']:.4f}"
                )
        plot_cluster_observed_data(
            raw_clust,
            anns,
            sample,
            os.path.join(
                plot_dir, f"{prefix}.{assay_type}.cluster_obs.{obs_label}.pdf"
            ),
            label_col=obs_label,
            base_props=raw_lambda,
            baf_metrics=baf_metrics,
            dpi=dpi,
        )
    plot_scatter_2d_per_cell(
        raw_clust,
        anns,
        sample,
        os.path.join(plot_dir, f"{prefix}.{assay_type}.cluster_2d.pdf"),
        label_col=primary_label,
        base_props=raw_lambda,
        dpi=dpi,
    )

    bbc_df_assay, X_bbc, Y_bbc, D_bbc = bbc_data
    agg_bbc = None
    if bbc_df_assay is not None:
        agg_bbc = adaptive_bin_bbc(
            bbc_df_assay,
            X_bbc,
            Y_bbc,
            D_bbc,
            seg_sx,
            min_snp_count,
            max_bin_length,
        )
    agg_lambda = baseline_fn(agg_bbc) if agg_bbc is not None else None

    rep_ids = sorted(seg_sx.barcodes["REP_ID"].unique())
    rep_views = {}
    for rep_id in rep_ids:
        seg_sx_rep, rep_mask = seg_sx.subset_by_rep(rep_id)
        rep_views[rep_id] = {
            "seg_sx": seg_sx_rep,
            "anns": anns.iloc[rep_mask].reset_index(drop=True),
            "theta": theta[rep_mask] if theta is not None else None,
        }

    # one heatmap file per agg level; labels shown as color strips, no per-label spawn
    for agg in [1, heatmap_agg]:
        fname = os.path.join(
            plot_dir,
            f"{prefix}.{assay_type}.heatmap.agg{agg}.pdf",
        )
        with PdfPages(fname) as pdf:
            for rep_id in rep_ids:
                v = rep_views[rep_id]
                for val in ["BAF", "log2RDR"]:
                    if val == "log2RDR" and seg_lambda is None:
                        continue
                    plot_cnv_heatmap(
                        sample,
                        assay_type,
                        cnv_blocks,
                        v["seg_sx"],
                        v["anns"],
                        region_bed,
                        proportions=v["theta"],
                        val=val,
                        base_props=seg_lambda,
                        agg_size=agg,
                        label_cols=plot_labels,
                        primary_label=primary_label,
                        pdf_pages=pdf,
                        dpi=dpi,
                        figsize=(20, 6 if agg > 1 else 15),
                        rep_id=rep_id,
                        ascn_profile=ascn_profile,
                    )

    if agg_bbc is not None:
        for my_label in plot_labels:
            fname = os.path.join(
                plot_dir, f"{prefix}.{assay_type}.1d_scatter.{my_label}.pdf"
            )
            with PdfPages(fname) as pdf:
                for rep_id in rep_ids:
                    v = rep_views[rep_id]
                    agg_bbc_rep, _ = agg_bbc.subset_by_rep(rep_id)
                    plot_rdr_baf_1d_pseudobulk(
                        agg_bbc_rep,
                        v["anns"],
                        agg_lambda,
                        sample,
                        assay_type,
                        genome_size,
                        haplo_blocks=cnv_blocks,
                        region_bed=region_bed,
                        lab_type=my_label,
                        is_inferred=(my_label == primary_label),
                        pdf_pages=pdf,
                        platform=platform_str,
                        subtitle=f"rep={rep_id}",
                        ascn_profile=ascn_profile,
                    )


def plot_count_histograms(
    read_counts: dict[str, np.ndarray],
    total_allele_counts: dict[str, np.ndarray],
    cn_A: dict[str, np.ndarray],
    cn_B: dict[str, np.ndarray],
    cn_C: dict[str, np.ndarray],
    barcodes: dict[str, pd.DataFrame],
    sample: str,
    outfile: str,
    dpi: int = 100,
):
    """Per-rep 2x2 histogram: total/aneuploid read counts, total/imbalanced allele counts.

    Each argument is keyed by assay_type. All assay_types go in a single PDF, one
    page per (assay_type, rep_id).

    Args:
        read_counts: assay -> (G, N) read depth / feature counts.
        total_allele_counts: assay -> (G, N) total-allele counts (A + B).
        cn_A/cn_B/cn_C: assay -> (G, K) per-clone copy numbers (for masks).
        barcodes: assay -> per-cell metadata holding REP_ID.
    """

    def _hist(ax, vals, xlabel, title):
        ax.hist(vals, bins=50, edgecolor="black", linewidth=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.set_title(f"{title} (n={len(vals)}, med={int(np.median(vals))})", fontsize=9)

    with PdfPages(outfile) as pdf:
        for assay_type in read_counts:
            reads = read_counts[assay_type]
            total_allele = total_allele_counts[assay_type]
            mask = get_cnp_mask(cn_A[assay_type], cn_B[assay_type], cn_C[assay_type])
            aneu = mask["ANEUPLOID"]
            imb = mask["IMBALANCED"]
            n_aneu = int(aneu.sum())
            n_imb = int(imb.sum())
            num_cells = reads.shape[1]

            total_x = reads.sum(axis=0)
            aneu_x = reads[aneu].sum(axis=0) if n_aneu > 0 else np.zeros(num_cells)
            total_d = total_allele.sum(axis=0)
            imb_d = total_allele[imb].sum(axis=0) if n_imb > 0 else np.zeros(num_cells)
            rep_ids = barcodes[assay_type]["REP_ID"].unique()

            for rep_id in rep_ids:
                rm = barcodes[assay_type]["REP_ID"].values == rep_id
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                _hist(axes[0, 0], total_x[rm], "read count", "total read count")
                _hist(
                    axes[0, 1],
                    aneu_x[rm],
                    "read count",
                    f"aneuploid read count ({n_aneu} bins)",
                )
                _hist(axes[1, 0], total_d[rm], "allele count", "total allele count")
                _hist(
                    axes[1, 1],
                    imb_d[rm],
                    "allele count",
                    f"imbalanced allele count ({n_imb} bins)",
                )
                fig.suptitle(
                    f"{sample} — {assay_type} — {rep_id}",
                    fontsize=12,
                    fontweight="bold",
                )
                fig.tight_layout()
                pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
    logging.info(f"saved count histograms to {outfile}")
