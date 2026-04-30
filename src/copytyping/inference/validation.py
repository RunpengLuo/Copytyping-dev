import logging

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from copytyping.utils import NA_CELLTYPE, is_tumor_label


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
    """Compute metrics for a subset of annotations."""
    known_mask = ~anns_sub[ref_label].isin(NA_CELLTYPE)
    anns_known = anns_sub[known_mask]
    na_count = int((anns_sub[qry_label] == "NA").sum())
    total = len(anns_sub)

    y_true = anns_known[ref_label].apply(is_tumor_label).to_numpy(dtype=int)
    has_both = len(y_true) > 0 and 0 < y_true.sum() < len(y_true)

    precision = recall = f1 = accuracy = auc_hard = np.nan
    if has_both:
        y_pred = anns_known[qry_label].apply(is_tumor_label).to_numpy(dtype=int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0.0
        )
        accuracy = accuracy_score(y_true, y_pred)
        auc_hard = roc_auc_score(y_true, y_pred)

    auc_soft = np.nan
    if has_both and tumor_post in anns_known:
        auc_soft = roc_auc_score(y_true, anns_known[tumor_post])

    # ARI_binary: ref/pred both mapped to {normal, tumor}
    ari_binary = np.nan
    if has_both:
        ref_binary = np.where(y_true, "tumor", "normal")
        pred_binary = np.where(
            anns_known[qry_label].apply(is_tumor_label).to_numpy(), "tumor", "normal"
        )
        ari_binary = adjusted_rand_score(ref_binary, pred_binary)

    # ARI_clone: ref={normal,clone_1,...} vs pred={normal,clone1,...}
    ari_clone = np.nan
    if len(anns_known) > 0 and anns_known[ref_label].nunique() > 1:
        ari_clone = adjusted_rand_score(anns_known[ref_label], anns_known[qry_label])

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


def compute_joincount_zscores(anns, label, spatial_graphs, data_types):
    """Compute joincount z-scores across all reps and data types.

    Returns dict with keys like JC_{rep_id}_{clone_label}.
    """
    jc_metric = {}
    anns_indexed = anns.set_index("BARCODE")
    for rep_id in anns["REP_ID"].unique():
        for data_type in data_types:
            if data_type not in spatial_graphs:
                continue
            sg_reps = spatial_graphs[data_type]
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
            logging.info(f"joincount z-score (rep={rep_id}, {data_type}):")
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
