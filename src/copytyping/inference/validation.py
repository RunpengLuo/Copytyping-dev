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

    ari = np.nan
    tumor_mask = anns_known[ref_label].apply(is_tumor_label)
    gt_tumor = anns_known[tumor_mask]
    if gt_tumor[ref_label].nunique() > 1:
        ari = adjusted_rand_score(gt_tumor[ref_label], gt_tumor[qry_label])

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
        "ARI": ari,
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
    logging.info(f"  ARI       = {_fmt(metric['ARI'])}")
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


def _mean_delta_ll(ll_gnk, g, map_k, assigned):
    r"""Mean discriminative score for cluster g.

    .. math::
        \Delta_{g,n} = \ell_{g,n,k^*} - \max_{k \neq k^*} \ell_{g,n,k}

    .. math::
        \bar{\Delta}_g = \frac{1}{N'} \sum_{n \in \text{assigned}} \Delta_{g,n}
    """
    ll_g = ll_gnk[g, assigned, :]  # (N', K)
    k_star = map_k[assigned]
    ll_map = ll_g[np.arange(len(k_star)), k_star]
    ll_g_copy = ll_g.copy()
    ll_g_copy[np.arange(len(k_star)), k_star] = -np.inf
    ll_2nd = ll_g_copy.max(axis=1)
    deltas = ll_map - ll_2nd
    return float(np.mean(deltas)), float(np.median(deltas))


def rank_clusters(ll_allele, ll_total, anns, label, clones, sx_data):
    r"""Per-cluster discrimination scores for allele and total LL separately.

    For each cluster g, computes mean \bar{\Delta}_g for allele-LL and total-LL.
    Allele score is only computed for IMBALANCED clusters;
    total score is only computed for ANEUPLOID clusters.

    Args:
        ll_allele: (G, N, K) per-cluster allele log-likelihood.
        ll_total: (G, N, K) per-cluster total log-likelihood.
        anns: annotations DataFrame.
        label: MAP label column name.
        clones: ordered clone names.
        sx_data: SX_Data with cnv_blocks and MASK.

    Returns:
        DataFrame with one row per cluster (not sorted).
    """
    G = ll_allele.shape[0]
    map_labels = anns[label].values
    clone_to_k = {c: k for k, c in enumerate(clones)}
    map_k = np.array([clone_to_k.get(lab, -1) for lab in map_labels])
    assigned = map_k >= 0

    imb = sx_data.MASK["IMBALANCED"]
    aneu = sx_data.MASK["ANEUPLOID"]
    cnv = sx_data.cnv_blocks

    rows = []
    for g in range(G):
        row = {"cluster": g}
        if "SEGMENTS" in cnv.columns:
            row["SEGMENTS"] = cnv["SEGMENTS"].iloc[g]
        if "#BBC" in cnv.columns:
            row["#BBC"] = cnv["#BBC"].iloc[g]
        if "CNP" in cnv.columns:
            row["CNP"] = cnv["CNP"].iloc[g]
        if "#SNPS" in cnv.columns:
            row["#SNPS"] = cnv["#SNPS"].iloc[g]
        if "#gene" in cnv.columns:
            row["#gene"] = cnv["#gene"].iloc[g]

        if imb[g]:
            mean_d, med_d = _mean_delta_ll(ll_allele, g, map_k, assigned)
            row["allele_score"] = mean_d
            row["allele_score_med"] = med_d
        else:
            row["allele_score"] = ""
            row["allele_score_med"] = ""

        if aneu[g]:
            mean_d, med_d = _mean_delta_ll(ll_total, g, map_k, assigned)
            row["total_score"] = mean_d
            row["total_score_med"] = med_d
        else:
            row["total_score"] = ""
            row["total_score_med"] = ""

        rows.append(row)
    return pd.DataFrame(rows)


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
