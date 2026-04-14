import logging

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from copytyping.utils import NA_CELLTYPE, is_tumor_label


def _eval_subset(anns_sub, cell_label, cell_type, tumor_post, skip_binary=False):
    """Compute metrics for a subset of annotations."""
    known_mask = ~anns_sub[cell_type].isin(NA_CELLTYPE)
    anns_known = anns_sub[known_mask]
    na_count = int((anns_sub[cell_label] == "NA").sum())
    total = len(anns_sub)

    y_true = anns_known[cell_type].apply(is_tumor_label).to_numpy(dtype=int)
    has_both = len(y_true) > 0 and 0 < y_true.sum() < len(y_true)

    precision = recall = f1 = accuracy = auc_hard = np.nan
    if not skip_binary and has_both:
        y_pred = anns_known[cell_label].apply(is_tumor_label).to_numpy(dtype=int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary"
        )
        accuracy = accuracy_score(y_true, y_pred)
        auc_hard = roc_auc_score(y_true, y_pred)

    auc_soft = np.nan
    if has_both and tumor_post in anns_known:
        auc_soft = roc_auc_score(y_true, anns_known[tumor_post])

    ari = np.nan
    tumor_mask = anns_known[cell_type].apply(is_tumor_label)
    gt_tumor = anns_known[tumor_mask]
    if gt_tumor[cell_type].nunique() > 1:
        ari = adjusted_rand_score(gt_tumor[cell_type], gt_tumor[cell_label])

    label_counts = anns_sub[cell_label].value_counts()
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
    return metric


def evaluate_malignant_accuracy(
    anns,
    cell_label="cell_label",
    cell_type="cell_type",
    tumor_post="tumor",
    skip_binary=False,
):
    """Evaluate classification accuracy. Returns metric dict.

    skip_binary: skip precision/recall/f1/accuracy/AUC_hard
        (degenerate for spot model which never predicts normal).
    """
    metric = _eval_subset(
        anns, cell_label, cell_type, tumor_post, skip_binary=skip_binary
    )

    known_mask = ~anns[cell_type].isin(NA_CELLTYPE)
    anns_known = anns[known_mask]
    ct = pd.crosstab(
        anns_known[cell_type],
        anns_known[cell_label],
        margins=True,
        margins_name="total",
    )

    def _fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "NA"
        return f"{v:.4f}"

    logging.info("evaluation:")
    if not skip_binary:
        logging.info(f"  precision = {_fmt(metric['precision'])}")
        logging.info(f"  recall    = {_fmt(metric['recall'])}")
        logging.info(f"  f1        = {_fmt(metric['f1'])}")
        logging.info(f"  accuracy  = {_fmt(metric['accuracy'])}")
        logging.info(f"  AUC_hard  = {_fmt(metric['AUC_hard'])}")
    logging.info(f"  AUC_soft  = {_fmt(metric['AUC_soft'])}")
    logging.info(f"  ARI       = {_fmt(metric['ARI'])}")
    logging.info(f"  #NA       = {metric['#NA']}/{metric['total']}")
    logging.info(f"  crosstab (rows=ref {cell_type}, cols=pred {cell_label}):")
    for line in ct.to_string().splitlines():
        logging.info(f"    {line}")

    return metric


def joincount_zscore(labels, coords, adjacent_dist=105.0):
    """Per-label joincount z-score for spatial coherence.

    Builds row-normalized adjacency from coordinates, binarizes each
    label, computes z-score(J_11). Higher = more spatially coherent.

    Ref: Bouayad Agha & Bellefon, Handbook of Spatial Analysis (2018).
    """
    labels = np.asarray(labels)
    coords = np.asarray(coords, dtype=np.float64)
    N = len(labels)

    dists = distance.cdist(coords, coords)
    W = (dists < adjacent_dist).astype(np.float64)
    np.fill_diagonal(W, 0.0)
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    W = W / row_sums[:, None]

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
