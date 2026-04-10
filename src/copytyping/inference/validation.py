import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    adjusted_rand_score,
    precision_recall_fscore_support,
    accuracy_score,
)

from copytyping.utils import is_tumor_label, NA_CELLTYPE


def _eval_subset(anns_sub, cell_label, cell_type, tumor_post):
    """Compute metrics for a subset of annotations. Returns dict."""
    known_mask = ~anns_sub[cell_type].isin(NA_CELLTYPE)
    anns_known = anns_sub[known_mask]
    na_count = int((anns_sub[cell_label] == "NA").sum())
    total = len(anns_sub)

    y_true = anns_known[cell_type].apply(is_tumor_label).to_numpy(dtype=int)
    y_pred_hard = anns_known[cell_label].apply(is_tumor_label).to_numpy(dtype=int)

    if len(y_true) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        precision = recall = f1 = accuracy = auc_hard = np.nan
        auc_soft = np.nan
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_hard, average="binary"
        )
        accuracy = accuracy_score(y_true, y_pred_hard)
        auc_hard = roc_auc_score(y_true, y_pred_hard)
        auc_soft = None
        if tumor_post in anns_known:
            auc_soft = roc_auc_score(y_true, anns_known[tumor_post])

    # multi-clone ARI: only meaningful when GT distinguishes >1 tumor label
    ari = np.nan
    tumor_mask = anns_known[cell_type].apply(is_tumor_label)
    gt_tumor_labels = anns_known.loc[tumor_mask, cell_type]
    if gt_tumor_labels.nunique() > 1:
        pred_tumor_labels = anns_known.loc[tumor_mask, cell_label]
        ari = adjusted_rand_score(gt_tumor_labels, pred_tumor_labels)

    # per-clone counts
    label_counts = anns_sub[cell_label].value_counts()
    clone_cols = sorted([c for c in label_counts.index if c.startswith("clone")])
    metric = {
        "total": total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "ROC-AUC (hard)": auc_hard,
        "ROC-AUC (soft)": auc_soft,
        "ARI": ari,
        "#normal": int(label_counts.get("normal", 0)),
    }
    for c in clone_cols:
        metric[f"#{c}"] = int(label_counts.get(c, 0))
    metric["#NA"] = na_count
    return metric


def evaluate_malignant_accuracy(
    anns: pd.DataFrame,
    cell_label="cell_label",
    cell_type="cell_type",
    tumor_post="tumor",
):
    """Evaluate tumor/normal classification accuracy.

    Returns per-REP_ID rows plus an "ALL" aggregate row.
    Cells with Unknown cell_type are excluded from all metrics.
    """
    # overall metrics
    metric_all = _eval_subset(anns, cell_label, cell_type, tumor_post)
    metric_all["REP_ID"] = "ALL"

    # GT (rows) -> predicted (columns) crosstab with row/col totals
    known_mask = ~anns[cell_type].isin(NA_CELLTYPE)
    anns_known = anns[known_mask]
    ct = pd.crosstab(
        anns_known[cell_type],
        anns_known[cell_label],
        margins=True,
        margins_name="total",
    )

    auc_soft = metric_all.get("ROC-AUC (soft)")
    auc_soft_str = f"{auc_soft:.4f}" if auc_soft is not None else "NA"
    ari = metric_all.get("ARI")
    ari_str = f"{ari:.4f}" if ari is not None and not np.isnan(ari) else "NA"
    logging.info("tumor/normal evaluation:")
    logging.info(f"  precision = {metric_all['precision']:.4f}")
    logging.info(f"  recall    = {metric_all['recall']:.4f}")
    logging.info(f"  f1        = {metric_all['f1']:.4f}")
    logging.info(f"  accuracy  = {metric_all['accuracy']:.4f}")
    logging.info(f"  AUC_hard  = {metric_all['ROC-AUC (hard)']:.4f}")
    logging.info(f"  AUC_soft  = {auc_soft_str}")
    logging.info(f"  ARI       = {ari_str}")
    logging.info(f"  #NA       = {metric_all['#NA']}/{metric_all['total']}")
    logging.info(f"  crosstab (rows=GT {cell_type}, cols=pred {cell_label}):")
    for line in ct.to_string().splitlines():
        logging.info(f"    {line}")

    # per-REP_ID metrics
    rows = [metric_all]
    if "REP_ID" in anns.columns:
        for rep_id, anns_rep in anns.groupby("REP_ID", sort=True):
            m = _eval_subset(anns_rep, cell_label, cell_type, tumor_post)
            m["REP_ID"] = rep_id
            rows.append(m)

    metric_str = (
        f"precision={metric_all['precision']:.3f}, "
        f"recall={metric_all['recall']:.3f}, "
        f"f1={metric_all['f1']:.3f}, "
        f"accuracy={metric_all['accuracy']:.3f}"
    )
    return metric_all, metric_str, rows


def refine_labels_by_reference(
    anns: pd.DataFrame,
    ref_label="cell_type",
    cell_label="cell_label",
    out_label="refined_label",
):
    # TODO, ref label vals
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
