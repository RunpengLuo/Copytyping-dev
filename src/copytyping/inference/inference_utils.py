import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from copytyping.utils import NA_CELLTYPE, is_tumor_label


##################################################
# validate
##################################################


def _eval_subset(
    anns_sub: pd.DataFrame, qry_label: str, ref_label: str, tumor_post: str
) -> dict:
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

    # ARI_clone: only when the reference resolves >1 tumor-like label (i.e. it
    # actually carries clone structure). For a plain cell-type reference with a
    # single tumor label this would compare clones to cell types -> leave NaN.
    ari_clone = np.nan
    tumor_ref_labels = {c for c in anns_bin[ref_label].unique() if is_tumor_label(c)}
    if len(tumor_ref_labels) > 1:
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
) -> dict:
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
