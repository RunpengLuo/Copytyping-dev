import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


TUMOR_LABELS = {"Tumor", "Tumor_cell", "tumor"}
UNKNOWN_LABELS = {"Unknown", "NA"}


def evaluate_malignant_accuracy(
    anns: pd.DataFrame,
    cell_label="cell_label",
    cell_type="cell_type",
    tumor_post="tumor",
):
    """
    true-positive: tumor cell be assigned to one of tumor clones.
    Cells with Unknown cell_type are excluded from all metrics.
    """
    known_mask = ~anns[cell_type].isin(UNKNOWN_LABELS)
    anns_known = anns[known_mask]
    na_count = int((anns[cell_label] == "NA").sum())

    y_true = anns_known[cell_type].isin(TUMOR_LABELS).to_numpy(dtype=int)
    y_pred_hard = (
        anns_known[cell_label].str.startswith("clone")
        | anns_known[cell_label].str.startswith("tumor")
    ).to_numpy(dtype=int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_hard, average="binary"
    )
    accuracy = accuracy_score(y_true, y_pred_hard)
    auc_from_hard = roc_auc_score(y_true, y_pred_hard)
    acc_report = classification_report(
        y_true, y_pred_hard, target_names=["normal", "tumor"]
    )
    cm = confusion_matrix(y_true, y_pred_hard)

    logging.info("==================================================")
    logging.info("ROC AUC (hard classification): %s", auc_from_hard)
    auc_from_post = None
    if tumor_post in anns_known:
        auc_from_post = roc_auc_score(y_true, anns_known[tumor_post])
        logging.info("ROC AUC (soft classification): %s", auc_from_post)

    logging.info("Confusion matrix:\n%s", cm)
    logging.info("\nClassification report:\n%s", acc_report)
    logging.info(
        f"#NA={na_count}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, accuracy={accuracy:.4f}"
    )

    metric_str = f"precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}, accuracy={accuracy:.3f}"
    # metric_str += f"\nROC-AUC (hard)={auc_from_hard:.4f}"
    # if not auc_from_post is None:
    #     metric_str += f", ROC-AUC (soft)={auc_from_post:.4f}"
    # metric_str += f"\nROC-AUC={auc_from_post:.4f}"

    metric = {
        "#NA": na_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "ROC-AUC (hard)": auc_from_hard,
        "ROC-AUC (soft)": auc_from_post,
    }
    return metric, metric_str


def refine_labels_by_reference(
    anns: pd.DataFrame,
    ref_label="cell_type",
    cell_label="cell_label",
    out_label="refined_label",
):
    # TODO, ref label vals
    num_na_before = (anns[cell_label] == "NA").sum()
    anns[out_label] = anns[cell_label]
    anns.loc[
        anns[ref_label].isin(TUMOR_LABELS) & (anns[cell_label] == "normal"), out_label
    ] = "NA"
    anns.loc[
        (~anns[ref_label].isin(TUMOR_LABELS)) & (anns[cell_label] != "normal"),
        out_label,
    ] = "NA"
    num_na_after = (anns[out_label] == "NA").sum()
    logging.info(
        f"#NA before/after refinement={num_na_before}->{num_na_after} / {len(anns)}"
    )
    return anns
