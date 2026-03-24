#!/usr/bin/env python3
"""
Evaluate copytyping predictions against simulation ground truth.

Computes clone-level classification metrics beyond binary tumor/normal AUC:
  - Adjusted Rand Index (ARI)
  - Clone-level accuracy, precision, recall, F1
  - Confusion matrix
  - Theta (tumor purity) correlation

Usage:
    python simulations/evaluate_simulation.py \
        --ground_truth simulations/sim_HT112C1_pure/ground_truth.tsv \
        --annotations simulations/sim_HT112C1_pure/outs_hybrid/sample.VISIUM.annotations.tsv \
        -o simulations/sim_HT112C1_pure/outs_hybrid/clone_evaluation.tsv
"""

import argparse
import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    adjusted_rand_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def evaluate(ground_truth_file, annotations_file, label_col="copytyping-label"):
    gt = pd.read_csv(ground_truth_file, sep="\t")
    anns = pd.read_csv(annotations_file, sep="\t")

    # merge on barcode
    merged = pd.merge(gt, anns, on="BARCODE", how="inner")
    logging.info(f"Matched {len(merged)} spots")

    y_true = merged["true_label"].values
    y_pred = merged[label_col].values

    # exclude NA predictions
    valid = y_pred != "NA"
    n_na = (~valid).sum()
    logging.info(f"NA predictions: {n_na}/{len(merged)}")
    y_true_v = y_true[valid]
    y_pred_v = y_pred[valid]

    # ARI (clone-level)
    ari = adjusted_rand_score(y_true_v, y_pred_v)
    logging.info(f"Adjusted Rand Index (ARI): {ari:.4f}")

    # clone-level accuracy
    acc = accuracy_score(y_true_v, y_pred_v)
    logging.info(f"Clone-level accuracy: {acc:.4f}")

    # per-clone F1
    all_labels = sorted(set(y_true_v) | set(y_pred_v))
    f1_macro = f1_score(y_true_v, y_pred_v, average="macro", labels=all_labels)
    f1_weighted = f1_score(y_true_v, y_pred_v, average="weighted", labels=all_labels)
    logging.info(f"F1 (macro): {f1_macro:.4f}")
    logging.info(f"F1 (weighted): {f1_weighted:.4f}")

    # classification report
    report = classification_report(y_true_v, y_pred_v, labels=all_labels)
    logging.info(f"\nClassification report:\n{report}")

    # confusion matrix
    cm = confusion_matrix(y_true_v, y_pred_v, labels=all_labels)
    cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
    logging.info(f"\nConfusion matrix (rows=true, cols=pred):\n{cm_df}")

    # theta correlation (if available)
    theta_corr = None
    if "true_theta" in merged.columns and "tumor_purity" in merged.columns:
        theta_true = merged["true_theta"].values
        theta_pred = merged["tumor_purity"].values
        pearson_r, pearson_pval = pearsonr(theta_true, theta_pred)
        spearman_rho, _ = spearmanr(theta_true, theta_pred)
        theta_mae = np.mean(np.abs(theta_true - theta_pred))
        logging.info(f"Theta Pearson r={pearson_r:.4f} (p={pearson_pval:.2e})")
        logging.info(f"Theta Spearman rho={spearman_rho:.4f}")
        logging.info(f"Theta MAE={theta_mae:.4f}")
        theta_corr = {
            "theta_pearson_r": pearson_r,
            "theta_spearman_rho": spearman_rho,
            "theta_mae": theta_mae,
        }

    metrics = {
        "n_spots": len(merged),
        "n_na": n_na,
        "ari": ari,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }
    if theta_corr:
        metrics.update(theta_corr)
    return metrics, cm_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate simulation results")
    parser.add_argument("--ground_truth", required=True, type=str)
    parser.add_argument("--annotations", required=True, type=str)
    parser.add_argument("--label_col", default="copytyping-label", type=str)
    parser.add_argument("-o", "--out_file", type=str, default=None)
    args = parser.parse_args()

    metrics, cm_df = evaluate(args.ground_truth, args.annotations, args.label_col)

    if args.out_file:
        pd.DataFrame([metrics]).to_csv(args.out_file, sep="\t", index=False)
        cm_df.to_csv(args.out_file.replace(".tsv", ".confusion.tsv"), sep="\t")
        logging.info(f"Saved to {args.out_file}")

    # print summary
    print("\n=== Simulation Evaluation ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
