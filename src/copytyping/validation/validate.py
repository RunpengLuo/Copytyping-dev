"""Standalone validation: evaluate clone labels against reference.

Usage:
    copytyping validate \
        --processed_data <dir> \
        --pred_labels <file.tsv> \
        --ref_labels <file.tsv> \
        --ref_label <column_name> \
        --pred_label <column_name> \
        -o <outdir>

Required columns in pred_labels: BARCODE, <pred_label>
Optional columns: tumor_purity, REP_ID

Required columns in ref_labels: BARCODE, <ref_label>
Optional columns: <ref_label>-tumor_purity
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from scipy import sparse

from copytyping.inference.model_utils import compute_baseline_proportions
from copytyping.io_utils import load_spatial_neighbors
from copytyping.plot.plot_common import (
    plot_cluster_observed_data,
    plot_crosstab,
    plot_purity_histograms,
)
from copytyping.sx_data.sx_data import SX_Data
from copytyping.utils import add_file_logging, setup_logging
from copytyping.validation.metrics import (
    compute_cluster_baf_metrics,
    compute_joincount_zscores,
    evaluate_malignant_accuracy,
    refine_labels_by_reference,
)


def add_arguments_validate(parser):
    parser.add_argument(
        "--processed_data",
        required=True,
        type=str,
        help="Directory with cnp_profile.tsv and X/Y/D.npz files",
    )
    parser.add_argument(
        "--pred_labels",
        required=True,
        type=str,
        help="TSV with BARCODE + predicted label columns",
    )
    parser.add_argument(
        "--pred_label",
        required=False,
        type=str,
        default="label",
        help="Column name for predicted labels (default: label)",
    )
    parser.add_argument(
        "--ref_labels",
        required=False,
        type=str,
        default=None,
        help="TSV with BARCODE + reference label columns",
    )
    parser.add_argument(
        "--ref_label",
        required=False,
        type=str,
        default="path_label",
        help="Column name for reference labels (default: path_label)",
    )
    parser.add_argument("--sample", required=True, type=str)
    parser.add_argument("--data_type", required=False, type=str, default="gex")
    parser.add_argument("--genome_size", required=False, type=str, default=None)
    parser.add_argument("--region_bed", required=False, type=str, default=None)
    parser.add_argument(
        "--h5ad",
        required=False,
        type=str,
        default=None,
        help="h5ad for spatial neighbors (joincount)",
    )
    parser.add_argument("--n_neighs", required=False, type=int, default=6)
    parser.add_argument("-o", "--out_dir", required=True, type=str)
    parser.add_argument("--dpi", required=False, type=int, default=200)
    parser.add_argument("-v", "--verbosity", required=False, type=int, default=1)


def run(args=None):
    if isinstance(args, argparse.Namespace):
        args = vars(args)

    sample = args["sample"]
    data_type = args["data_type"]
    proc_dir = args["processed_data"]
    out_dir = args["out_dir"]
    pred_label = args["pred_label"]
    ref_label = args["ref_label"]
    os.makedirs(out_dir, exist_ok=True)
    _file_handler = add_file_logging(out_dir)

    # ── Load processed data ──
    cnp_file = None
    for f in os.listdir(proc_dir):
        if f.endswith(".cnp_profile.tsv"):
            cnp_file = os.path.join(proc_dir, f)
            break
    assert cnp_file, f"No cnp_profile.tsv found in {proc_dir}"
    cnp_df = pd.read_csv(cnp_file, sep="\t")

    # Load X/Y/D if available
    sx_data = None
    x_file = [f for f in os.listdir(proc_dir) if f.endswith(f".{data_type}.X.npz")]
    if x_file:
        prefix = os.path.join(proc_dir, x_file[0].rsplit(".X.npz", 1)[0])
        X = sparse.load_npz(f"{prefix}.X.npz").toarray()
        Y = sparse.load_npz(f"{prefix}.Y.npz").toarray()
        D = sparse.load_npz(f"{prefix}.D.npz").toarray()
        logging.info(f"loaded count matrices: X/Y/D shape={X.shape}")

    # ── Load predictions ──
    pred_df = pd.read_csv(args["pred_labels"], sep="\t")
    assert "BARCODE" in pred_df.columns, "pred_labels must have BARCODE column"
    assert pred_label in pred_df.columns, f"pred_labels must have {pred_label} column"
    logging.info(
        f"predictions: {len(pred_df)} spots, labels: {pred_df[pred_label].value_counts().to_dict()}"
    )

    # ── Load reference ──
    ref_df = None
    if args.get("ref_labels"):
        ref_df = pd.read_csv(args["ref_labels"], sep="\t")
        assert "BARCODE" in ref_df.columns
        assert ref_label in ref_df.columns

    # ── Merge ──
    anns = pred_df.copy()
    if ref_df is not None:
        merge_cols = ["BARCODE", ref_label]
        ref_purity_col = f"{ref_label}-tumor_purity"
        if ref_purity_col in ref_df.columns:
            merge_cols.append(ref_purity_col)
        anns = anns.merge(ref_df[merge_cols], on="BARCODE", how="left")

    # ── Evaluate ──
    metric = {"SAMPLE": sample}
    if ref_df is not None and ref_label in anns.columns:
        purity_col = "tumor_purity" if "tumor_purity" in anns.columns else None
        m = evaluate_malignant_accuracy(
            anns,
            qry_label=pred_label,
            ref_label=ref_label,
            tumor_post=purity_col if purity_col else "tumor",
        )
        metric.update(m)

        # Purity comparison
        ref_purity_col = f"{ref_label}-tumor_purity"
        if ref_purity_col in anns.columns and "tumor_purity" in anns.columns:
            valid = anns[ref_purity_col].notna() & anns["tumor_purity"].notna()
            if valid.sum() > 10:
                rp = anns.loc[valid, ref_purity_col].values
                ip = anns.loc[valid, "tumor_purity"].values
                corr = np.corrcoef(rp, ip)[0, 1]
                mae = np.mean(np.abs(rp - ip))
                metric["purity_corr"] = corr
                metric["purity_MAE"] = mae
                logging.info(f"purity: corr={corr:.4f}, MAE={mae:.4f}")

        # Crosstab
        plot_crosstab(
            anns,
            sample,
            os.path.join(out_dir, f"{sample}.crosstab.png"),
            metric=metric,
            acol=pred_label,
            bcol=ref_label,
        )

        # Refine
        anns = refine_labels_by_reference(
            anns, ref_label, pred_label, f"{pred_label}-refined"
        )

    # ── Joincount ──
    if args.get("h5ad") and "REP_ID" in anns.columns:
        spatial_graphs = load_spatial_neighbors(args["h5ad"], n_neighs=args["n_neighs"])
        jc = compute_joincount_zscores(anns, pred_label, spatial_graphs, [data_type])
        metric.update(jc)

    # ── Save evaluation ──
    eval_df = pd.DataFrame([metric])
    eval_df.to_csv(
        os.path.join(out_dir, f"{sample}.evaluation.tsv"),
        sep="\t",
        index=False,
        na_rep="",
    )
    logging.info(f"saved evaluation to {out_dir}/{sample}.evaluation.tsv")

    # ── Cluster obs + BAF metrics ──
    if x_file and sx_data is None:
        # Build SX_Data from loaded matrices
        barcodes_df = pred_df[["BARCODE"]].copy()
        if "REP_ID" in pred_df.columns:
            barcodes_df["REP_ID"] = pred_df["REP_ID"]
        else:
            barcodes_df["REP_ID"] = "R1"
        sx_data = SX_Data(barcodes_df, cnp_df, X, Y, D, baf_clip=0.1)
        raw_clust = sx_data.to_cluster_level()

        is_normal = (anns[pred_label] == "normal").values
        raw_lambda = (
            compute_baseline_proportions(raw_clust.X, raw_clust.T, is_normal)
            if is_normal.sum() > 0
            else None
        )

        for obs_label in [pred_label] + ([ref_label] if ref_df is not None else []):
            if obs_label not in anns.columns:
                continue
            baf_metrics = compute_cluster_baf_metrics(raw_clust, anns[obs_label].values)
            for g, m in sorted(baf_metrics.items()):
                logging.info(
                    f"  cluster {g} ({obs_label}): within_var={m['within_var']:.4f}"
                )
            plot_cluster_observed_data(
                raw_clust,
                anns,
                sample,
                os.path.join(
                    out_dir, f"{sample}.{data_type}.cluster_obs.{obs_label}.pdf"
                ),
                label_col=obs_label,
                baf_metrics=baf_metrics,
                base_props=raw_lambda,
                dpi=args["dpi"],
            )

    # ── Purity histograms ──
    if "tumor_purity" in anns.columns:
        plot_purity_histograms(
            anns,
            sample,
            os.path.join(out_dir, f"{sample}.purity_histograms.pdf"),
            label_col=pred_label,
            dpi=args["dpi"],
        )

    logging.info("validation done")
    logging.root.removeHandler(_file_handler)
    _file_handler.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate clone labels")
    add_arguments_validate(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)
