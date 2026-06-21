import argparse
import logging
import os
from functools import partial

import numpy as np
import pandas as pd
from scipy import sparse

from copytyping.inference.model_utils import compute_rdr_baseline
from copytyping.io_utils import load_spatial_neighbors
from copytyping.plot.plot_visium import plot_visium_all
from copytyping.utils import add_file_logging, normalize_args, setup_logging

from analysis_plots import (
    compute_joincount_zscores,
    evaluate_init_normal,
    evaluate_malignant_accuracy,
    plot_count_histograms,
    plot_crosstab,
    plot_init_baf_histograms,
    plot_modality_panel,
    plot_purity_histograms,
    refine_labels_by_reference,
)
from sx_data import SX_Data


def _find_prefix(proc_dir):
    """Discover the output prefix from the metadata file in proc_dir."""
    for f in os.listdir(proc_dir):
        if f.endswith(".metadata.tsv"):
            return f.rsplit(".metadata.tsv", 1)[0]
    return None


def _load_npz(path):
    """Load .npz if it exists, else return None."""
    return dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else None


def _load_metadata(proc_dir, prefix):
    """Load metadata.tsv (key/value rows) into a dict."""
    path = os.path.join(proc_dir, f"{prefix}.metadata.tsv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t", header=None, names=["key", "value"])
    return dict(zip(df["key"], df["value"]))


def _load_shared_proc_data(proc_dir, prefix):
    """Load shared (modality-independent) processed data."""
    p = os.path.join(proc_dir, prefix)
    cnp_df = pd.read_csv(f"{p}.cnp_profile.tsv", sep="\t")
    params = _load_npz(f"{p}.model_params.npz") or {}
    trace = _load_npz(f"{p}.labeling_trace.npz")
    return cnp_df, params, trace


def _load_modality_proc_data(proc_dir, prefix, assay_type):
    """Load per-modality matrices: returns (X, Y, D, bbc_df, X_bbc, Y_bbc, D_bbc)."""
    assay_p = os.path.join(proc_dir, f"{prefix}.{assay_type}")

    X = Y = D = None
    if os.path.exists(f"{assay_p}.seg.X.npz"):
        X = sparse.load_npz(f"{assay_p}.seg.X.npz").toarray()
        Y = sparse.load_npz(f"{assay_p}.seg.Y.npz").toarray()
        D = sparse.load_npz(f"{assay_p}.seg.D.npz").toarray()
        logging.info(f"loaded {assay_type} seg matrices: X/Y/D shape={X.shape}")

    bbc_df = X_bbc = Y_bbc = D_bbc = None
    bbc_p = f"{assay_p}.bbc"
    if os.path.exists(f"{bbc_p}.X.npz"):
        X_bbc = sparse.load_npz(f"{bbc_p}.X.npz")
        Y_bbc = sparse.load_npz(f"{bbc_p}.Y.npz")
        D_bbc = sparse.load_npz(f"{bbc_p}.D.npz")
        bbc_tsv = f"{bbc_p}.tsv.gz"
        if os.path.exists(bbc_tsv):
            bbc_df = pd.read_csv(bbc_tsv, sep="\t")
        logging.info(f"loaded {assay_type} BBC matrices: shape={X_bbc.shape}")

    return X, Y, D, bbc_df, X_bbc, Y_bbc, D_bbc


def run(args=None):
    args = normalize_args(args)

    sample = args["sample"]
    proc_dir = args["processed_data"]
    out_dir = args["out_dir"]
    pred_label = args["pred_label"]
    ref_label = args["ref_label"]
    method = args["method"]
    is_copytyping = method == "copytyping"
    dpi = args["dpi"]
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "plots")
    val_dir = os.path.join(out_dir, "validation")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    _file_handler = add_file_logging(out_dir)

    # ── Load shared processed data ──
    prefix = _find_prefix(proc_dir)
    assert prefix, f"No metadata.tsv found in {proc_dir}"
    metadata = _load_metadata(proc_dir, prefix)
    assay_types = [d.strip() for d in metadata.get("assay_types", "gex").split(",")]
    logging.info(f"assay_types: {assay_types}")
    cnp_df, params, trace = _load_shared_proc_data(proc_dir, prefix)

    # ── Load predictions (also serves as the barcodes table) ──
    pred_df = pd.read_csv(args["pred_labels"], sep="\t")
    assert "BARCODE" in pred_df.columns
    assert pred_label in pred_df.columns
    pred_df[pred_label] = pred_df[pred_label].fillna("NA").astype(str)
    logging.info(
        f"predictions: {len(pred_df)} spots, labels: {pred_df[pred_label].value_counts().to_dict()}"
    )

    barcodes_df = pred_df[["BARCODE"]].copy()
    barcodes_df["REP_ID"] = pred_df["REP_ID"] if "REP_ID" in pred_df.columns else "R1"

    # ── Load reference ──
    ref_df = None
    if args["ref_labels"]:
        ref_df = pd.read_csv(args["ref_labels"], sep="\t")
        assert "BARCODE" in ref_df.columns
        assert ref_label in ref_df.columns

    # ── Build per-modality SX_Data ──
    seg_sx_by_assay = {}
    raw_clust_by_assay = {}
    bbc_by_assay = {}
    for assay in assay_types:
        X, Y, D, bbc_df_assay, X_bbc, Y_bbc, D_bbc = _load_modality_proc_data(
            proc_dir, prefix, assay
        )
        if X is None:
            logging.warning(f"no seg matrices for {assay}, skipping")
            continue
        seg_sx = SX_Data(barcodes_df, cnp_df, X, Y, D, baf_clip=0.1)
        seg_sx_by_assay[assay] = seg_sx
        raw_clust_by_assay[assay] = seg_sx.to_cluster_level()
        bbc_by_assay[assay] = (bbc_df_assay, X_bbc, Y_bbc, D_bbc)

    # ── Init normal evaluation (copytyping only) ──
    if is_copytyping and trace is not None and ref_df is not None:
        init_is_normal = np.asarray(trace["labels_0"]) == "normal"
        evaluate_init_normal(
            init_is_normal,
            pred_df.merge(ref_df[["BARCODE", ref_label]], on="BARCODE", how="left"),
            ref_label,
        )
        if raw_clust_by_assay:
            plot_init_baf_histograms(
                raw_clust_by_assay,
                init_is_normal,
                sample,
                val_dir,
                dpi=dpi,
            )

    # ── Merge ref ──
    anns = pred_df.copy()
    if ref_df is not None:
        merge_cols = ["BARCODE", ref_label]
        ref_purity_col = f"{ref_label}-tumor_purity"
        if ref_purity_col in ref_df.columns:
            merge_cols.append(ref_purity_col)
        # Drop existing ref columns to avoid _x/_y collision
        drop_cols = [c for c in merge_cols if c != "BARCODE" and c in anns.columns]
        if drop_cols:
            anns = anns.drop(columns=drop_cols)
        anns = anns.merge(ref_df[merge_cols], on="BARCODE", how="left")

    # ── Cutoff sweep: (purity_cutoff, post_cutoff) grid ──
    pcuts = (
        [float(x) for x in args["purity_cutoff"].split(",")]
        if args["purity_cutoff"]
        else []
    )
    post_cuts = (
        [float(x) for x in args["post_cutoff"].split(",")]
        if args["post_cutoff"]
        else []
    )
    has_purity = "tumor_purity" in anns.columns
    has_post = "max_posterior" in anns.columns

    base_labels = anns[pred_label].values.copy()
    purity = anns["tumor_purity"].values if has_purity else None
    max_post = anns["max_posterior"].values if has_post else None

    # Build (pcut, post_cut) combos: (0,0) = raw, then all grid points
    cutoff_combos = [(0.0, 0.0)]
    if has_purity:
        for pc in pcuts:
            cutoff_combos.append((pc, 0.0))
    if has_purity and has_post:
        for pc in pcuts:
            for pt in post_cuts:
                cutoff_combos.append((pc, pt))

    for pc, pt in cutoff_combos:
        labels = base_labels.copy()
        # 1. purity <= cutoff -> normal
        if pc > 0 and purity is not None:
            low_pur = (labels != "NA") & (~np.isnan(purity)) & (purity <= pc)
            labels[low_pur] = "normal"
        # 2. remaining tumor spots with max_posterior <= cutoff -> normal
        if pt > 0 and max_post is not None:
            is_tumor = (labels != "NA") & (labels != "normal")
            low_post = is_tumor & (max_post <= pt)
            labels[low_post] = "normal"
        suffix = ""
        if pc > 0:
            suffix += f"_pcut{pc}"
        if pt > 0:
            suffix += f"_post{pt}"
        col = f"{pred_label}{suffix}" if suffix else pred_label
        if col != pred_label:
            anns[col] = labels

    # ── Evaluate ──
    eval_rows = []
    base_meta = {"SAMPLE": sample}
    ref_purity_col = f"{ref_label}-tumor_purity"
    if ref_df is not None and ref_purity_col in anns.columns and has_purity:
        valid = anns[ref_purity_col].notna() & anns["tumor_purity"].notna()
        if valid.sum() > 10:
            rp = anns.loc[valid, ref_purity_col].values
            ip = anns.loc[valid, "tumor_purity"].values
            base_meta["purity_corr"] = np.corrcoef(rp, ip)[0, 1]
            base_meta["purity_MAE"] = np.mean(np.abs(rp - ip))
            logging.info(
                f"purity: corr={base_meta['purity_corr']:.4f}, MAE={base_meta['purity_MAE']:.4f}"
            )

    eval_labels = [pred_label]
    for pc, pt in cutoff_combos:
        suffix = ""
        if pc > 0:
            suffix += f"_pcut{pc}"
        if pt > 0:
            suffix += f"_post{pt}"
        col = f"{pred_label}{suffix}" if suffix else pred_label
        if col not in eval_labels:
            eval_labels.append(col)

    best_label = pred_label
    if ref_df is not None and ref_label in anns.columns:
        purity_col = "tumor_purity" if has_purity else "tumor"
        for eval_label in eval_labels:
            m = evaluate_malignant_accuracy(
                anns,
                qry_label=eval_label,
                ref_label=ref_label,
                tumor_post=purity_col,
            )
            eval_rows.append({**base_meta, "label": eval_label, **m})
        best_row = max(
            eval_rows,
            key=lambda r: r.get("ARI_clone", 0)
            if not np.isnan(r.get("ARI_clone", 0))
            else 0,
        )
        best_label = best_row["label"]
        plot_crosstab(
            anns,
            sample,
            os.path.join(val_dir, f"{sample}.crosstab.png"),
            metric=best_row,
            acol=best_label,
            bcol=ref_label,
        )
        anns = refine_labels_by_reference(
            anns, ref_label, pred_label, f"{pred_label}-refined"
        )
    else:
        eval_rows.append(base_meta)

    # ── Joincount ──
    if args["h5ad"] and "REP_ID" in anns.columns:
        spatial_graphs = load_spatial_neighbors(args["h5ad"], n_neighs=args["n_neighs"])
        jc = compute_joincount_zscores(anns, pred_label, spatial_graphs, assay_types)
        for row in eval_rows:
            row.update(jc)

    # ── Save evaluation ──
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(
        os.path.join(out_dir, f"{sample}.evaluation.tsv"),
        sep="\t",
        index=False,
        na_rep="",
    )
    cols = [
        c
        for c in [
            "label",
            "f1",
            "precision",
            "recall",
            "AUC_hard",
            "AUC_soft",
            "ARI_binary",
            "ARI_clone",
        ]
        if c in eval_df.columns
    ]
    if cols:
        logging.info(f"\n{eval_df[cols].to_string(index=False)}")

    # ── RDR baseline: validate has no model instance, so reference = "normal"
    # labeled cells, no CNP correction ──
    baseline_fn = partial(
        compute_rdr_baseline, ref_cells=(anns[best_label] == "normal").to_numpy()
    )

    # ── Count histograms (copytyping only, all modalities) ──
    # Skip per-modality plots when seg matrices weren't saved at inference
    # time (default; enable with --save_processed_data). Inference plots
    # these inline anyway.
    if not seg_sx_by_assay:
        logging.info("skipping per-modality plots (no seg matrices in processed_data)")
    if is_copytyping and seg_sx_by_assay:
        plot_count_histograms(
            seg_sx_by_assay,
            sample,
            os.path.join(val_dir, f"{sample}.count_histograms.pdf"),
            dpi=dpi,
        )

    # ── Per-modality plots ──
    plot_labels = [best_label] + (
        [ref_label] if ref_df is not None and ref_label in anns.columns else []
    )
    for assay_type, seg_sx in seg_sx_by_assay.items():
        plot_modality_panel(
            sample=sample,
            assay_type=assay_type,
            prefix=sample,
            plot_dir=plot_dir,
            seg_sx=seg_sx,
            raw_clust=raw_clust_by_assay[assay_type],
            bbc_data=bbc_by_assay[assay_type],
            cnv_blocks=cnp_df,
            anns=anns,
            baseline_fn=baseline_fn,
            primary_label=best_label,
            plot_labels=plot_labels,
            theta=params.get(f"{assay_type}_theta"),
            region_bed=args["region_bed"],
            genome_size=args["genome_size"],
            dpi=dpi,
            heatmap_agg=args["heatmap_agg"],
            min_snp_count=args["min_snp_count"],
            max_bin_length=args["max_bin_length"],
            platform_str="spatial",
            compute_baf_metrics=True,
            ascn_profile=args["ascn_profile"],
        )

    # ── Purity histograms (copytyping only) ──
    if is_copytyping and has_purity:
        plot_purity_histograms(
            anns,
            sample,
            os.path.join(val_dir, f"{sample}.purity_histograms.pdf"),
            label_col=best_label,
            dpi=dpi,
        )
    if is_copytyping and has_post:
        plot_purity_histograms(
            anns,
            sample,
            os.path.join(val_dir, f"{sample}.posterior_histograms.pdf"),
            label_col=best_label,
            purity_col="max_posterior",
            xlabel="max posterior",
            dpi=dpi,
        )

    # ── Visium plots (spatial only, single gex modality) ──
    if args["h5ad"] and "REP_ID" in anns.columns and raw_clust_by_assay:
        labeling_trace = None
        if is_copytyping and trace is not None:
            n_iters = int(trace["n_iters"][0])
            labeling_trace = [
                {
                    "labels": trace[f"labels_{i}"],
                    "max_posterior": trace[f"max_posterior_{i}"],
                    **(
                        {"tumor_purity": trace[f"tumor_purity_{i}"]}
                        if f"tumor_purity_{i}" in trace
                        else {}
                    ),
                }
                for i in range(n_iters)
            ]
        gex_clust = next(iter(raw_clust_by_assay.values()))
        plot_visium_all(
            sample=sample,
            anns=anns,
            h5ad_source=args["h5ad"],
            raw_clust=gex_clust,
            plot_dir=plot_dir,
            spot_label=pred_label,
            ref_label=ref_label,
            best_cutoff_label=best_label if best_label != pred_label else None,
            best_cutoff_metrics=best_row
            if best_label != pred_label and ref_df is not None
            else None,
            labeling_trace=labeling_trace,
            barcodes=pred_df,
            clones=list(gex_clust.clones),
            dpi=dpi,
        )

    logging.info("validation done")
    logging.root.removeHandler(_file_handler)
    _file_handler.close()


def build_parser():
    """Standalone argparse for the relocated validate workflow.

    Tuning knobs left as SUPPRESS are filled from the packaged copytyping.yaml
    by normalize_args; --ref_labels/--ref_label/--h5ad carry explicit defaults.
    """
    p = argparse.ArgumentParser(prog="validate", description=__doc__)
    p.add_argument(
        "--processed_data",
        required=True,
        help="dir with cnp_profile.tsv, X/Y/D.npz, model_params.npz",
    )
    p.add_argument(
        "--pred_labels",
        required=True,
        help="TSV with BARCODE + predicted label columns",
    )
    p.add_argument("--sample", required=True, help="sample name")
    p.add_argument("--genome_size", required=True, help="chromosome sizes file")
    p.add_argument("--region_bed", required=True, help="chromosome regions BED file")
    p.add_argument("-o", "--out_dir", required=True, help="output directory")
    p.add_argument(
        "--ref_labels", default=None, help="TSV with BARCODE + reference label columns"
    )
    p.add_argument(
        "--ref_label",
        default="path_label",
        help="reference label column (default: path_label)",
    )
    p.add_argument(
        "--h5ad", default=None, help="h5ad for spatial neighbors (joincount + visium)"
    )
    p.add_argument(
        "--pred_label", default=argparse.SUPPRESS, help="predicted label column"
    )
    p.add_argument(
        "--method", default=argparse.SUPPRESS, help="method that produced the labels"
    )
    p.add_argument(
        "--purity_cutoff",
        default=argparse.SUPPRESS,
        help="comma-separated purity cutoffs",
    )
    p.add_argument(
        "--post_cutoff",
        default=argparse.SUPPRESS,
        help="comma-separated max_posterior cutoffs",
    )
    p.add_argument("--dpi", type=int, default=argparse.SUPPRESS, help="plot DPI")
    p.add_argument(
        "--heatmap_agg",
        type=int,
        default=argparse.SUPPRESS,
        help="cells per heatmap row",
    )
    p.add_argument(
        "--min_snp_count",
        type=int,
        default=argparse.SUPPRESS,
        help="min SNP count per adaptive bin",
    )
    p.add_argument(
        "--max_bin_length",
        type=int,
        default=argparse.SUPPRESS,
        help="max adaptive bin length (bp)",
    )
    p.add_argument(
        "--n_neighs",
        type=int,
        default=argparse.SUPPRESS,
        help="number of spatial neighbors",
    )
    p.add_argument(
        "--ascn_profile",
        action="store_true",
        default=argparse.SUPPRESS,
        help="allele-specific CN profile track",
    )
    p.add_argument(
        "-v", "--verbosity", type=int, default=argparse.SUPPRESS, help="verbose level"
    )
    return p


def main(argv=None):
    args = normalize_args(build_parser().parse_args(argv))
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
