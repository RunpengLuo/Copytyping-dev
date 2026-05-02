"""Standalone validation: evaluate clone labels + generate all plots.

Usage:
    copytyping validate \
        --processed_data <dir> \
        --pred_labels <file.tsv> \
        --ref_labels <file.tsv> \
        --ref_label <column_name> \
        --pred_label <column_name> \
        --genome_size <file> --region_bed <file> \
        -o <outdir>
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from scipy import sparse

from copytyping.inference.inference_utils import adaptive_bin_bbc
from copytyping.inference.model_utils import compute_baseline_proportions
from copytyping.io_utils import load_spatial_neighbors
from copytyping.plot.plot_common import (
    plot_cluster_observed_data,
    plot_count_histograms,
    plot_crosstab,
    plot_init_baf_histograms,
    plot_purity_histograms,
)
from copytyping.plot.plot_heatmap import plot_cnv_heatmap
from copytyping.plot.plot_scatter_1d import plot_rdr_baf_1d_pseudobulk
from copytyping.plot.plot_scatter_2d import plot_scatter_2d_per_cell
from copytyping.sx_data.sx_data import SX_Data
from copytyping.utils import add_file_logging, setup_logging
from copytyping.validation.metrics import (
    compute_cluster_baf_metrics,
    compute_joincount_zscores,
    evaluate_init_normal,
    evaluate_malignant_accuracy,
    refine_labels_by_reference,
)


def _find_prefix(proc_dir):
    """Discover the output prefix from the metadata file in proc_dir."""
    for f in os.listdir(proc_dir):
        if f.endswith(".metadata.tsv"):
            return f.rsplit(".metadata.tsv", 1)[0]
    return None


def _load_npz(path):
    """Load .npz if it exists, else return None."""
    return dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else None


def _load_proc_data(proc_dir, data_type, prefix):
    """Load processed data from proc_dir using known prefix."""
    p = os.path.join(proc_dir, prefix)

    cnp_df = pd.read_csv(f"{p}.cnp_profile.tsv", sep="\t")

    dt_p = f"{p}.{data_type}"
    X = Y = D = None
    if os.path.exists(f"{dt_p}.seg.X.npz"):
        X = sparse.load_npz(f"{dt_p}.seg.X.npz").toarray()
        Y = sparse.load_npz(f"{dt_p}.seg.Y.npz").toarray()
        D = sparse.load_npz(f"{dt_p}.seg.D.npz").toarray()
        logging.info(f"loaded seg matrices: X/Y/D shape={X.shape}")

    bbc_df = X_bbc = Y_bbc = D_bbc = None
    bbc_p = f"{dt_p}.bbc"
    if os.path.exists(f"{bbc_p}.X.npz"):
        X_bbc = sparse.load_npz(f"{bbc_p}.X.npz")
        Y_bbc = sparse.load_npz(f"{bbc_p}.Y.npz")
        D_bbc = sparse.load_npz(f"{bbc_p}.D.npz")
        bbc_tsv = f"{bbc_p}.tsv.gz"
        if os.path.exists(bbc_tsv):
            bbc_df = pd.read_csv(bbc_tsv, sep="\t")
        logging.info(f"loaded BBC matrices: shape={X_bbc.shape}")

    params = _load_npz(f"{p}.model_params.npz") or {}
    trace = _load_npz(f"{p}.labeling_trace.npz")

    barcodes_df = None
    bc_path = f"{p}.barcodes.tsv"
    if os.path.exists(bc_path):
        barcodes_df = pd.read_csv(bc_path, sep="\t")

    return cnp_df, X, Y, D, bbc_df, X_bbc, Y_bbc, D_bbc, params, trace, barcodes_df


def run(args=None):
    if isinstance(args, argparse.Namespace):
        args = vars(args)

    sample = args["sample"]
    data_type = args["data_type"]
    proc_dir = args["processed_data"]
    out_dir = args["out_dir"]
    pred_label = args["pred_label"]
    ref_label = args["ref_label"]
    method = args.get("method", "copytyping")
    is_copytyping = method == "copytyping"
    dpi = args.get("dpi", 200)
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "plots")
    val_dir = os.path.join(out_dir, "validation")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    _file_handler = add_file_logging(out_dir)

    # ── Load processed data ──
    prefix = _find_prefix(proc_dir)
    assert prefix, f"No metadata.tsv found in {proc_dir}"
    cnp_df, X, Y, D, bbc_df, X_bbc, Y_bbc, D_bbc, params, trace, barcodes_df = (
        _load_proc_data(proc_dir, data_type, prefix)
    )

    # ── Load predictions ──
    pred_df = pd.read_csv(args["pred_labels"], sep="\t")
    assert "BARCODE" in pred_df.columns
    assert pred_label in pred_df.columns
    pred_df[pred_label] = pred_df[pred_label].fillna("NA").astype(str)
    logging.info(
        f"predictions: {len(pred_df)} spots, labels: {pred_df[pred_label].value_counts().to_dict()}"
    )

    # ── Load reference ──
    ref_df = None
    if args.get("ref_labels"):
        ref_df = pd.read_csv(args["ref_labels"], sep="\t")
        assert "BARCODE" in ref_df.columns
        assert ref_label in ref_df.columns

    # ── Build SX_Data (use saved barcodes to match matrix dimensions) ──
    seg_sx = None
    raw_clust = None
    if X is not None:
        if barcodes_df is None:
            barcodes_df = pred_df[["BARCODE"]].copy()
            barcodes_df["REP_ID"] = (
                pred_df["REP_ID"] if "REP_ID" in pred_df.columns else "R1"
            )
        seg_sx = SX_Data(barcodes_df, cnp_df, X, Y, D, baf_clip=0.1)
        raw_clust = seg_sx.to_cluster_level()

    # ── Align pred_df to proc barcodes ──
    if barcodes_df is not None and len(pred_df) != len(barcodes_df):
        proc_bcs = set(barcodes_df["BARCODE"])
        anns_base = barcodes_df.copy()
        pred_map = pred_df.set_index("BARCODE")
        for col in pred_df.columns:
            if col == "BARCODE":
                continue
            anns_base[col] = anns_base["BARCODE"].map(pred_map[col].to_dict())
        anns_base[pred_label] = anns_base[pred_label].fillna("NA").astype(str)
        pred_df = anns_base
        logging.info(f"aligned pred_df to proc barcodes: {len(pred_df)} spots")

    # ── Init normal evaluation (copytyping only) ──
    if is_copytyping:
        init_path = os.path.join(proc_dir, f"{prefix}.init_labels.tsv")
        if os.path.exists(init_path) and ref_df is not None:
            init_df = pd.read_csv(init_path, sep="\t")
            init_is_normal = (init_df["init_label"] == "normal").values
            evaluate_init_normal(
                init_is_normal,
                pred_df.merge(ref_df[["BARCODE", ref_label]], on="BARCODE", how="left"),
                ref_label,
            )
            if raw_clust is not None:
                plot_init_baf_histograms(
                    {data_type: raw_clust},
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
        if args.get("purity_cutoff")
        else []
    )
    post_cuts = (
        [float(x) for x in args["post_cutoff"].split(",")]
        if args.get("post_cutoff")
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
                tumor_post=purity_col if purity_col in anns.columns else "tumor",
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
    if args.get("h5ad") and "REP_ID" in anns.columns:
        spatial_graphs = load_spatial_neighbors(
            args["h5ad"], n_neighs=args.get("n_neighs", 6)
        )
        jc = compute_joincount_zscores(anns, pred_label, spatial_graphs, [data_type])
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

    # ── Determine baseline lambda ──
    is_normal = (
        (anns[best_label] == "normal").values
        if best_label in anns.columns
        else np.zeros(len(anns), dtype=bool)
    )
    seg_lambda = None
    if seg_sx is not None and is_normal.sum() > 0:
        seg_lambda = compute_baseline_proportions(seg_sx.X, seg_sx.T, is_normal)

    # ── Count histograms (copytyping only) ──
    if is_copytyping and seg_sx is not None:
        plot_count_histograms(
            {data_type: seg_sx},
            sample,
            os.path.join(val_dir, f"{sample}.count_histograms.pdf"),
            dpi=dpi,
        )

    # ── Cluster obs + 2d scatter ──
    if raw_clust is not None:
        raw_lambda = (
            compute_baseline_proportions(raw_clust.X, raw_clust.T, is_normal)
            if is_normal.sum() > 0
            else None
        )
        plot_labels = [pred_label] + (
            [ref_label] if ref_df is not None and ref_label in anns.columns else []
        )
        for obs_label in plot_labels:
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
                    val_dir, f"{sample}.{data_type}.cluster_obs.{obs_label}.pdf"
                ),
                label_col=obs_label,
                baf_metrics=baf_metrics,
                base_props=raw_lambda,
                dpi=dpi,
            )
        plot_scatter_2d_per_cell(
            raw_clust,
            anns,
            sample,
            os.path.join(val_dir, f"{sample}.{data_type}.cluster_2d.pdf"),
            label_col=best_label,
            base_props=raw_lambda,
            dpi=dpi,
        )

    # ── Heatmaps ──
    region_bed = args.get("region_bed")
    if seg_sx is not None and region_bed:
        heatmap_agg = args.get("heatmap_agg", 10)
        theta = params.get(f"{data_type}_theta")
        plot_labels = [best_label] + (
            [ref_label] if ref_df is not None and ref_label in anns.columns else []
        )
        for val in ["BAF", "log2RDR"]:
            if val == "log2RDR" and seg_lambda is None:
                continue
            for agg in [1, heatmap_agg]:
                # Unlabeled
                logging.info(f"  heatmap {val} agg={agg} unlabeled")
                plot_cnv_heatmap(
                    sample,
                    data_type,
                    cnp_df,
                    seg_sx,
                    None,
                    region_bed,
                    val=val,
                    base_props=seg_lambda,
                    agg_size=agg,
                    filename=os.path.join(
                        plot_dir,
                        f"{sample}.{val}_heatmap.{data_type}.agg{agg}.unlabeled.pdf",
                    ),
                    dpi=dpi,
                    figsize=(20, 6 if agg > 1 else 15),
                )
                # Per label
                for my_label in plot_labels:
                    logging.info(f"  heatmap {val} agg={agg} {my_label}")
                    plot_cnv_heatmap(
                        sample,
                        data_type,
                        cnp_df,
                        seg_sx,
                        anns,
                        region_bed,
                        proportions=theta,
                        val=val,
                        base_props=seg_lambda,
                        agg_size=agg,
                        lab_type=my_label,
                        filename=os.path.join(
                            plot_dir,
                            f"{sample}.{val}_heatmap.{data_type}.agg{agg}.{my_label}.pdf",
                        ),
                        dpi=dpi,
                        figsize=(20, 6 if agg > 1 else 15),
                    )

    # ── 1D scatter ──
    genome_size = args.get("genome_size")
    if seg_sx is not None and bbc_df is not None and genome_size and region_bed:
        agg_bbc = adaptive_bin_bbc(
            bbc_df,
            X_bbc,
            Y_bbc,
            D_bbc,
            seg_sx,
            args.get("min_snp_count", 300),
            args.get("max_bin_length", 5_000_000),
        )
        agg_lambda = (
            compute_baseline_proportions(agg_bbc.X, agg_bbc.T, is_normal)
            if is_normal.sum() > 0
            else None
        )
        plot_labels = [best_label] + (
            [ref_label] if ref_df is not None and ref_label in anns.columns else []
        )
        for my_label in plot_labels:
            logging.info(f"  1d scatter {my_label}")
            plot_rdr_baf_1d_pseudobulk(
                agg_bbc,
                anns,
                agg_lambda,
                sample,
                data_type,
                genome_size,
                haplo_blocks=cnp_df,
                region_bed=region_bed,
                lab_type=my_label,
                is_inferred=(my_label == best_label),
                filename=os.path.join(
                    plot_dir, f"{sample}.1d_scatter.{data_type}.{my_label}.pdf"
                ),
                platform="spatial",
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

    # ── Visium panel ──
    if args.get("h5ad") and "REP_ID" in anns.columns:
        import scanpy as sc_mod
        from copytyping.plot.plot_visium import plot_visium_panel

        h5ad_adata = sc_mod.read_h5ad(args["h5ad"])
        anns_indexed = anns.set_index("BARCODE")
        visium_slices = []
        for rep_id in sorted(anns["REP_ID"].dropna().unique()):
            anns_rep = anns[anns["REP_ID"] == rep_id]
            vis_adata = h5ad_adata[
                h5ad_adata.obs_names.isin(anns_rep["BARCODE"].values)
            ].copy()
            anns_vis = anns_indexed.reindex(vis_adata.obs_names)
            if ref_label not in anns_vis.columns:
                anns_vis[ref_label] = "Unknown"
            visium_slices.append((rep_id, anns_vis, vis_adata))
        # Best cutoff label for visium (if different from raw pred_label)
        bcl = best_label if best_label != pred_label else None
        bcm = best_row if best_label != pred_label and ref_df is not None else None
        plot_visium_panel(
            sample,
            visium_slices,
            plot_dir,
            spot_label=pred_label,
            path_label=ref_label,
            best_cutoff_label=bcl,
            best_cutoff_metrics=bcm,
            dpi=dpi,
        )

        # LOH BAF visium
        if raw_clust is not None:
            from copytyping.inference.inference_utils import compute_loh_baf
            from copytyping.plot.plot_visium import plot_visium_loh_baf

            loh_baf, loh_info = compute_loh_baf(raw_clust)
            plot_visium_loh_baf(
                sample,
                visium_slices,
                raw_clust,
                loh_baf,
                loh_info,
                os.path.join(val_dir, f"{sample}.visium_loh_baf.pdf"),
                dpi=dpi,
            )

        # Visium iters (copytyping only, requires labeling trace)
        if is_copytyping and trace is not None:
            from copytyping.plot.plot_visium import plot_visium_iters

            n_iters = int(trace["n_iters"][0])
            labeling_trace = []
            for i in range(n_iters):
                lt = {
                    "labels": trace[f"labels_{i}"],
                    "max_posterior": trace[f"max_posterior_{i}"],
                }
                if f"tumor_purity_{i}" in trace:
                    lt["tumor_purity"] = trace[f"tumor_purity_{i}"]
                labeling_trace.append(lt)
            clones_list = list(seg_sx.clones) if seg_sx else None
            plot_visium_iters(
                sample,
                visium_slices,
                labeling_trace,
                barcodes=pred_df,
                out_dir=val_dir,
                clones=clones_list,
                ref_label=ref_label if ref_df is not None else None,
                dpi=dpi,
            )

    logging.info("validation done")
    logging.root.removeHandler(_file_handler)
    _file_handler.close()


if __name__ == "__main__":
    from copytyping.copytyping_parser import add_arguments_validate

    parser = argparse.ArgumentParser(description="Validate clone labels")
    add_arguments_validate(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)
