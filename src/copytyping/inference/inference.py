import argparse
import logging
import os

import numpy as np
import pandas as pd
import scanpy as sc

from copytyping.copytyping_parser import (
    add_arguments_inference,
    check_arguments_inference,
)
from copytyping.inference.cell_model import Cell_Model
from copytyping.inference.model_utils import compute_baseline_proportions
from copytyping.inference.spot_model import Spot_Model
from copytyping.inference.validation import (
    evaluate_clone_accuracy,
    evaluate_malignant_accuracy,
    refine_labels_by_reference,
)
from copytyping.io_utils import (
    load_modality_data,
    subset_model_params,
    subset_sx_data,
    union_align_barcodes,
)
from copytyping.plot.plot_cell import plot_cnv_heatmap
from copytyping.plot.plot_common import (
    plot_cross_heatmap,
    plot_posteriors,
    plot_rdr_baf_1d_aggregated,
)
from copytyping.plot.plot_visium import plot_visium_debug, plot_visium_panel
from copytyping.sx_data.sx_data import SX_Data
from copytyping.utils import (
    NA_CELLTYPE,
    SPATIAL_PLATFORMS,
    add_file_logging,
    read_whitelist_segments,
    setup_logging,
)


def run(args=None):
    logging.info("run copytyping inference")
    if isinstance(args, argparse.Namespace):
        args = vars(args)

    args = check_arguments_inference(args)
    sample = args["sample"]
    platform = args["platform"]
    data_types = args["data_types"]
    ref_label = args["ref_label"]
    out_dir = args["out_dir"]
    out_prefix = args["out_prefix"]
    if out_prefix == "":
        out_prefix = str(sample)
    os.makedirs(out_dir, exist_ok=True)
    add_file_logging(out_dir)
    plot_dir = os.path.join(out_dir, "plots")
    heatmap_dir = os.path.join(plot_dir, "heatmaps")
    heatmap_agg1_dir = os.path.join(heatmap_dir, "agg1")
    heatmap_aggx_dir = os.path.join(heatmap_dir, f"agg{args['heatmap_agg']}")
    scatter_dir = plot_dir
    validation_dir = plot_dir
    for d in [heatmap_agg1_dir, heatmap_aggx_dir, plot_dir]:
        os.makedirs(d, exist_ok=True)
    if platform in SPATIAL_PLATFORMS:
        visium_dir = plot_dir
    work_dir = os.path.join(out_dir, "work")
    os.makedirs(work_dir, exist_ok=True)
    verbosity = args["verbosity"]

    logging.info(f"sample={sample}, platform={platform}, data_types={data_types}")

    cell_type_df = None
    if args.get("cell_type") is not None:
        cell_type_df = pd.read_table(args["cell_type"])
        assert "BARCODE" in cell_type_df.columns
        assert ref_label in cell_type_df.columns

    aggr_mode = args.get("aggr_mode", "clust")
    data_sources = {}  # used by EM (seg or clust level)
    seg_data_sources = {}  # always seg level, used for plotting
    adatas = {}
    for data_type in data_types:
        barcodes_df, seg_df, X_seg, Y_seg, D_seg = load_modality_data(
            args[f"{data_type}_barcodes"],
            args[f"{data_type}_cnv_segments"],
            args[f"{data_type}_X_count"],
            args[f"{data_type}_A_allele"],
            args[f"{data_type}_B_allele"],
            args["bbc_phases"],
            data_type,
            args["seg_ucn"],
            solfile=args.get("solfile"),
        )

        if cell_type_df is not None:
            barcodes_df = pd.merge(
                left=barcodes_df,
                right=cell_type_df[["BARCODE", ref_label]],
                on="BARCODE",
                how="left",
                validate="1:1",
                sort=False,
            )
            barcodes_df[ref_label] = (
                barcodes_df[ref_label].fillna("Unknown").astype(str)
            )
            if barcodes_df[ref_label].isin(NA_CELLTYPE).all():
                logging.warning(
                    f"all {data_type} barcodes have "
                    f"uninformative {ref_label} labels "
                    f"(all in NA_CELLTYPE={NA_CELLTYPE})"
                )
                barcodes_df = barcodes_df.drop(columns=[ref_label])

        seg_sx = SX_Data(barcodes_df, seg_df, X_seg, Y_seg, D_seg)
        seg_data_sources[data_type] = seg_sx

        if aggr_mode == "clust":
            data_sources[data_type] = seg_sx.to_cluster_level()
        else:
            data_sources[data_type] = seg_sx

        if args.get(f"{data_type}_h5ad") is not None:
            adatas[data_type] = sc.read_h5ad(args[f"{data_type}_h5ad"])
            if cell_type_df is not None and ref_label in cell_type_df.columns:
                ct_map = cell_type_df.set_index("BARCODE")[ref_label]
                adata = adatas[data_type]
                if ref_label in adata.obs.columns:
                    logging.warning(
                        f"overwriting existing '{ref_label}' column "
                        f"in {data_type} h5ad obs with cell_type_df"
                    )
                adata.obs[ref_label] = (
                    adata.obs_names.to_series().map(ct_map).fillna("Unknown").values
                )

    # Union barcodes across modalities and realign matrices
    barcodes, modality_masks = union_align_barcodes(seg_data_sources, data_types)
    if aggr_mode == "clust":
        _, clust_masks = union_align_barcodes(data_sources, data_types)
    else:
        clust_masks = modality_masks

    # Merge cell_type info into union barcodes
    if cell_type_df is not None and ref_label in cell_type_df.columns:
        barcodes = pd.merge(
            left=barcodes,
            right=cell_type_df[["BARCODE", ref_label]],
            on="BARCODE",
            how="left",
            sort=False,
        )
        barcodes[ref_label] = barcodes[ref_label].fillna("Unknown").astype(str)

    cnv_blocks = seg_data_sources[data_types[0]].cnv_blocks

    method = args["method"]
    fit_mode = args.get("fit_mode", "hybrid")
    label = f"{method}-label"
    num_iters = args["niters"]
    posterior_thres = args["posterior_thres"]
    margin_thres = args["margin_thres"]

    bulk_props = np.array(list(map(float, cnv_blocks["PROPS"].iloc[0].split(";"))))
    if platform in SPATIAL_PLATFORMS:
        pi_init = bulk_props[1:]
        pi_init = pi_init / pi_init.sum()
    else:
        pi_init = bulk_props
    tau_prior_a = args["tau_prior_a"]
    tau_prior_b = args["tau_prior_b"]
    invphi_prior_a = args["invphi_prior_a"]
    invphi_prior_b = args["invphi_prior_b"]
    init_params = {
        "pi": pi_init,
        "tau0": tau_prior_a / tau_prior_b,  # prior mean
        "phi0": 1.0 / (invphi_prior_a / invphi_prior_b),  # 1/prior mean of invphi
        "pi_alpha": args["pi_alpha"],
        "tau_prior_a": tau_prior_a,
        "tau_prior_b": tau_prior_b,
        "invphi_prior_a": invphi_prior_a,
        "invphi_prior_b": invphi_prior_b,
        "theta_prior_a": args["theta_prior_a"],
        "theta_prior_b": args["theta_prior_b"],
    }
    fix_params = {"pi": False}
    for data_type in data_types:
        fix_params[f"{data_type}-inv_phi"] = not args["update_NB_dispersion"]
        fix_params[f"{data_type}-tau"] = not args["update_BB_dispersion"]

    if platform == "single_cell":
        model = Cell_Model
    elif platform == "spatial":
        model = Spot_Model
    else:
        raise ValueError(f"unknown platform={platform}")

    instance = model(
        barcodes,
        platform,
        data_types,
        data_sources,
        work_dir,
        out_prefix,
        verbosity,
        modality_masks=clust_masks,
    )
    model_params = instance.fit(
        fit_mode,
        fix_params=fix_params,
        init_params=init_params,
        max_iter=num_iters,
    )
    anns, clone_props = instance.predict(
        fit_mode,
        model_params,
        label=label,
        posterior_thres=posterior_thres,
        margin_thres=margin_thres,
        tumorprop_threshold=args["tumorprop_threshold"],
    )
    logging.info(f"clone fractions: {clone_props}")

    if aggr_mode == "clust":
        is_normal = getattr(instance, "_init_is_normal", None)
        for data_type in data_types:
            clust_obj = data_sources[data_type]
            if not hasattr(clust_obj, "cluster_ids"):
                continue
            seg_sx = seg_data_sources[data_type]
            lam_key = f"{data_type}-lambda"
            if lam_key in model_params and is_normal is not None:
                model_params[lam_key] = compute_baseline_proportions(
                    seg_sx.X, seg_sx.T, is_normal
                )
            for disp_key in [
                f"{data_type}-tau",
                f"{data_type}-inv_phi",
            ]:
                if disp_key in model_params and len(model_params[disp_key]) > 0:
                    val = model_params[disp_key][0]
                    n_seg = (
                        seg_sx.nrows_imbalanced
                        if "tau" in disp_key
                        else seg_sx.nrows_aneuploid
                    )
                    model_params[disp_key] = np.full(n_seg, val, dtype=np.float32)

    # ---- Evaluation (per-rep) ----
    metric = {}
    metric_str = ""
    if ref_label in barcodes.columns:
        logging.info("evaluate performance against reference labels")
        if args["refine_label_by_reference"]:
            anns = refine_labels_by_reference(
                anns, ref_label, label, f"{label}-refined"
            )
        tumor_post = "tumor_purity" if platform in SPATIAL_PLATFORMS else "tumor"
        metric, metric_str, eval_rows = evaluate_malignant_accuracy(
            anns,
            cell_label=label,
            cell_type=ref_label,
            tumor_post=tumor_post,
        )
        for row in eval_rows:
            row["SAMPLE"] = sample
            row["platform"] = platform
        eval_df = pd.DataFrame(eval_rows)
        front_cols = ["SAMPLE", "platform", "REP_ID"]
        other_cols = [c for c in eval_df.columns if c not in front_cols]
        eval_df = eval_df[front_cols + other_cols]
        eval_df.to_csv(
            os.path.join(out_dir, f"{out_prefix}.{platform}.evaluation.tsv"),
            sep="\t",
            header=True,
            index=False,
        )

        # Clone-level evaluation (ARI)
        clone_metric = evaluate_clone_accuracy(
            anns,
            pred_label=label,
            gt_label=ref_label,
        )
        if clone_metric:
            for k, v in clone_metric.items():
                if k != "crosstab":
                    eval_rows[0][k] = v
                    metric[k] = v
            eval_df = pd.DataFrame(eval_rows)
            front_cols = ["SAMPLE", "platform", "REP_ID"]
            other_cols = [c for c in eval_df.columns if c not in front_cols]
            eval_df = eval_df[front_cols + other_cols]
            eval_df.to_csv(
                os.path.join(
                    out_dir,
                    f"{out_prefix}.{platform}.evaluation.tsv",
                ),
                sep="\t",
                header=True,
                index=False,
            )

    anns.to_csv(
        os.path.join(out_dir, f"{out_prefix}.{platform}.annotations.tsv"),
        sep="\t",
        header=True,
        index=False,
    )

    # ---- Per-rep plotting ----
    wl_segments = read_whitelist_segments(args["region_bed"])
    genome_size = args["genome_size"]
    agg_size = args["heatmap_agg"]
    img_type = args["img_type"]
    dpi = args["dpi"]
    transparent = args["transparent"]
    rep_ids = anns["REP_ID"].unique()

    for rep_id in rep_ids:
        rep_mask = (anns["REP_ID"] == rep_id).to_numpy()
        anns_rep = anns.loc[rep_mask].reset_index(drop=True)
        params_rep = subset_model_params(model_params, rep_mask, data_types)
        rep_tag = f".{rep_id}" if len(rep_ids) > 1 else ""

        # posteriors
        plot_posteriors(
            anns_rep,
            os.path.join(
                validation_dir,
                f"{out_prefix}.{platform}{rep_tag}.posteriors.png",
            ),
            lab_type=ref_label if ref_label in anns_rep else label,
        )

        # cross heatmap (only if ref_label available)
        if ref_label in anns_rep.columns:
            plot_cross_heatmap(
                anns_rep,
                sample,
                os.path.join(
                    validation_dir,
                    f"{out_prefix}.{platform}{rep_tag}.cross_heatmap.png",
                ),
                acol=label,
                bcol=ref_label,
            )

        # per-data_type plots
        for data_type in data_types:
            sx_rep = subset_sx_data(seg_data_sources[data_type], rep_mask)
            for val in ["BAF", "log2RDR"]:
                if val == "log2RDR" and f"{data_type}-lambda" not in params_rep:
                    continue
                for my_label in [label, ref_label]:
                    if my_label not in anns_rep:
                        continue
                    agg_levels = [
                        (1, heatmap_agg1_dir),
                        (agg_size, heatmap_aggx_dir),
                    ]
                    for agg, agg_dir in agg_levels:
                        plot_cnv_heatmap(
                            sample,
                            data_type,
                            cnv_blocks,
                            sx_rep,
                            anns_rep,
                            wl_segments,
                            proportions=params_rep.get(f"{data_type}-theta", None),
                            val=val,
                            base_props=params_rep.get(f"{data_type}-lambda", None),
                            agg_size=agg,
                            lab_type=my_label,
                            filename=os.path.join(
                                agg_dir,
                                f"{out_prefix}.{platform}{rep_tag}"
                                f".{val}_heatmap.{data_type}"
                                f".{my_label}.{img_type}",
                            ),
                            dpi=dpi,
                            figsize=(20, 6 if agg > 1 else 15),
                            transparent=transparent,
                        )

            plot_rdr_baf_1d_aggregated(
                sx_rep,
                anns_rep,
                params_rep.get(f"{data_type}-lambda", None),
                sample,
                data_type,
                genome_size,
                mask_cnp=False,
                lab_type=label,
                filename=os.path.join(
                    scatter_dir,
                    f"{out_prefix}.{platform}{rep_tag}"
                    f".1d_scatter.{data_type}.{label}.pdf",
                ),
            )

    if args["umap"]:
        pass  # not yet implemented

    # ---- Visium spatial plots (already per-rep) ----
    if platform in SPATIAL_PLATFORMS:
        gex_adata = adatas.get("gex", adatas[data_types[0]])
        anns_indexed = anns.set_index("BARCODE")
        visium_slices = []
        for rep_id in rep_ids:
            anns_rep = anns[anns["REP_ID"] == rep_id]
            vis_adata = gex_adata[
                gex_adata.obs_names.isin(anns_rep["BARCODE"].values)
            ].copy()
            anns_vis = anns_indexed.reindex(vis_adata.obs_names)
            if ref_label not in anns_vis.columns:
                anns_vis[ref_label] = "Unknown"
            visium_slices.append((rep_id, anns_vis, vis_adata))
        visium_title = ""
        if metric.get("ROC-AUC (soft)") is not None:
            visium_title += f"AUC={metric['ROC-AUC (soft)']:.3f}"
        plot_visium_panel(
            sample,
            visium_slices,
            visium_dir,
            spot_label=label,
            path_label=ref_label,
            dpi=dpi,
            title_info=visium_title,
        )
        if hasattr(instance, "param_trace") and instance.param_trace:
            plot_visium_debug(
                sample,
                visium_slices,
                instance.param_trace,
                barcodes=barcodes,
                data_type=data_types[0],
                out_dir=visium_dir,
                clones=seg_data_sources[data_types[0]].clones,
                ref_label=(ref_label if ref_label in barcodes.columns else None),
                dpi=dpi,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Copytyping inference",
        description="copytyping inference",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_arguments_inference(parser)
    args = parser.parse_args()
    setup_logging(args)
    run(args)
