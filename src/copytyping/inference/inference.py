import argparse
import copy
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
from copytyping.inference.clustering import kmeans_copytyping, cluster_label_major_vote
from copytyping.inference.inference_utils import (
    adaptive_bin_bbc,
    annotate_adata_celltype,
)
from copytyping.inference.model_utils import (
    compute_baseline_proportions,
    prepare_params,
)
from copytyping.inference.spot_model import Spot_Model
from copytyping.inference.validation import (
    compute_cluster_baf_metrics,
    compute_joincount_zscores,
    evaluate_init_normal,
    evaluate_malignant_accuracy,
    refine_labels_by_reference,
)
from copytyping.io_utils import (
    load_modality_data,
    load_spatial_neighbors,
    union_align_barcodes,
)
from copytyping.plot.plot_heatmap import plot_cnv_heatmap
from copytyping.plot.plot_common import (
    plot_cluster_observed_data,
    plot_count_histograms,
    plot_crosstab,
    plot_init_baf_histograms,
    plot_purity_histograms,
)
from copytyping.plot.plot_scatter_1d import plot_rdr_baf_1d_pseudobulk
from copytyping.plot.plot_scatter_2d import plot_scatter_2d_per_cell
from copytyping.plot.plot_visium import (
    plot_visium_iters,
    plot_visium_panel,
)
from copytyping.sx_data.sx_data import SX_Data
from copytyping.utils import (
    SPATIAL_PLATFORMS,
    add_file_logging,
    save_cnp_profile,
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
    out_prefix = args["out_prefix"] or str(sample)
    os.makedirs(out_dir, exist_ok=True)
    _file_handler = add_file_logging(out_dir)
    plot_dir = os.path.join(out_dir, "plots")
    val_dir = os.path.join(out_dir, "validation")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    logging.info(f"sample={sample}, platform={platform}, data_types={data_types}")

    cell_type_df = None
    if args.get("cell_type") is not None:
        cell_type_df = pd.read_table(args["cell_type"])
        assert "BARCODE" in cell_type_df.columns
        assert ref_label in cell_type_df.columns

    min_snp_agg_bbc = args["min_snp_count"]
    max_len_agg_bbc = args["max_bin_length"]
    data_sources = {}
    raw_data_sources = {}
    seg_data_sources = {}
    agg_bbc_data_sources = {}
    adatas = {}
    spatial_graphs = {}
    exclude_set = set(args["exclude"].split(",")) if args.get("exclude") else None
    for data_type in data_types:
        barcodes_df, seg_df, X_seg, Y_seg, D_seg, bbc_df, X_bbc, Y_bbc, D_bbc = (
            load_modality_data(
                args[f"{data_type}_barcodes"],
                args[f"{data_type}_cnv_segments"],
                args[f"{data_type}_X_count"],
                args[f"{data_type}_A_allele"],
                args[f"{data_type}_B_allele"],
                args["bbc_phases"],
                data_type,
                args["seg_ucn"],
                solfile=args.get("solfile"),
                cell_type_df=cell_type_df,
                ref_label=ref_label,
                exclude_labels=exclude_set,
            )
        )

        seg_sx = SX_Data(
            barcodes_df, seg_df, X_seg, Y_seg, D_seg, baf_clip=args["baf_clip"]
        )
        seg_data_sources[data_type] = seg_sx
        agg_bbc_data_sources[data_type] = adaptive_bin_bbc(
            bbc_df,
            X_bbc,
            Y_bbc,
            D_bbc,
            seg_sx,
            min_snp_agg_bbc,
            max_len_agg_bbc,
        )

        if args.get(f"{data_type}_h5ad") is not None:
            adatas[data_type] = sc.read_h5ad(args[f"{data_type}_h5ad"])
            if cell_type_df is not None:
                annotate_adata_celltype(
                    adatas[data_type], cell_type_df, ref_label, data_type
                )
            if platform in SPATIAL_PLATFORMS and "spatial" in adatas[data_type].obsm:
                spatial_graphs[data_type] = load_spatial_neighbors(
                    args[f"{data_type}_h5ad"], n_neighs=args["n_neighs"]
                )

        raw_data_sources[data_type] = seg_sx.to_cluster_level()
        if data_type in spatial_graphs and args["smooth_k"] > 0:
            seg_sx_smoothed = copy.deepcopy(seg_sx)
            seg_sx_smoothed.apply_spatial_smoothing(
                spatial_graphs[data_type], args["smooth_k"]
            )
            data_sources[data_type] = seg_sx_smoothed.to_cluster_level()
        else:
            data_sources[data_type] = raw_data_sources[data_type]

    save_cnp_profile(
        seg_data_sources,
        os.path.join(out_dir, f"{out_prefix}.cnp_profile.tsv"),
    )

    barcodes, modality_masks = union_align_barcodes(data_sources, data_types)

    plot_count_histograms(
        seg_data_sources,
        sample,
        os.path.join(val_dir, f"{out_prefix}.count_histograms.pdf"),
        dpi=args["dpi"],
    )

    cnv_blocks = seg_data_sources[data_types[0]].cnv_blocks
    init_params, fix_params = prepare_params(args, cnv_blocks, platform, data_types)

    instance = {"single_cell": Cell_Model, "spatial": Spot_Model}[platform](
        barcodes,
        platform,
        data_types,
        data_sources,
        val_dir,
        out_prefix,
        args["verbosity"],
        modality_masks=modality_masks,
    )
    model_params, final_ll = instance.fit(
        fit_mode=args["fit_mode"],
        fix_params=fix_params,
        init_params=init_params,
        max_iter=args["niters"],
    )

    # init normal diagnostics
    if instance.labeling_trace:
        init_is_normal = instance.labeling_trace[0]["labels"] == "normal"
        evaluate_init_normal(init_is_normal, barcodes, ref_label)
        plot_init_baf_histograms(
            data_sources, init_is_normal, sample, val_dir, dpi=args["dpi"]
        )

    label = f"{args['method']}-label"
    if args["method"] == "kmeans":
        # K-means clustering on BAF+RDR features, using EM baseline
        K = data_sources[data_types[0]].K
        raw_labels = kmeans_copytyping(data_sources, barcodes, ref_label, K)
        anns = barcodes.copy(deep=True)
        if ref_label in barcodes.columns:
            anns, clone_props = cluster_label_major_vote(
                anns, raw_labels, cell_label=label, ref_label=ref_label
            )
        else:
            anns[label] = ["cluster" + str(x) for x in raw_labels]
            clone_props = {
                lab: np.mean(anns[label].to_numpy() == lab)
                for lab in sorted(anns[label].unique())
            }
    else:
        anns, clone_props = instance.predict(
            args["fit_mode"],
            model_params,
            label=label,
        )
    logging.info(
        "clone fractions: " + ", ".join(f"{k}={v:.3f}" for k, v in clone_props.items())
    )

    is_spot = platform in SPATIAL_PLATFORMS
    is_normal = (anns[label] == "normal").to_numpy()
    logging.info(f"baseline from predicted {label}: {is_normal.sum()} normals")

    # Baseline proportions from predicted normals (seg + agg-bbc level)
    seg_lambda, agg_bbc_lambda = {}, {}
    if is_normal.sum() > 0:
        for data_type in data_types:
            seg_lambda[data_type] = compute_baseline_proportions(
                seg_data_sources[data_type].X, seg_data_sources[data_type].T, is_normal
            )
            agg_bbc_lambda[data_type] = compute_baseline_proportions(
                agg_bbc_data_sources[data_type].X,
                agg_bbc_data_sources[data_type].T,
                is_normal,
            )

    eval_rows = []
    base_meta = {"SAMPLE": sample, "PLATFORM": platform, "log_likelihood": final_ll}

    # Joincount z-scores (shared across labels)
    jc_metric = {}
    if is_spot and spatial_graphs:
        jc_metric = compute_joincount_zscores(anns, label, spatial_graphs, data_types)

    # Evaluate
    if ref_label in barcodes.columns:
        purity_col = "tumor_purity" if is_spot else "tumor"
        m = evaluate_malignant_accuracy(
            anns,
            qry_label=label,
            ref_label=ref_label,
            tumor_post=purity_col if purity_col in anns.columns else "tumor",
        )
        eval_rows.append({**base_meta, **m, **jc_metric})
        anns = refine_labels_by_reference(anns, ref_label, label, f"{label}-refined")
    else:
        eval_rows.append({**base_meta, **jc_metric})

    # Compare inferred purity vs reference purity if available
    ref_purity_col = f"{ref_label}-tumor_purity"
    if ref_purity_col in anns.columns and "tumor_purity" in anns.columns:
        ref_pur = anns[ref_purity_col].dropna().values
        inf_pur = anns.loc[anns[ref_purity_col].notna(), "tumor_purity"].values
        corr = np.corrcoef(ref_pur, inf_pur)[0, 1]
        mae = np.mean(np.abs(ref_pur - inf_pur))
        logging.info(
            f"purity comparison vs {ref_purity_col}: "
            f"corr={corr:.4f}, MAE={mae:.4f}, n={len(ref_pur)}"
        )

    eval_df = pd.DataFrame(eval_rows)
    metric = eval_rows[0] if eval_rows else base_meta
    eval_df.to_csv(
        os.path.join(out_dir, f"{out_prefix}.{platform}.evaluation.tsv"),
        sep="\t",
        header=True,
        index=False,
        na_rep="",
    )

    anns.to_csv(
        os.path.join(out_dir, f"{out_prefix}.{platform}.annotations.tsv"),
        sep="\t",
        header=True,
        index=False,
    )

    # ---- Plotting (all reps combined per data_type) ----
    if ref_label in anns.columns:
        plot_crosstab(
            anns,
            sample,
            os.path.join(
                val_dir,
                f"{out_prefix}.{platform}.crosstab.png",
            ),
            metric=metric,
            acol=label,
            bcol=ref_label,
        )
    scatter_subtitle = f"min_snp_count={min_snp_agg_bbc}  max_bin_length={max_len_agg_bbc / 1e6:.1f}Mbp"
    plot_labels = [lb for lb in [label, ref_label] if lb in anns]

    logging.info("generating plots...")
    plot_purity_histograms(
        anns,
        sample,
        os.path.join(val_dir, f"{out_prefix}.{platform}.purity_histograms.pdf"),
        label_col=label,
        dpi=args["dpi"],
    )
    for data_type in data_types:
        raw_clust = raw_data_sources[data_type]
        raw_lambda = (
            compute_baseline_proportions(
                raw_clust.X,
                raw_clust.T,
                is_normal,
            )
            if is_normal.sum() > 0
            else None
        )
        obs_labels = [label]
        if ref_label in anns.columns:
            obs_labels = [ref_label] + obs_labels
        for obs_label in obs_labels:
            baf_metrics = compute_cluster_baf_metrics(raw_clust, anns[obs_label].values)
            for g, m in sorted(baf_metrics.items()):
                logging.info(
                    f"  [{data_type}] cluster {g} ({obs_label}): "
                    f"within_var={m['within_var']:.4f} within_var_tumor={m['within_var_tumor']:.4f}"
                )
            plot_cluster_observed_data(
                raw_clust,
                anns,
                sample,
                os.path.join(
                    val_dir, f"{out_prefix}.{data_type}.cluster_obs.{obs_label}.pdf"
                ),
                label_col=obs_label,
                baf_metrics=baf_metrics,
                base_props=raw_lambda,
                dpi=args["dpi"],
            )
        plot_scatter_2d_per_cell(
            raw_clust,
            anns,
            sample,
            os.path.join(val_dir, f"{out_prefix}.{data_type}.cluster_2d.pdf"),
            label_col=label,
            base_props=raw_lambda,
            dpi=args["dpi"],
        )
        seg_sx = seg_data_sources[data_type]
        for val in ["BAF", "log2RDR"]:
            if val == "log2RDR" and data_type not in seg_lambda:
                continue
            for agg in [1, args["heatmap_agg"]]:
                logging.info(f"  heatmap {val} agg={agg} unlabeled")
                plot_cnv_heatmap(
                    sample,
                    data_type,
                    cnv_blocks,
                    seg_sx,
                    None,
                    args["region_bed"],
                    val=val,
                    base_props=seg_lambda.get(data_type),
                    agg_size=agg,
                    filename=os.path.join(
                        plot_dir,
                        f"{out_prefix}.{platform}"
                        f".{val}_heatmap.{data_type}"
                        f".agg{agg}.unlabeled.{args['img_type']}",
                    ),
                    dpi=args["dpi"],
                    figsize=(20, 6 if agg > 1 else 15),
                    transparent=args["transparent"],
                )
        for my_label in plot_labels:
            for val in ["BAF", "log2RDR"]:
                if val == "log2RDR" and data_type not in seg_lambda:
                    continue
                for agg in [1, args["heatmap_agg"]]:
                    logging.info(f"  heatmap {val} agg={agg} {my_label}")
                    plot_cnv_heatmap(
                        sample,
                        data_type,
                        cnv_blocks,
                        seg_sx,
                        anns,
                        args["region_bed"],
                        proportions=model_params.get(f"{data_type}-theta", None),
                        val=val,
                        base_props=seg_lambda.get(data_type),
                        agg_size=agg,
                        lab_type=my_label,
                        filename=os.path.join(
                            plot_dir,
                            f"{out_prefix}.{platform}"
                            f".{val}_heatmap.{data_type}"
                            f".agg{agg}.{my_label}.{args['img_type']}",
                        ),
                        dpi=args["dpi"],
                        figsize=(20, 6 if agg > 1 else 15),
                        transparent=args["transparent"],
                    )

            logging.info(f"  1d scatter {my_label}")
            plot_rdr_baf_1d_pseudobulk(
                agg_bbc_data_sources[data_type],
                anns,
                agg_bbc_lambda.get(data_type),
                sample,
                data_type,
                args["genome_size"],
                haplo_blocks=cnv_blocks,
                region_bed=args["region_bed"],
                lab_type=my_label,
                is_inferred=(my_label == label),
                filename=os.path.join(
                    plot_dir,
                    f"{out_prefix}.{platform}.1d_scatter.{data_type}.{my_label}.pdf",
                ),
                subtitle=scatter_subtitle,
                platform=platform,
            )

    if args["umap"]:
        pass  # not yet implemented

    # ---- Visium spatial plots (per-rep for spatial images) ----
    rep_ids = anns["REP_ID"].unique()
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
        if metric.get("AUC_soft") is not None:
            visium_title += f"AUC={metric['AUC_soft']:.3f}"
        plot_visium_panel(
            sample,
            visium_slices,
            plot_dir,
            spot_label=label,
            path_label=ref_label,
            dpi=args["dpi"],
            title_info=visium_title,
        )
        if hasattr(instance, "labeling_trace") and instance.labeling_trace:
            plot_visium_iters(
                sample,
                visium_slices,
                instance.labeling_trace,
                barcodes=barcodes,
                out_dir=val_dir,
                clones=seg_data_sources[data_types[0]].clones,
                ref_label=(ref_label if ref_label in barcodes.columns else None),
                dpi=args["dpi"],
            )

    logging.root.removeHandler(_file_handler)
    _file_handler.close()


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
