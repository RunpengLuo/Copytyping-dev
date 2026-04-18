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
from copytyping.inference.inference_utils import (
    adaptive_bin_bbc,
    annotate_adata_celltype,
    merge_celltype_into_barcodes,
)
from copytyping.inference.model_utils import (
    compute_baseline_proportions,
    prepare_params,
)
from copytyping.inference.spot_model import Spot_Model
from copytyping.inference.validation import (
    evaluate_malignant_accuracy,
    joincount_zscore,
    refine_labels_by_reference,
)
from copytyping.io_utils import (
    load_modality_data,
    union_align_barcodes,
)
from copytyping.plot.plot_heatmap import plot_cnv_heatmap
from copytyping.plot.plot_common import (
    plot_crosstab,
    plot_rdr_baf_1d_pseudobulk,
)
from copytyping.plot.plot_visium import plot_visium_iters, plot_visium_panel
from copytyping.sx_data.sx_data import SX_Data
from copytyping.utils import (
    SPATIAL_PLATFORMS,
    add_file_logging,
    read_whitelist_segments,
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
    dirs = {
        "work": os.path.join(out_dir, "work"),
        "plots": os.path.join(out_dir, "plots"),
        "heatmap_agg1": os.path.join(out_dir, "plots", "heatmaps", "agg1"),
        "heatmap_aggx": os.path.join(
            out_dir, "plots", "heatmaps", f"agg{args['heatmap_agg']}"
        ),
        "scatter": os.path.join(out_dir, "plots", "scatters"),
        "validation": os.path.join(out_dir, "plots", "validation"),
    }
    if platform in SPATIAL_PLATFORMS:
        dirs["visium"] = os.path.join(out_dir, "plots", "spatial_images")
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    verbosity = args["verbosity"]

    logging.info(f"sample={sample}, platform={platform}, data_types={data_types}")

    cell_type_df = None
    if args.get("cell_type") is not None:
        cell_type_df = pd.read_table(args["cell_type"])
        assert "BARCODE" in cell_type_df.columns
        assert ref_label in cell_type_df.columns

    aggr_mode = args.get("aggr_mode", "clust")
    data_sources = {}  # used by EM (seg or clust level)
    seg_data_sources = {}
    agg_bbc_data_sources = {}
    adatas = {}
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
            )
        )

        if cell_type_df is not None:
            barcodes_df = merge_celltype_into_barcodes(
                barcodes_df, cell_type_df, ref_label, data_type
            )

        seg_sx = SX_Data(barcodes_df, seg_df, X_seg, Y_seg, D_seg)
        seg_data_sources[data_type] = seg_sx
        agg_bbc_data_sources[data_type] = adaptive_bin_bbc(
            bbc_df,
            X_bbc,
            Y_bbc,
            D_bbc,
            seg_sx,
            args["min_snp_count"],
            args["max_bin_length"],
        )

        data_sources[data_type] = seg_sx.to_cluster_level() if aggr_mode == "clust" else seg_sx

        if args.get(f"{data_type}_h5ad") is not None:
            adatas[data_type] = sc.read_h5ad(args[f"{data_type}_h5ad"])
            if cell_type_df is not None:
                annotate_adata_celltype(
                    adatas[data_type], cell_type_df, ref_label, data_type
                )

    save_cnp_profile(
        seg_data_sources[data_types[0]],
        os.path.join(out_dir, f"{out_prefix}.cnp_profile.tsv"),
    )

    barcodes, modality_masks = union_align_barcodes(data_sources, data_types)

    cnv_blocks = seg_data_sources[data_types[0]].cnv_blocks
    init_params, fix_params = prepare_params(
        args, cnv_blocks, platform, data_types, SPATIAL_PLATFORMS
    )

    instance = {"single_cell": Cell_Model, "spatial": Spot_Model}[platform](
        barcodes,
        platform,
        data_types,
        data_sources,
        dirs["work"],
        out_prefix,
        verbosity,
        modality_masks=modality_masks,
        hard_em=args["hard_em"],
    )
    model_params = instance.fit(
        fit_mode=args["fit_mode"],
        fix_params=fix_params,
        init_params=init_params,
        max_iter=args["niters"],
    )

    label = f"{args['method']}-label"
    anns, clone_props = instance.predict(
        args["fit_mode"],
        model_params,
        label=label,
        posterior_thres=args["posterior_thres"],
        margin_thres=args["margin_thres"],
        purity_threshold=args["purity_threshold"],
    )
    logging.info(
        "clone fractions: " + ", ".join(f"{k}={v:.3f}" for k, v in clone_props.items())
    )

    is_spot = platform in SPATIAL_PLATFORMS
    hard_label = f"{label}-purity_cutoff" if is_spot else label
    is_normal = (anns[hard_label] == "normal").to_numpy()
    if is_normal.sum() == 0:
        is_normal = None
    # Compute segment-level baseline proportions from predicted normal labels
    seg_lambda = {}
    for data_type in data_types:
        seg_sx = seg_data_sources[data_type]
        if is_normal is not None:
            seg_lambda[data_type] = compute_baseline_proportions(
                seg_sx.X, seg_sx.T, is_normal
            )
        elif f"{data_type}-lambda" in model_params:
            seg_lambda[data_type] = model_params[f"{data_type}-lambda"]

    # Compute agg-bbc baseline proportions
    agg_bbc_lambda = {}
    for data_type in data_types:
        if data_type in agg_bbc_data_sources and is_normal is not None:
            agg_sx = agg_bbc_data_sources[data_type]
            agg_bbc_lambda[data_type] = compute_baseline_proportions(
                agg_sx.X, agg_sx.T, is_normal
            )

    metric = {}
    if ref_label in barcodes.columns:
        metric = evaluate_malignant_accuracy(
            anns,
            qry_label=hard_label,
            ref_label=ref_label,
            tumor_post="tumor_purity" if is_spot else "tumor",
        )
        anns = refine_labels_by_reference(anns, ref_label, label, f"{label}-refined")

    if is_spot:
        gex_adata = adatas.get("gex")
        if gex_adata is not None and "spatial" in gex_adata.obsm:
            common = anns["BARCODE"].values
            adata_sub = gex_adata[gex_adata.obs_names.isin(common)]
            anns_indexed = anns.set_index("BARCODE").loc[adata_sub.obs_names]
            coords = adata_sub.obsm["spatial"]
            clone_labels = anns_indexed[hard_label].to_numpy()

            jc = joincount_zscore(clone_labels, coords)
            logging.info("joincount z-score:")
            for lab, z in sorted(jc.items()):
                logging.info(f"  {lab:8s}: {z:.4f}")
                metric[f"JC_{lab}"] = z

    if metric:
        eval_df = pd.DataFrame([{"SAMPLE": sample, "PLATFORM": platform, **metric}])
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
    wl_segments = read_whitelist_segments(args["region_bed"])
    genome_size = args["genome_size"]
    agg_size = args["heatmap_agg"]
    img_type = args["img_type"]
    dpi = args["dpi"]
    transparent = args["transparent"]

    # Posterior statistics
    if "max_posterior" in anns.columns:
        logging.info("posterior statistics:")
        for grp, sub in anns.groupby(hard_label, sort=True):
            mp = sub["max_posterior"].to_numpy()
            md = sub["margin_delta"].to_numpy()
            logging.info(
                f"  {grp:8s} (n={len(sub):4d}): "
                f"max_post min={mp.min():.3f} mean={mp.mean():.3f} "
                f"median={np.median(mp):.3f} max={mp.max():.3f}  "
                f"margin min={md.min():.3f} mean={md.mean():.3f}"
            )

    # Crosstab (all spots/cells)
    if ref_label in anns.columns:
        plot_crosstab(
            anns,
            sample,
            os.path.join(
                dirs["validation"],
                f"{out_prefix}.{platform}.crosstab.png",
            ),
            metric=metric,
            acol=hard_label,
            bcol=ref_label,
        )

    # Per-data_type plots (all reps combined)
    for data_type in data_types:
        seg_sx = seg_data_sources[data_type]
        for val in ["BAF", "log2RDR"]:
            if val == "log2RDR" and data_type not in seg_lambda:
                continue
            for my_label in [hard_label, ref_label]:
                if my_label not in anns:
                    continue
                agg_levels = [
                    (1, dirs["heatmap_agg1"]),
                    (agg_size, dirs["heatmap_aggx"]),
                ]
                for agg, agg_dir in agg_levels:
                    plot_cnv_heatmap(
                        sample,
                        data_type,
                        cnv_blocks,
                        seg_sx,
                        anns,
                        wl_segments,
                        proportions=model_params.get(f"{data_type}-theta", None),
                        val=val,
                        base_props=seg_lambda.get(data_type),
                        agg_size=agg,
                        lab_type=my_label,
                        filename=os.path.join(
                            agg_dir,
                            f"{out_prefix}.{platform}"
                            f".{val}_heatmap.{data_type}"
                            f".{my_label}.{img_type}",
                        ),
                        dpi=dpi,
                        figsize=(20, 6 if agg > 1 else 15),
                        transparent=transparent,
                    )

        # Segment-level 1D scatter
        plot_rdr_baf_1d_pseudobulk(
            seg_sx,
            anns,
            seg_lambda.get(data_type),
            sample,
            data_type,
            genome_size,
            haplo_blocks=cnv_blocks,
            wl_segments=wl_segments,
            mask_cnp=False,
            lab_type=hard_label,
            filename=os.path.join(
                dirs["scatter"],
                f"{out_prefix}.{platform}.1d_scatter.{data_type}.{hard_label}.pdf",
            ),
        )

        if data_type in agg_bbc_data_sources:
            plot_rdr_baf_1d_pseudobulk(
                agg_bbc_data_sources[data_type],
                anns,
                agg_bbc_lambda.get(data_type),
                sample,
                data_type,
                genome_size,
                haplo_blocks=cnv_blocks,
                wl_segments=wl_segments,
                resolution="agg-bbc",
                mask_cnp=False,
                lab_type=hard_label,
                filename=os.path.join(
                    dirs["scatter"],
                    f"{out_prefix}.{platform}"
                    f".1d_scatter_agg_bbc.{data_type}.{hard_label}.pdf",
                ),
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
            dirs["visium"],
            spot_label=label,
            path_label=ref_label,
            dpi=dpi,
            title_info=visium_title,
        )
        if hasattr(instance, "param_trace") and instance.param_trace:
            plot_visium_iters(
                sample,
                visium_slices,
                instance.param_trace,
                barcodes=barcodes,
                data_type=data_types[0],
                out_dir=dirs["visium"],
                clones=seg_data_sources[data_types[0]].clones,
                ref_label=(ref_label if ref_label in barcodes.columns else None),
                dpi=dpi,
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
