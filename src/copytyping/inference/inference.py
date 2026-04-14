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
    evaluate_malignant_accuracy,
    joincount_zscore,
    refine_labels_by_reference,
)
from copytyping.io_utils import (
    build_bbc_sx,
    load_modality_data,
    subset_model_params,
    subset_sx_data,
    union_align_barcodes,
)
from copytyping.plot.plot_heatmap import plot_cnv_heatmap
from copytyping.plot.plot_common import (
    plot_crosstab,
    plot_rdr_baf_1d_pseudobulk,
)
from copytyping.plot.plot_visium import plot_visium_debug, plot_visium_panel
from copytyping.sx_data.sx_data import SX_Data
from copytyping.utils import (
    NA_CELLTYPE,
    SPATIAL_PLATFORMS,
    add_file_logging,
    is_tumor_label,
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
    _file_handler = add_file_logging(out_dir)
    dirs = {
        "work": os.path.join(out_dir, "work"),
        "plots": os.path.join(out_dir, "plots"),
        "heatmap_agg1": os.path.join(out_dir, "plots", "heatmaps", "agg1"),
        "heatmap_aggx": os.path.join(
            out_dir, "plots", "heatmaps", f"agg{args['heatmap_agg']}"
        ),
        "scatter": os.path.join(out_dir, "plots", "scatters"),
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
    bbc_data_sources = {}
    adatas = {}
    for data_type in data_types:
        barcodes_df, seg_df, X_seg, Y_seg, D_seg, bbc_data = load_modality_data(
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
        bbc_data_sources[data_type] = bbc_data

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
        fix_params[f"{data_type}-theta"] = not args["update_purity"]

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
        dirs["work"],
        out_prefix,
        verbosity,
        modality_masks=clust_masks,
        hard_em=args.get("hard_em", False),
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
    )
    logging.info(
        "clone fractions: " + ", ".join(f"{k}={v:.3f}" for k, v in clone_props.items())
    )

    is_normal = getattr(instance, "_init_is_normal", None)
    if aggr_mode == "clust":
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

    metric = {}
    is_spot = platform in SPATIAL_PLATFORMS
    if ref_label in barcodes.columns:
        anns = refine_labels_by_reference(anns, ref_label, label, f"{label}-refined")
        metric = evaluate_malignant_accuracy(
            anns,
            cell_label=label,
            cell_type=ref_label,
            tumor_post="tumor_purity" if is_spot else "tumor",
            skip_binary=is_spot,
        )

    if is_spot:
        gex_adata = adatas.get("gex")
        if gex_adata is not None and "spatial" in gex_adata.obsm:
            common = anns["BARCODE"].values
            adata_sub = gex_adata[gex_adata.obs_names.isin(common)]
            anns_indexed = anns.set_index("BARCODE").loc[adata_sub.obs_names]
            coords = adata_sub.obsm["spatial"]
            clone_labels = anns_indexed[label].to_numpy()

            jc_all = joincount_zscore(clone_labels, coords)
            logging.info("joincount z-score (all spots):")
            for lab, z in sorted(jc_all.items()):
                logging.info(f"  {lab:8s}: {z:.4f}")
                metric[f"JC_{lab}"] = z

            if ref_label in anns_indexed.columns:
                gt_tumor = anns_indexed[ref_label].apply(is_tumor_label).to_numpy()
                if gt_tumor.any():
                    jc_tumor = joincount_zscore(
                        clone_labels[gt_tumor], coords[gt_tumor]
                    )
                    logging.info("joincount z-score (GT-tumor only):")
                    for lab, z in sorted(jc_tumor.items()):
                        logging.info(f"  {lab:8s}: {z:.4f}")
                        metric[f"JC_tumor_{lab}"] = z

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

        post_label = label
        if "max_posterior" in anns_rep.columns:
            logging.info(f"posterior statistics{rep_tag}:")
            for grp, sub in anns_rep.groupby(post_label, sort=True):
                mp = sub["max_posterior"].to_numpy()
                md = sub["margin_delta"].to_numpy()
                logging.info(
                    f"  {grp:8s} (n={len(sub):4d}): "
                    f"max_post min={mp.min():.3f} mean={mp.mean():.3f} "
                    f"median={np.median(mp):.3f} max={mp.max():.3f}  "
                    f"margin min={md.min():.3f} mean={md.mean():.3f}"
                )

        if ref_label in anns_rep.columns:
            plot_crosstab(
                anns_rep,
                sample,
                os.path.join(
                    dirs["plots"],
                    f"{out_prefix}.{platform}{rep_tag}.crosstab.png",
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
                        (1, dirs["heatmap_agg1"]),
                        (agg_size, dirs["heatmap_aggx"]),
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

            # Segment-level 1D scatter
            plot_rdr_baf_1d_pseudobulk(
                sx_rep,
                anns_rep,
                params_rep.get(f"{data_type}-lambda", None),
                sample,
                data_type,
                genome_size,
                haplo_blocks=cnv_blocks,
                wl_segments=wl_segments,
                mask_cnp=False,
                lab_type=label,
                filename=os.path.join(
                    dirs["scatter"],
                    f"{out_prefix}.{platform}{rep_tag}"
                    f".1d_scatter.{data_type}.{label}.pdf",
                ),
            )

            if data_type in bbc_data_sources:
                bbc_sx = build_bbc_sx(
                    bbc_data_sources[data_type],
                    seg_data_sources[data_type],
                    rep_mask=rep_mask,
                )
                bbc_lambda = None
                if is_normal is not None:
                    bbc_lambda = compute_baseline_proportions(
                        bbc_sx.X, bbc_sx.T, is_normal[rep_mask]
                    )
                plot_rdr_baf_1d_pseudobulk(
                    bbc_sx,
                    anns_rep,
                    bbc_lambda,
                    sample,
                    data_type,
                    genome_size,
                    haplo_blocks=cnv_blocks,
                    wl_segments=wl_segments,
                    resolution="bbc",
                    mask_cnp=False,
                    lab_type=label,
                    markersize=2,
                    filename=os.path.join(
                        dirs["scatter"],
                        f"{out_prefix}.{platform}{rep_tag}"
                        f".1d_scatter_bbc.{data_type}.{label}.pdf",
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
            plot_visium_debug(
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

    # Remove per-run file handler to avoid log leaking across pipeline runs
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
