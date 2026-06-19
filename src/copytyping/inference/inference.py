import copy
import logging
import os

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from copytyping.copytyping_parser import check_arguments_inference
from copytyping.inference.cell_model import Cell_Model
from copytyping.inference.clustering import kmeans_copytyping
from copytyping.inference.model_utils import make_baseline_fn
from copytyping.inference.spot_model import Spot_Model
from copytyping.io_utils import (
    read_cell_types,
    load_bulk_cnprofile,
    read_bbc_phases,
    load_bbc_modality,
    apply_bbc_phasing,
    aggregate_bbc_to_seg,
    load_spatial_neighbors,
    union_align_barcodes,
)
from copytyping.plot.plot_common import plot_count_histograms
from copytyping.plot.plot_modality import plot_modality_panel
from copytyping.plot.plot_visium import plot_visium_all
from copytyping.sx_data.sx_data import SX_Data
from copytyping.utils import (
    SPATIAL_PLATFORMS,
    add_file_logging,
    normalize_args,
    save_phased_bbc,
)


def run(args=None):
    logging.info("run copytyping inference")
    args = normalize_args(args)
    args = check_arguments_inference(args)
    sample = args["sample"]
    platform = args["platform"]
    data_types = args["data_types"]
    ref_label = args["ref_label"]
    out_dir = args["out_dir"]
    out_prefix = args["out_prefix"] or str(sample)
    os.makedirs(out_dir, exist_ok=True)
    _file_handler = add_file_logging(out_dir)
    proc_dir = os.path.join(out_dir, "processed_data")
    os.makedirs(proc_dir, exist_ok=True)

    logging.info(f"sample={sample}, platform={platform}, data_types={data_types}")

    # bulk data
    bbc_phases, phase_bbc, switchprobs_bbc = read_bbc_phases(args["bbc_phases"])
    seg_df, clones, clone_props = load_bulk_cnprofile(args["seg_ucn"], solfile=args["solfile"])

    cell_type_df = read_cell_types(args["cell_type"], req_cols={"BARCODE", ref_label})

    data_sources = {}
    raw_data_sources = {}
    seg_data_sources = {}
    bbc_data_by_dt = {}
    spatial_graphs = {}
    exclude_set = set(args["exclude"].split(",")) if args["exclude"] else None
    for data_type in data_types:
        barcodes_df, bbc_df, X_bbc, a_bbc, b_bbc = load_bbc_modality(
            args, data_type, cell_type_df, ref_label, exclude_set
        )
        bbc_df, B_bbc, N_bbc = apply_bbc_phasing(
            bbc_df, bbc_phases, a_bbc, b_bbc, data_type
        )
        seg_df_dt, X_seg, B_seg, N_seg = aggregate_bbc_to_seg(
            bbc_df, seg_df, X_bbc, B_bbc, N_bbc, data_type
        )

        # Build the FULL seg_sx (saved to disk so validate can plot all CN states)
        seg_sx_full = SX_Data(
            barcodes_df, seg_df_dt, X_seg, B_seg, N_seg, baf_clip=args["baf_clip"]
        )
        seg_data_sources[data_type] = seg_sx_full

        bbc_data_by_dt[data_type] = (bbc_df, X_bbc, B_bbc, N_bbc)
        if args["save_processed_data"]:
            save_phased_bbc(
                bbc_df,
                X_bbc,
                B_bbc,
                N_bbc,
                os.path.join(proc_dir, f"{out_prefix}.{data_type}.bbc"),
            )

        keep_cn_row = args["keep_cn_row"]
        if keep_cn_row:
            keep_set = {r.strip() for r in keep_cn_row.split(",") if r.strip()}
            keep_mask = seg_df_dt["CNP"].isin(keep_set).to_numpy()
            n_kept = int(keep_mask.sum())
            logging.info(
                f"[{data_type}] keep_cn_row: model uses {n_kept}/{len(seg_df_dt)} "
                f"segments; full data saved to disk (whitelist: {sorted(keep_set)})"
            )
            assert n_kept > 0, "keep_cn_row matched 0 segments; check format vs CNP"
            seg_sx = SX_Data(
                barcodes_df,
                seg_df_dt[keep_mask].reset_index(drop=True),
                X_seg[keep_mask],
                B_seg[keep_mask],
                N_seg[keep_mask],
                baf_clip=args["baf_clip"],
            )
        else:
            seg_sx = seg_sx_full

        h5ad_path = args[f"{data_type}_h5ad"]
        if h5ad_path is not None and platform in SPATIAL_PLATFORMS:
            adata = sc.read_h5ad(h5ad_path)
            if "spatial" in adata.obsm:
                spatial_graphs[data_type] = load_spatial_neighbors(
                    h5ad_path, n_neighs=args["n_neighs"]
                )

        raw_data_sources[data_type] = seg_sx.to_cluster_level()
        max_k = args["max_smooth_k"]
        if data_type in spatial_graphs and max_k > 0:
            seg_sx_smoothed = copy.deepcopy(seg_sx)
            seg_sx_smoothed.apply_adaptive_smoothing(
                spatial_graphs[data_type],
                max_k=max_k,
                min_umi=args["min_umi_per_spot"],
                min_snp_umi=args["min_snp_umi_per_spot"],
            )
            data_sources[data_type] = seg_sx_smoothed.to_cluster_level()
        else:
            data_sources[data_type] = raw_data_sources[data_type]

    cnv_blocks = seg_data_sources[data_types[0]].cnv_blocks
    cnv_blocks.to_csv(
        os.path.join(proc_dir, f"{out_prefix}.cnp_profile.tsv"),
        sep="\t",
        index=False,
    )

    barcodes, modality_masks = union_align_barcodes(data_sources, data_types)

    instance = {"single_cell": Cell_Model, "spatial": Spot_Model}[platform](
        barcodes,
        platform,
        data_types,
        data_sources,
        proc_dir,
        out_prefix,
        args["verbosity"],
        modality_masks=modality_masks,
        args=args,
    )
    model_params, final_ll = instance.fit(fit_mode=args["fit_mode"])

    label = f"{args['method']}_label"
    if args["method"] == "kmeans":
        anns, clone_props = kmeans_copytyping(
            data_sources, barcodes, ref_label, data_sources[data_types[0]].K, label
        )
    else:
        anns, clone_props = instance.predict(
            args["fit_mode"],
            model_params,
            label=label,
        )
    logging.info(
        "clone fractions: " + ", ".join(f"{k}={v:.3f}" for k, v in clone_props.items())
    )

    anns.to_csv(
        os.path.join(out_dir, f"{out_prefix}.annotations.tsv"),
        sep="\t",
        index=False,
    )

    param_dict = {"log_likelihood": np.array([final_ll])}
    for data_type in data_types:
        for key in ["lambda", "theta", "tau", "inv_phi"]:
            pk = f"{data_type}-{key}"
            if pk in model_params:
                param_dict[pk.replace("-", "_")] = np.atleast_1d(model_params[pk])
    np.savez(os.path.join(proc_dir, f"{out_prefix}.model_params.npz"), **param_dict)

    pd.Series(
        {
            "sample": sample,
            "platform": platform,
            "data_types": ",".join(data_types),
            "label": label,
            "ref_label": ref_label,
        }
    ).to_csv(
        os.path.join(proc_dir, f"{out_prefix}.metadata.tsv"),
        sep="\t",
        header=False,
    )

    if args["save_processed_data"]:
        for data_type in data_types:
            sx = seg_data_sources[data_type]
            prefix = os.path.join(proc_dir, f"{out_prefix}.{data_type}")
            sparse.save_npz(f"{prefix}.seg.X.npz", sparse.csr_matrix(sx.X))
            sparse.save_npz(f"{prefix}.seg.Y.npz", sparse.csr_matrix(sx.Y))
            sparse.save_npz(f"{prefix}.seg.D.npz", sparse.csr_matrix(sx.D))
            logging.info(
                f"saved {data_type} count matrices: X/Y/D shape=({sx.G}, {sx.N})"
            )

        if instance.labeling_trace:
            trace_dict = {}
            for i, lt in enumerate(instance.labeling_trace):
                trace_dict[f"labels_{i}"] = lt["labels"]
                trace_dict[f"max_posterior_{i}"] = lt["max_posterior"]
                if "tumor_purity" in lt:
                    trace_dict[f"tumor_purity_{i}"] = lt["tumor_purity"]
            trace_dict["n_iters"] = np.array([len(instance.labeling_trace)])
            np.savez(
                os.path.join(proc_dir, f"{out_prefix}.labeling_trace.npz"),
                **trace_dict,
            )

    region_bed = args["region_bed"]
    genome_size = args["genome_size"]
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    dpi = args["dpi"]
    heatmap_agg = args["heatmap_agg"]
    # RDR baseline from model reference cells; fall back to normal labels (kmeans)
    is_reference = instance.is_reference
    ref_clone = instance.ref_clone
    no_normal = args["no_normal"]
    if is_reference is None:
        is_reference = (anns[label] == "normal").to_numpy()
        ref_clone = 0
        no_normal = False
    baseline_fn = make_baseline_fn(is_reference, ref_clone, no_normal)
    plot_labels = [label]
    if ref_label in anns.columns:
        plot_labels.append(ref_label)

    if args["method"] == "copytyping":
        plot_count_histograms(
            seg_data_sources,
            sample,
            os.path.join(plot_dir, f"{out_prefix}.count_histograms.pdf"),
            dpi=dpi,
        )

    is_spatial = platform in SPATIAL_PLATFORMS
    platform_str = "spatial" if is_spatial else "single_cell"
    for data_type in data_types:
        plot_modality_panel(
            sample=sample,
            data_type=data_type,
            prefix=out_prefix,
            plot_dir=plot_dir,
            seg_sx=seg_data_sources[data_type],
            raw_clust=raw_data_sources[data_type],
            bbc_data=bbc_data_by_dt[data_type],
            cnv_blocks=cnv_blocks,
            anns=anns,
            baseline_fn=baseline_fn,
            cluster_base_props=model_params.get(f"{data_type}-lambda"),
            primary_label=label,
            plot_labels=plot_labels,
            theta=model_params.get(f"{data_type}-theta"),
            region_bed=region_bed,
            genome_size=genome_size,
            dpi=dpi,
            heatmap_agg=heatmap_agg,
            min_snp_count=args["min_snp_count"],
            max_bin_length=args["max_bin_length"],
            platform_str=platform_str,
            ascn_profile=args["ascn_profile"],
        )

    if is_spatial and args["gex_h5ad"]:
        plot_visium_all(
            sample=sample,
            anns=anns,
            h5ad_source=args["gex_h5ad"],
            raw_clust=raw_data_sources["gex"],
            plot_dir=plot_dir,
            spot_label=label,
            ref_label=ref_label,
            labeling_trace=getattr(instance, "labeling_trace", None) or None,
            barcodes=anns,
            clones=list(seg_data_sources["gex"].clones),
            dpi=dpi,
        )

    logging.info(f"inference complete. outputs in {out_dir}")
    logging.root.removeHandler(_file_handler)
    _file_handler.close()
