import logging
import os

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from copytyping.copytyping_parser import check_arguments_inference
from copytyping.inference.base_model import Base_Model
from copytyping.inference.cell_model import Cell_Model
from copytyping.inference.spot_model import Spot_Model
from copytyping.inference.count_data import (
    CountData,
    count_data_cnprofile,
    initialize_count_data,
    restrict_masks_to_cnp,
    save_count_data,
    segment_count_data,
    smooth_spatial_neighbors,
)
from copytyping.inference.model_utils import (
    compute_rdr_baseline,
    model_kwargs_from_args,
    save_model_params,
)
from copytyping.io_utils import (
    read_cell_types,
    load_bulk_cnprofile,
    read_bbc_phases,
    build_spatial_graphs,
)
from copytyping.plot.plot_heatmap import plot_cnv_heatmap
from copytyping.plot.plot_scatter_1d import plot_rdr_baf_1d_pseudobulk
from copytyping.plot.plot_scatter_2d import plot_scatter_2d_per_cell
from copytyping.plot.plot_visium import plot_visium_all
from copytyping.utils import (
    SPATIAL_PLATFORMS,
    add_file_logging,
    log_arguments,
    normalize_args,
)


##################################################
# orchestrator
##################################################


def run(args: dict | None = None) -> None:
    """Validate args, load full inputs, then dispatch to the platform pipeline."""
    logging.info("run copytyping inference")
    args = normalize_args(args)
    args = check_arguments_inference(args)
    platform = args["platform"]
    assay_types = args["assay_types"]
    out_dir = args["out_dir"]
    sample_id = args["sample"]
    out_prefix = args["out_prefix"] or sample_id
    proc_dir = os.path.join(out_dir, "processed_data")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    _file_handler = add_file_logging(out_dir)

    log_arguments(args)

    # ---- load inputs -------------------------------
    bbc_phases = read_bbc_phases(args["bbc_phases"])
    seg_df, clones, clone_props, cn_A, cn_B, cn_C, cn_BAF = load_bulk_cnprofile(
        args["seg_ucn"], solfile=args["solfile"], baf_clip=args["baf_clip"]
    )
    cell_type_df = read_cell_types(
        args["cell_type"], req_cols={"BARCODE", args["ref_label"]}
    )

    bbc_count_datas = {}
    for assay_type in assay_types:
        bbc_count_datas[assay_type] = initialize_count_data(
            args[f"{assay_type}_barcodes"],
            args[f"{assay_type}_X_count"],
            args[f"{assay_type}_A_allele"],
            args[f"{assay_type}_B_allele"],
            args[f"{assay_type}_cnv_segments"],
            cell_type_df,
            args["ref_label"],
            args["exclude_cell_types"],
        )
        bbc_count_datas[assay_type].annotate_cnps(
            bbc_phases, seg_df, clones, cn_A, cn_B, cn_C, cn_BAF
        )

    spatial_graphs = None
    if platform in SPATIAL_PLATFORMS:
        spatial_graphs = build_spatial_graphs(
            assay_types=assay_types,
            h5ad_paths={assay: args[f"{assay}_h5ad"] for assay in assay_types},
            n_neighs=args["n_neighs"],
        )

    # ---- Copy-typing inference -------------------------------
    label = f"{args['method']}_label"
    model, anns, model_params = run_copytyping(
        assay_types, bbc_count_datas, platform, label, spatial_graphs, args
    )

    # ---- write annotations + model params -------------------------------
    anns.to_csv(
        os.path.join(out_dir, f"{out_prefix}.annotations.tsv"), sep="\t", index=False
    )

    bin_count_datas = segment_count_data(
        bbc_count_datas, "cnp_bin", args["min_snp_count"], args["max_bin_length"]
    )
    seg_count_datas = segment_count_data(bbc_count_datas, "cnp_segment")
    if args["save_processed_data"]:
        save_count_data(bbc_count_datas, os.path.join(proc_dir, f"{out_prefix}.bbc"))
        save_count_data(seg_count_datas, os.path.join(proc_dir, f"{out_prefix}.seg"))

    cluster_count_datas = segment_count_data(bbc_count_datas, "cnp_cluster")

    # per-assay CNP tables
    cluster_cnprofile = {
        a: count_data_cnprofile(count_data)
        for a, count_data in cluster_count_datas.items()
    }
    seg_cnprofile = {
        a: count_data_cnprofile(count_data) for a, count_data in seg_count_datas.items()
    }
    bin_cnprofile = {
        a: count_data_cnprofile(count_data) for a, count_data in bin_count_datas.items()
    }
    seg_cnprofile[assay_types[0]].to_csv(
        os.path.join(proc_dir, f"{out_prefix}.cnp_profile.tsv"), sep="\t", index=False
    )

    # ---- plots ----------------------------------------------------------
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    # RDR baseline from model reference cells; fall back to normal labels
    is_reference = model.is_reference
    ref_clone = model.ref_clone
    no_normal = args["no_normal"]
    if is_reference is None:
        is_reference = (anns[label] == "normal").to_numpy()
        ref_clone = 0
        no_normal = False
    plot_labels = [label]
    if args["ref_label"] in anns.columns:
        plot_labels.append(args["ref_label"])

    for assay_type in assay_types:
        cluster_count_data = cluster_count_datas[assay_type]
        seg_count_data = seg_count_datas[assay_type]
        bin_count_data = bin_count_datas[assay_type]
        cluster_profile = cluster_cnprofile[assay_type]
        seg_profile = seg_cnprofile[assay_type]
        bin_profile = bin_cnprofile[assay_type]
        seg_baseline = compute_rdr_baseline(
            seg_count_data, is_reference, ref_clone, no_normal
        )
        bin_baseline = compute_rdr_baseline(
            bin_count_data, is_reference, ref_clone, no_normal
        )
        # cluster 2D uses the model's fitted baseline (cluster-level lambda)
        cluster_baseline = model_params.get(f"{assay_type}-lambda")
        if cluster_baseline is None:
            cluster_baseline = compute_rdr_baseline(
                cluster_count_data, is_reference, ref_clone, no_normal
            )
        theta = model_params.get(f"{assay_type}-theta")
        rep_ids = sorted(seg_count_data.barcodes["REP_ID"].unique())

        # 2D BAF-vs-log2RDR scatter per CNP cluster (all cells)
        plot_scatter_2d_per_cell(
            cluster_count_data.count_X,
            cluster_count_data.count_B,
            cluster_count_data.count_C,
            cluster_count_data.cn_A,
            cluster_count_data.cn_B,
            cluster_count_data.cn_C,
            cluster_count_data.clones,
            cluster_profile,
            anns,
            sample_id,
            os.path.join(plot_dir, f"{out_prefix}.{assay_type}.cluster_2d.pdf"),
            label,
            base_props=cluster_baseline,
            dpi=args["dpi"],
        )

        # CNV heatmaps: one PDF per agg level; pages = rep_id x [BAF, log2RDR]
        for agg in [1, args["heatmap_agg"]]:
            fname = os.path.join(
                plot_dir, f"{out_prefix}.{assay_type}.heatmap.agg{agg}.pdf"
            )
            with PdfPages(fname) as pdf:
                for rep_id in rep_ids:
                    seg_rep_data, rep_mask = seg_count_data.subset_by_rep(rep_id)
                    anns_rep = anns.iloc[rep_mask].reset_index(drop=True)
                    theta_rep = theta[rep_mask] if theta is not None else None
                    for val in ["BAF", "log2RDR"]:
                        if val == "log2RDR" and seg_baseline is None:
                            continue
                        plot_cnv_heatmap(
                            sample_id,
                            assay_type,
                            seg_profile,
                            seg_rep_data.count_X,
                            seg_rep_data.count_B,
                            seg_rep_data.count_C,
                            seg_profile,
                            len(seg_count_data.clones),
                            anns_rep,
                            args["region_bed"],
                            proportions=theta_rep,
                            val=val,
                            base_props=seg_baseline,
                            agg_size=agg,
                            label_cols=plot_labels,
                            primary_label=label,
                            pdf_pages=pdf,
                            dpi=args["dpi"],
                            figsize=(20, 6 if agg > 1 else 15),
                            rep_id=rep_id,
                            ascn_profile=args["ascn_profile"],
                        )

        # 1D pseudobulk RDR + BAF along the genome, per rep, per label
        for my_label in plot_labels:
            fname = os.path.join(
                plot_dir, f"{out_prefix}.{assay_type}.1d_scatter.{my_label}.pdf"
            )
            with PdfPages(fname) as pdf:
                for rep_id in rep_ids:
                    bin_rep_data, rep_mask = bin_count_data.subset_by_rep(rep_id)
                    anns_rep = anns.iloc[rep_mask].reset_index(drop=True)
                    plot_rdr_baf_1d_pseudobulk(
                        bin_rep_data.count_X,
                        bin_rep_data.count_B,
                        bin_rep_data.count_C,
                        bin_rep_data.cn_A,
                        bin_rep_data.cn_B,
                        bin_rep_data.cn_C,
                        bin_rep_data.cn_BAF,
                        bin_rep_data.clones,
                        bin_profile,
                        anns_rep,
                        bin_baseline,
                        sample_id,
                        assay_type,
                        args["genome_size"],
                        args["region_bed"],
                        haplo_blocks=seg_profile,
                        lab_type=my_label,
                        is_inferred=(my_label == label),
                        pdf_pages=pdf,
                        platform=platform,
                        subtitle=f"rep={rep_id}",
                        ascn_profile=args["ascn_profile"],
                    )

    if platform in SPATIAL_PLATFORMS and args["gex_h5ad"]:
        gex_cluster_data = cluster_count_datas["gex"]
        plot_visium_all(
            sample=sample_id,
            anns=anns,
            h5ad_source=args["gex_h5ad"],
            ballele_counts=gex_cluster_data.count_B,
            total_allele_counts=gex_cluster_data.count_C,
            cn_A=gex_cluster_data.cn_A,
            cn_B=gex_cluster_data.cn_B,
            cluster_barcodes=gex_cluster_data.barcodes,
            clones=gex_cluster_data.clones,
            plot_dir=plot_dir,
            spot_label=label,
            ref_label=args["ref_label"],
            labeling_trace=getattr(model, "labeling_trace", None) or None,
            barcodes=anns,
            dpi=args["dpi"],
        )

    logging.info(f"inference complete. outputs in {out_dir}")
    logging.root.removeHandler(_file_handler)
    _file_handler.close()


##################################################
# model fit
##################################################


def run_copytyping(
    assay_types: list[str],
    bbc_data: dict[str, CountData],
    platform: str,
    label: str,
    spatial_graphs: dict | None,
    args: dict,
) -> tuple[Base_Model, pd.DataFrame, dict, float]:
    """Cluster the annotated BBC CountData and fit the platform model.

    Operates purely on the CountData ``bbc_data`` (+ spatial neighbor graphs).
    Returns ``(model, anns, model_params,
    model_ll)``. Single-cell / multiome run Cell_Model on the jointly-clustered
    CountData. Spatial first smooths the clustered CountData over the spot
    neighbor graphs (``smooth_spatial_neighbors``), then runs Spot_Model.
    """
    out_prefix = args["out_prefix"] or str(args["sample"])
    proc_dir = os.path.join(args["out_dir"], "processed_data")

    # joint cluster-level aggregation across modalities (CountData EM input)
    cluster_count_data = segment_count_data(bbc_data, agg_level="cnp_cluster")
    restrict_masks_to_cnp(cluster_count_data, args["keep_cn_row"])

    if platform in SPATIAL_PLATFORMS:
        smoothed_count_data = smooth_spatial_neighbors(
            cluster_count_data,
            spatial_graphs or {},
            max_k=args["max_smooth_k"],
            min_umi=args["min_umi_per_spot"],
            min_snp_umi=args["min_snp_umi_per_spot"],
        )
        model = Spot_Model(
            count_data=smoothed_count_data,
            platform=platform,
            assay_types=assay_types,
            work_dir=proc_dir,
            prefix=out_prefix,
            **model_kwargs_from_args(args),
        )
    else:
        model = Cell_Model(
            count_data=cluster_count_data,
            platform=platform,
            assay_types=assay_types,
            work_dir=proc_dir,
            prefix=out_prefix,
            **model_kwargs_from_args(args),
        )
    model_params, model_ll = model.fit(fit_mode=args["fit_mode"])
    anns, clone_props = model.predict(args["fit_mode"], label=label)
    logging.info(
        "clone fractions: " + ", ".join(f"{k}={v:.3f}" for k, v in clone_props.items())
    )

    if args["save_processed_data"]:
        if model.labeling_trace:
            trace_dict = {}
            for i, lt in enumerate(model.labeling_trace):
                trace_dict[f"labels_{i}"] = lt["labels"]
                trace_dict[f"max_posterior_{i}"] = lt["max_posterior"]
                if "tumor_purity" in lt:
                    trace_dict[f"tumor_purity_{i}"] = lt["tumor_purity"]
            trace_dict["n_iters"] = np.array([len(model.labeling_trace)])
            np.savez(
                os.path.join(proc_dir, f"{out_prefix}.labeling_trace.npz"), **trace_dict
            )
        save_model_params(
            model_params,
            model_ll,
            assay_types,
            os.path.join(proc_dir, f"{out_prefix}.model_params.npz"),
        )

    return model, anns, model_params


##################################################
# kmeans (disabled)
##################################################


def run_kmeans() -> None:
    # ---- kmeans baseline (disabled for now) -----------------------------
    # if args["method"] == "kmeans":
    #     barcodes, _ = union_align_barcodes(unsmoothed_data_sources, assay_types)
    #     anns, clone_props = kmeans_copytyping(
    #         unsmoothed_data_sources, barcodes, args["ref_label"],
    #         model.num_clones, label,
    #     )
    #     logging.info(
    #         "clone fractions: "
    #         + ", ".join(f"{k}={v:.3f}" for k, v in clone_props.items())
    #     )

    return
