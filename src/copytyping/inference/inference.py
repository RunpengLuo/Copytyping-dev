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
    count_data_cnv_blocks,
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
from copytyping.plot.plot_common import plot_count_histograms
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


def run(args=None):
    """Validate args, load full inputs, then dispatch to the platform pipeline."""
    logging.info("run copytyping inference")
    args = normalize_args(args)
    args = check_arguments_inference(args)
    platform = args["platform"]
    assay_types = args["assay_types"]
    out_dir = args["out_dir"]
    out_prefix = args["out_prefix"] or str(args["sample"])
    proc_dir = os.path.join(out_dir, "processed_data")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    _file_handler = add_file_logging(out_dir)

    log_arguments(args)

    # ---- load inputs -------------------------------
    bbc_phases = read_bbc_phases(args["bbc_phases"])
    seg_df, clones, clone_props, cn_A, cn_B, cn_C, BAF = load_bulk_cnprofile(
        args["seg_ucn"], solfile=args["solfile"], baf_clip=args["baf_clip"]
    )
    cell_type_df = read_cell_types(
        args["cell_type"], req_cols={"BARCODE", args["ref_label"]}
    )

    bbc_count_datas = {}
    for assay_type in assay_types:
        cd = initialize_count_data(
            args[f"{assay_type}_barcodes"],
            args[f"{assay_type}_X_count"],
            args[f"{assay_type}_A_allele"],
            args[f"{assay_type}_B_allele"],
            args[f"{assay_type}_cnv_segments"],
            assay_type,
            cell_type_df,
            args["ref_label"],
            args["exclude_cell_types"],
        )
        cd.annotate_cnps(bbc_phases, seg_df, clones, cn_A, cn_B, cn_C, BAF)
        bbc_count_datas[assay_type] = cd

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

    # per-assay genomic CNP tables (segment_count_data already returns dense)
    seg_cnv_blocks = {a: count_data_cnv_blocks(cd) for a, cd in seg_count_datas.items()}
    bin_cnv_blocks = {a: count_data_cnv_blocks(cd) for a, cd in bin_count_datas.items()}
    seg_cnv_blocks[assay_types[0]].to_csv(
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

    if args["method"] == "copytyping":
        plot_count_histograms(
            read_counts={a: cd.X for a, cd in seg_count_datas.items()},
            total_allele_counts={a: cd.A + cd.B for a, cd in seg_count_datas.items()},
            cn_A={a: cd.cn_A for a, cd in seg_count_datas.items()},
            cn_B={a: cd.cn_B for a, cd in seg_count_datas.items()},
            cn_C={a: cd.cn_C for a, cd in seg_count_datas.items()},
            barcodes={a: cd.barcodes for a, cd in seg_count_datas.items()},
            sample=args["sample"],
            outfile=os.path.join(plot_dir, f"{out_prefix}.count_histograms.pdf"),
            dpi=args["dpi"],
        )

    for assay_type in assay_types:
        seg_cd = seg_count_datas[assay_type]
        bin_cd = bin_count_datas[assay_type]
        seg_blocks = seg_cnv_blocks[assay_type]
        bin_blocks = bin_cnv_blocks[assay_type]
        seg_baseline = compute_rdr_baseline(seg_cd, is_reference, ref_clone, no_normal)
        bin_baseline = compute_rdr_baseline(bin_cd, is_reference, ref_clone, no_normal)
        theta = model_params.get(f"{assay_type}-theta")
        rep_ids = sorted(seg_cd.barcodes["REP_ID"].unique())

        # 2D BAF-vs-log2RDR scatter per CNP segment (all cells)
        plot_scatter_2d_per_cell(
            seg_cd.X,
            seg_cd.B,
            seg_cd.A + seg_cd.B,
            seg_cd.cn_A,
            seg_cd.cn_B,
            seg_cd.cn_C,
            seg_cd.clones,
            seg_blocks,
            anns,
            args["sample"],
            os.path.join(plot_dir, f"{out_prefix}.{assay_type}.cluster_2d.pdf"),
            label,
            base_props=seg_baseline,
            dpi=args["dpi"],
        )

        # CNV heatmaps: one PDF per agg level; pages = rep_id x [BAF, log2RDR]
        for agg in [1, args["heatmap_agg"]]:
            fname = os.path.join(
                plot_dir, f"{out_prefix}.{assay_type}.heatmap.agg{agg}.pdf"
            )
            with PdfPages(fname) as pdf:
                for rep_id in rep_ids:
                    seg_rep, rep_mask = seg_cd.subset_by_rep(rep_id)
                    anns_rep = anns.iloc[rep_mask].reset_index(drop=True)
                    theta_rep = theta[rep_mask] if theta is not None else None
                    for val in ["BAF", "log2RDR"]:
                        if val == "log2RDR" and seg_baseline is None:
                            continue
                        plot_cnv_heatmap(
                            args["sample"],
                            assay_type,
                            seg_blocks,
                            seg_rep.X,
                            seg_rep.B,
                            seg_rep.A + seg_rep.B,
                            seg_blocks,
                            len(seg_cd.clones),
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
                    bin_rep, rep_mask = bin_cd.subset_by_rep(rep_id)
                    anns_rep = anns.iloc[rep_mask].reset_index(drop=True)
                    plot_rdr_baf_1d_pseudobulk(
                        bin_rep.X,
                        bin_rep.B,
                        bin_rep.A + bin_rep.B,
                        bin_rep.cn_A,
                        bin_rep.cn_B,
                        bin_rep.cn_C,
                        bin_rep.cn_BAF,
                        bin_rep.clones,
                        bin_blocks,
                        anns_rep,
                        bin_baseline,
                        args["sample"],
                        assay_type,
                        args["genome_size"],
                        args["region_bed"],
                        haplo_blocks=seg_blocks,
                        lab_type=my_label,
                        is_inferred=(my_label == label),
                        pdf_pages=pdf,
                        platform=platform,
                        subtitle=f"rep={rep_id}",
                        ascn_profile=args["ascn_profile"],
                    )

    if platform in SPATIAL_PLATFORMS and args["gex_h5ad"]:
        cluster_gex = segment_count_data(
            {"gex": bbc_count_datas["gex"]}, "cnp_cluster"
        )["gex"]
        plot_visium_all(
            sample=args["sample"],
            anns=anns,
            h5ad_source=args["gex_h5ad"],
            ballele_counts=cluster_gex.B,
            total_allele_counts=cluster_gex.A + cluster_gex.B,
            cn_A=cluster_gex.cn_A,
            cn_B=cluster_gex.cn_B,
            cluster_barcodes=cluster_gex.barcodes,
            clones=cluster_gex.clones,
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


def run_copytyping(
    assay_types: list[str],
    bbc_data: dict[str, CountData],
    platform: str,
    label: str,
    spatial_graphs: dict | None,
    args: dict,
) -> tuple[Base_Model, pd.DataFrame, dict, float]:
    """Cluster the annotated BBC CountData and fit the platform model.

    Operates purely on the CountData ``bbc_data`` (+ spatial neighbor graphs); the
    legacy SX_Data is never touched. Returns ``(model, anns, model_params,
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


def run_kmeans():
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
