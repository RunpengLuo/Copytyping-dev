import argparse
import copy
import logging
import os

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from copytyping.copytyping_parser import (
    add_arguments_inference,
    check_arguments_inference,
)
from copytyping.inference.cell_model import Cell_Model
from copytyping.inference.clustering import kmeans_copytyping, cluster_label_major_vote
from copytyping.inference.inference_utils import (
    annotate_adata_celltype,
)
from copytyping.inference.model_utils import prepare_params
from copytyping.inference.spot_model import Spot_Model
from copytyping.io_utils import (
    load_modality_data,
    load_spatial_neighbors,
    union_align_barcodes,
)
from copytyping.sx_data.sx_data import SX_Data
from copytyping.utils import (
    SPATIAL_PLATFORMS,
    add_file_logging,
    save_phased_bbc,
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
    proc_dir = os.path.join(out_dir, "processed_data")
    os.makedirs(proc_dir, exist_ok=True)

    logging.info(f"sample={sample}, platform={platform}, data_types={data_types}")

    cell_type_df = None
    if args.get("cell_type") is not None:
        cell_type_df = pd.read_table(args["cell_type"])
        assert "BARCODE" in cell_type_df.columns
        assert ref_label in cell_type_df.columns

    data_sources = {}
    raw_data_sources = {}
    seg_data_sources = {}
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

        # Build the FULL seg_sx (saved to disk so validate can plot all CN states)
        seg_sx_full = SX_Data(
            barcodes_df, seg_df, X_seg, Y_seg, D_seg, baf_clip=args["baf_clip"]
        )
        seg_data_sources[data_type] = seg_sx_full

        # Save BBC-level data for validate to recompute agg_bbc
        save_phased_bbc(
            bbc_df,
            X_bbc,
            Y_bbc,
            D_bbc,
            os.path.join(proc_dir, f"{out_prefix}.{data_type}.bbc"),
        )

        # Optional: filter CN rows for the model fit only (full data still saved)
        keep_cn_row = args.get("keep_cn_row")
        if keep_cn_row:
            keep_set = {r.strip() for r in keep_cn_row.split(",") if r.strip()}
            keep_mask = seg_df["CNP"].isin(keep_set).to_numpy()
            n_kept = int(keep_mask.sum())
            logging.info(
                f"[{data_type}] keep_cn_row: model uses {n_kept}/{len(seg_df)} "
                f"segments; full data saved to disk (whitelist: {sorted(keep_set)})"
            )
            assert n_kept > 0, "keep_cn_row matched 0 segments; check format vs CNP"
            seg_sx = SX_Data(
                barcodes_df,
                seg_df[keep_mask].reset_index(drop=True),
                X_seg[keep_mask],
                Y_seg[keep_mask],
                D_seg[keep_mask],
                baf_clip=args["baf_clip"],
            )
        else:
            seg_sx = seg_sx_full

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
        max_k = args.get("max_smooth_k", 0)
        if data_type in spatial_graphs and max_k > 0:
            seg_sx_smoothed = copy.deepcopy(seg_sx)
            seg_sx_smoothed.apply_adaptive_smoothing(
                spatial_graphs[data_type],
                max_k=max_k,
                min_umi=args.get("min_umi_per_spot", 0),
                min_snp_umi=args.get("min_snp_umi_per_spot", 0),
            )
            data_sources[data_type] = seg_sx_smoothed.to_cluster_level()
        else:
            data_sources[data_type] = raw_data_sources[data_type]

    # Save segment CNV profile (cnv_blocks from first data_type)
    seg_data_sources[data_types[0]].cnv_blocks.to_csv(
        os.path.join(proc_dir, f"{out_prefix}.cnp_profile.tsv"),
        sep="\t",
        index=False,
    )

    barcodes, modality_masks = union_align_barcodes(data_sources, data_types)

    cnv_blocks = seg_data_sources[data_types[0]].cnv_blocks
    init_params, fix_params = prepare_params(args, cnv_blocks, platform, data_types)

    instance = {"single_cell": Cell_Model, "spatial": Spot_Model}[platform](
        barcodes,
        platform,
        data_types,
        data_sources,
        proc_dir,
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

    # Save init normal labels
    if instance.labeling_trace:
        init_labels = instance.labeling_trace[0]["labels"]
        init_df = pd.DataFrame(
            {
                "BARCODE": barcodes["BARCODE"].values,
                "init_label": init_labels,
            }
        )
        init_df.to_csv(
            os.path.join(proc_dir, f"{out_prefix}.init_labels.tsv"),
            sep="\t",
            index=False,
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
    # ---- Save outputs to processed_data ----
    anns.to_csv(
        os.path.join(out_dir, f"{out_prefix}.{platform}.annotations.tsv"),
        sep="\t",
        header=True,
        index=False,
    )

    # Save barcodes
    barcodes.to_csv(
        os.path.join(proc_dir, f"{out_prefix}.barcodes.tsv"), sep="\t", index=False
    )

    # Save segment-level count matrices
    for data_type in data_types:
        sx = seg_data_sources[data_type]
        prefix = os.path.join(proc_dir, f"{out_prefix}.{data_type}")
        sparse.save_npz(f"{prefix}.seg.X.npz", sparse.csr_matrix(sx.X))
        sparse.save_npz(f"{prefix}.seg.Y.npz", sparse.csr_matrix(sx.Y))
        sparse.save_npz(f"{prefix}.seg.D.npz", sparse.csr_matrix(sx.D))
        logging.info(f"saved {data_type} count matrices: X/Y/D shape=({sx.G}, {sx.N})")

    # Save model params
    param_dict = {"log_likelihood": np.array([final_ll])}
    for data_type in data_types:
        for key in ["lambda", "theta", "tau", "inv_phi"]:
            pk = f"{data_type}-{key}"
            if pk in model_params:
                param_dict[pk.replace("-", "_")] = np.atleast_1d(model_params[pk])
    np.savez(os.path.join(proc_dir, f"{out_prefix}.model_params.npz"), **param_dict)

    # Save labeling trace
    if hasattr(instance, "labeling_trace") and instance.labeling_trace:
        trace_dict = {}
        for i, lt in enumerate(instance.labeling_trace):
            trace_dict[f"labels_{i}"] = lt["labels"]
            trace_dict[f"max_posterior_{i}"] = lt["max_posterior"]
            if "tumor_purity" in lt:
                trace_dict[f"tumor_purity_{i}"] = lt["tumor_purity"]
        trace_dict["n_iters"] = np.array([len(instance.labeling_trace)])
        np.savez(
            os.path.join(proc_dir, f"{out_prefix}.labeling_trace.npz"), **trace_dict
        )

    # Save metadata
    meta = {
        "sample": sample,
        "platform": platform,
        "data_types": ",".join(data_types),
        "label": label,
        "ref_label": ref_label,
    }
    pd.Series(meta).to_csv(
        os.path.join(proc_dir, f"{out_prefix}.metadata.tsv"),
        sep="\t",
        header=False,
    )

    logging.info(f"inference complete. outputs in {out_dir}")
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
