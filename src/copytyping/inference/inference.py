import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd

import scanpy as sc
from scanpy import AnnData

from copytyping.utils import *
from copytyping.copytyping_parser import (
    add_arguments_inference,
    check_arguments_inference,
)
from copytyping.inference.cell_model import *
from copytyping.inference.spot_model import *

# from copytyping.inference.clustering import *
from copytyping.inference.validation import *
from copytyping.plot.plot_common import *
from copytyping.plot.plot_cell import *
from copytyping.plot.plot_visium import plot_visium_HE
from copytyping.plot.plot_umap import *

"""
Given X/Y/D count matrices and copy-number profile.
compute per-cell/spot clone-level posterior probabilies, and MAP solution.
If multiome data, assume barcode files and CNP files have same content for gex/atac
as preprocessed by unified-genotyping pipelien
"""


def run(args=None):
    logging.info(f"run copytyping inference")
    if isinstance(args, argparse.Namespace):
        args = vars(args)

    args = check_arguments_inference(args)
    sample = args["sample"]
    assay_type = args["assay_type"]
    data_types = args["data_types"]
    ref_label = args["ref_label"]
    out_dir = args["out_dir"]
    out_prefix = args["out_prefix"]
    if out_prefix == "":
        out_prefix = str(sample)
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    verbosity = args["verbosity"]

    logging.info(f"sample={sample}, assay_type={assay_type}, data_types={data_types}")

    ##################################################
    # load data
    data_sources = {}
    adatas = {}
    for data_type in data_types:
        sx_data = SX_Data(
            args[f"{data_type}_barcodes"],
            args[f"{data_type}_cnv_segments"],
            args[f"{data_type}_X_count"],
            args[f"{data_type}_Y_count"],
            args[f"{data_type}_D_count"],
            data_type,
        )
        adata: AnnData = sc.read_h5ad(args[f"{data_type}_h5ad"])
        if ref_label in adata.obs.columns:
            obs_df = adata.obs.reset_index(names=["BARCODE"])
            sx_data.barcodes = pd.merge(
                left=sx_data.barcodes,
                right=obs_df[["BARCODE", ref_label]],
                on="BARCODE",
                how="left",
                validate="1:1",
                sort=False,
            )
            sx_data.barcodes[ref_label] = (
                sx_data.barcodes[ref_label].fillna("Unknown").astype(str)
            )
        adatas[data_type] = adata
        data_sources[data_type] = sx_data
    barcodes: pd.DataFrame = data_sources[data_types[0]].barcodes.copy()
    cnv_blocks = data_sources[data_types[0]].cnv_blocks

    ##################################################
    # load config parameters
    method = args["method"]
    fit_mode = "hybrid"  # allele + total
    label = f"{method}-label"
    num_iters = args["niters"]
    posterior_thres = args["posterior_thres"]
    margin_thres = args["margin_thres"]

    bulk_props = np.array(list(map(float, cnv_blocks["PROPS"].iloc[0].split(";"))))
    init_params = {"pi": bulk_props, "tau0": args["tau"], "phi0": args["phi"]}
    fix_params = {"pi": True}
    share_params = {}
    for data_type in data_types:
        fix_params[f"{data_type}-inv_phi"] = args["fix_NB_dispersion"]
        share_params[f"{data_type}-inv_phi"] = args["share_NB_dispersion"]
        fix_params[f"{data_type}-tau"] = args["fix_BB_dispersion"]
        share_params[f"{data_type}-tau"] = args["share_BB_dispersion"]
        fix_params[f"{data_type}-theta"] = (
            assay_type in SPOT_ASSAYS and args["fix_tumor_purity"]
        )

    if assay_type in CELL_ASSAYS:
        model = Cell_Model
    elif assay_type in SPOT_ASSAYS:
        model = Spot_Model
    else:
        raise ValueError(f"unknown assay_type={assay_type}")

    instance = model(
        barcodes,
        assay_type,
        data_types,
        data_sources,
        out_dir,
        out_prefix,
        verbosity,
    )
    model_params = instance.fit(
        fit_mode,
        fix_params=fix_params,
        init_params=init_params,
        share_params=share_params,
        max_iter=num_iters,
    )
    anns, clone_props = instance.predict(
        fit_mode,
        model_params,
        label=label,
        posterior_thres=posterior_thres,
        margin_thres=margin_thres,
    )
    logging.info(f"clone fractions: {clone_props}")
    ##################################################
    if ref_label in barcodes.columns:
        logging.info("evaluate performance against reference labels")
        if args["refine_label_by_celltype"]:
            anns = refine_labels_by_reference(
                anns, ref_label, label, f"{label}-refined"
            )
        metric, metric_str = evaluate_malignant_accuracy(
            anns, cell_label=label, cell_type=ref_label
        )
        metric["SAMPLE"] = sample
        pd.DataFrame([metric]).to_csv(
            os.path.join(out_dir, f"{out_prefix}.{assay_type}.evaluation.tsv"),
            sep="\t",
            header=True,
            index=False,
        )
        plot_cross_heatmap(
            anns,
            sample,
            os.path.join(plot_dir, f"{out_prefix}.{assay_type}.cross_heatmap.png"),
            acol=label,
            bcol=ref_label,
        )
    anns.to_csv(
        os.path.join(out_dir, f"{out_prefix}.{assay_type}.annotations.tsv"),
        sep="\t",
        header=True,
        index=False,
    )

    plot_posteriors(
        anns,
        os.path.join(plot_dir, f"{out_prefix}.{assay_type}.posteriors.png"),
        lab_type=ref_label if ref_label in anns else label,
    )

    ##################################################
    wl_segments = read_whitelist_segments(args["region_bed"])
    genome_size = args["genome_size"]
    agg_size = args["heatmap_agg"]
    img_type = args["img_type"]
    dpi = args["dpi"]
    transparent = args["transparent"]
    for data_type in data_types:
        sx_data: SX_Data = data_sources[data_type]
        adata: AnnData = adatas[data_type]

        # plot heatmap
        for val in ["BAF", "pi_gk", "log2RDR"]:
            if val != "BAF" and f"{data_type}-lambda" not in model_params:
                continue
            for my_label in [label, ref_label]:
                if my_label not in anns:
                    continue
                for agg in [1, agg_size]:
                    plot_cnv_heatmap(
                        sample,
                        data_type,
                        cnv_blocks,
                        sx_data,
                        anns,
                        wl_segments,
                        proportions=model_params.get(f"{data_type}-theta", None),
                        val=val,
                        base_props=model_params.get(f"{data_type}-lambda", None),
                        agg_size=agg,
                        lab_type=my_label,
                        filename=os.path.join(
                            plot_dir,
                            f"{out_prefix}.{assay_type}.{val}_heatmap.{data_type}.agg{agg}.{my_label}.{img_type}",
                        ),
                        dpi=dpi,
                        figsize=(20, 6 if agg > 1 else 15),
                        transparent=transparent,
                    )

        # plot 1d aggregated scatter plot
        plot_rdr_baf_1d_aggregated(
            sx_data,
            anns,
            None,
            sample,
            data_type,
            genome_size,
            mask_cnp=False,
            lab_type=label,
            filename=os.path.join(
                plot_dir,
                f"{out_prefix}.{assay_type}.1d_scatter.{data_type}.{label}.{img_type}",
            ),
        )

    if args["umap"]:
        # TODO
        pass

    if assay_type in SPATIAL_ASSAYS:
        # TODO
        pass
        # plot_visium_HE(sample, anns, adata)


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
