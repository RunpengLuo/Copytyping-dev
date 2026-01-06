import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

import scanpy as sc

from copytyping.utils import *
from copytyping.copytyping_parser import add_arguments_inference
from copytyping.inference.cell_model import *
from copytyping.inference.spot_model import *
from copytyping.inference.clustering import *
from copytyping.inference.validation import *
from copytyping.plot.plot_common import *
from copytyping.plot.plot_cell import *
from copytyping.plot.plot_umap import *


# def run_sample(
#     sample: str,
#     rep_id: str,
#     haplo_blocks: pd.DataFrame,
#     wl_segments: pd.DataFrame,
#     barcodes: pd.DataFrame,
#     init_params: dict,
#     fix_params: dict,
#     data_types: list,
#     mod_dirs: dict,
#     modality: str,
#     method: str,
#     mode: str,
#     plot_dir: str,
#     out_dir: str,
#     genome_file: str,
#     refine_label_by_reference=True,
#     ref_label="cell_type",
#     agg_size=10,
#     posterior_thres=0.65,
#     margin_thres=0.1,
#     img_type="png",
#     transparent=False,
#     dpi=300,
#     show_metric_heatmap=False,
# ):
#     label = method
#     model = Cell_Model(barcodes, data_types, mod_dirs, out_dir, modality)
#     if method == "copytyping":
#         params = model.fit(
#             mode,
#             fix_params=fix_params,
#             init_params=init_params,
#         )
#         anns, clone_props = model.predict(
#             mode,
#             params,
#             label=label,
#             posterior_thres=posterior_thres,
#             margin_thres=margin_thres,
#         )
#     else:
#         params = {}
#         for data_type in model.data_types:
#             params[f"{data_type}-lambda"] = model.initialize_baseline_proportions(
#                 data_type
#             )
#         if method == "kmeans":
#             cluster_labels = kmeans_copytyping(model, params)
#         elif method == "leiden":
#             resolution = 0.5
#             cluster_labels = leiden_copytyping(model, params, resolution=resolution)
#         elif method == "ward":
#             cluster_labels = ward_copytyping(model, params)
#         anns = model.barcodes.copy(deep=True)
#         anns, clone_props = cluster_label_major_vote(
#             anns, cluster_labels, cell_label=label, cell_type=ref_label
#         )
#     print(f"clone proportions: {clone_props}")

#     ##################################################
#     # evaluation
#     title_info = f"{method}; "
#     if ref_label in anns:
#         if refine_label_by_reference:
#             anns = refine_labels_celltype(
#                 anns,
#                 cell_label=label,
#                 out_label=f"{label}-refined",
#                 cell_type=ref_label,
#             )
#         metric, metric_str = evaluate_malignant_accuracy(
#             anns, cell_label=label, cell_type=ref_label
#         )
#         title_info += metric_str
#         plot_cross_heatmap(
#             anns,
#             sample,
#             os.path.join(plot_dir, f"{sample}.{rep_id}.{method}.png"),
#             acol=label,
#             bcol="cell_type",
#         )
#         # save evalution metric to csv
#         metric["SAMPLE"] = sample
#         pd.DataFrame([metric]).to_csv(
#             os.path.join(out_dir, f"{sample}.{rep_id}.evaluation.{method}.tsv"),
#             sep="\t",
#             header=True,
#             index=False,
#         )
#     anns.to_csv(
#         os.path.join(out_dir, f"{sample}.{rep_id}.annotations.{method}.tsv"),
#         sep="\t",
#         header=True,
#         index=False,
#     )

#     if method == "copytyping":
#         plot_posteriors(
#             anns,
#             os.path.join(plot_dir, f"{sample}.{rep_id}.posteriors.{method}.png"),
#             lab_type="cell_type" if "cell_type" in anns else label,
#         )

#     ##################################################
#     # visualization
#     umap_features = []
#     for data_type in model.data_types:
#         os.makedirs(os.path.join(plot_dir, f"{data_type}_heatmap"), exist_ok=True)
#         os.makedirs(os.path.join(plot_dir, f"{data_type}_others"), exist_ok=True)
#         sx_data: SX_Data = model.data_sources[data_type]
#         features = prepare_rdr_baf_features(sx_data, params[f"{data_type}-lambda"])
#         umap_features.append(features)
#         plot_umap_copynumber(
#             sample,
#             data_type,
#             features,
#             anns,
#             lab_type=label,
#             filename=os.path.join(
#                 plot_dir,
#                 f"{data_type}_others",
#                 f"{sample}.{rep_id}.UMAP.{data_type}.{label}.pdf",
#             ),
#             dpi=dpi,
#         )
#         for val in ["BAF", "pi_gk", "log2RDR"]:
#             if val in ["log2RDR", "pi_gk"] and f"{data_type}-lambda" not in params:
#                 continue
#             for my_label in [label, "cell_type"]:
#                 if my_label not in anns:
#                     continue
#                 for agg in [1, agg_size]:
#                     plot_cnv_heatmap(
#                         sample,
#                         data_type,
#                         haplo_blocks,
#                         sx_data,
#                         anns,
#                         wl_segments,
#                         val=val,
#                         base_props=params.get(f"{data_type}-lambda", None),
#                         agg_size=agg,
#                         lab_type=my_label,
#                         filename=os.path.join(
#                             plot_dir,
#                             f"{data_type}_heatmap",
#                             f"{sample}.{rep_id}.{val}_heatmap.{data_type}.agg{agg}.{my_label}.{img_type}",
#                         ),
#                         dpi=dpi,
#                         figsize=(20, 6 if agg > 1 else 15),
#                         title_info=title_info if show_metric_heatmap else "",
#                         transparent=transparent,
#                     )

#         plot_rdr_baf_1d_aggregated(
#             sx_data,
#             anns,
#             None,
#             sample,
#             data_type,
#             genome_file,
#             mask_cnp=False,
#             lab_type=label,
#             filename=os.path.join(
#                 plot_dir,
#                 f"{data_type}_others",
#                 f"{sample}.{rep_id}.scatter.{data_type}.{label}.pdf",
#             ),
#         )
#     if len(umap_features) > 1:
#         features = np.concatenate(umap_features, axis=1)
#         plot_umap_copynumber(
#             sample,
#             "multiome",
#             features,
#             anns,
#             lab_type=label,
#             filename=os.path.join(
#                 plot_dir,
#                 f"{data_type}_others",
#                 f"{sample}.{rep_id}.UMAP.multiome.{label}.pdf",
#             ),
#             dpi=dpi,
#         )
#     #         plot_params(
#     #             params,
#     #             os.path.join(plot_dir, f"{sample}.{rep_id}.parameters.{data_type}.pdf"),
#     #             data_type,
#     #             names=["tau", "lambda", "inv_phi"],
#     #         )

#     return


def run(args=None):
    print("run copytyping inference")
    if isinstance(args, argparse.Namespace):
        args = vars(args)

    sample_file = args["sample_file"]
    genome_segment = args["genome_segment"]
    genome_size = args["genome_size"]
    method = args["method"]
    work_dir = args["work_dir"]
    out_prefix = args["out_prefix"]

    agg_size = 10
    img_type = args["img_type"]
    dpi = args["dpi"]
    transparent = args["transparent"]
    show_metric_heatmap = False
    # args["show_metric_heatmap"]

    # assigned cells has contradict labels with cell types are marked with NA.
    refine_label_by_reference = True
    get_ref_label = lambda d: {
        "GEX": "cell_type",
        "ATAC": "cell_type",
        "VISIUM": "path_label",
    }[d]
    qry_label = method

    prep_dir = os.path.join(work_dir, "preprocess")
    haplo_file = os.path.join(prep_dir, "haplotype_blocks.tsv")

    out_dir = os.path.join(work_dir, f"{out_prefix}_{method}_assignment")
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, f"{out_prefix}_{method}_plot")
    os.makedirs(plot_dir, exist_ok=True)
    sc.settings.figdir = plot_dir

    ##################################################
    wl_segments = read_whitelist_segments(genome_segment)
    haplo_blocks = pd.read_table(haplo_file, sep="\t")
    bulk_props = np.array(list(map(float, haplo_blocks["PROPS"].iloc[0].split(";"))))

    init_params = {"pi": bulk_props, "tau0": 50, "phi0": 30}
    fix_params = {
        "pi": True,
        # "GEX-tau": True,
        # "ATAC-tau": True,
        # "GEX-inv_phi": True,
        # "ATAC-inv_phi": True
    }
    num_iters = args["niters"]
    posterior_thres = args["posterior_thres"]
    margin_thres = args["margin_thres"]
    allele_only_posterior_thres = 0.95

    print(f"posterior_thres={posterior_thres}")
    print(f"margin_thres={margin_thres}")
    ##################################################
    sample_df = pd.read_table(sample_file, sep="\t", index_col=False).fillna("")
    for _, sample_infos in sample_df.groupby(by=["SAMPLE", "REP_ID"], sort=False):
        sample = sample_infos["SAMPLE"].iloc[0]
        rep_id = sample_infos["REP_ID"].iloc[0]
        data_types = sample_infos["DATA_TYPE"].unique().tolist()
        if len(sample_infos) > 1:
            assert len(sample_infos) == 2
            assert {"GEX", "ATAC"} <= set(data_types)
            modality = "multiome"
            barcode_file = os.path.join(prep_dir, f"GEX_{rep_id}/Barcodes.tsv")
            data_types = ["GEX", "ATAC"]
            mod_dirs = {
                "GEX": os.path.join(prep_dir, f"GEX_{rep_id}/"),
                "ATAC": os.path.join(prep_dir, f"ATAC_{rep_id}/"),
            }
        else:
            modality = data_types[0]
            barcode_file = os.path.join(prep_dir, f"{modality}_{rep_id}/Barcodes.tsv")
            data_types = [modality]
            mod_dirs = {
                modality: os.path.join(prep_dir, f"{modality}_{rep_id}/"),
            }
        ref_label = get_ref_label(data_types[0])

        barcodes = pd.read_table(barcode_file, sep="\t")
        print(f"process sample={sample}.{rep_id}, rep_id={rep_id}, modality={modality}")
        print(f"#barcodes={len(barcodes)}")

        ##################################################
        if modality in ["multiome", "GEX", "ATAC"]:
            model = Cell_Model(barcodes, haplo_blocks, data_types, mod_dirs, modality)
            params = model.fit(
                "hybrid",
                fix_params=fix_params,
                init_params=init_params,
                max_iter=num_iters,
            )
            anns, clone_props = model.predict(
                "hybrid",
                params,
                label=qry_label,
                posterior_thres=posterior_thres,
                margin_thres=margin_thres,
            )
            print(f"clone proportions: {clone_props}")
        else:
            model = Spot_Model(barcodes, haplo_blocks, data_types, mod_dirs, modality)
            pass
            # spot_model =

        ##################################################
        title_info = ""
        if ref_label in anns:
            if refine_label_by_reference:
                anns = refine_labels_celltype(
                    anns,
                    cell_label=qry_label,
                    out_label=f"{qry_label}-refined",
                    cell_type=ref_label,
                )
            metric, metric_str = evaluate_malignant_accuracy(
                anns, cell_label=qry_label, cell_type=ref_label
            )
            if show_metric_heatmap:
                title_info += metric_str
            # save evalution metric to csv
            metric["SAMPLE"] = sample
            pd.DataFrame([metric]).to_csv(
                os.path.join(out_dir, f"{sample}.{rep_id}.evaluation.{method}.tsv"),
                sep="\t",
                header=True,
                index=False,
            )
        anns.to_csv(
            os.path.join(out_dir, f"{sample}.{rep_id}.annotations.{method}.tsv"),
            sep="\t",
            header=True,
            index=False,
        )
        plot_posteriors(
            anns,
            os.path.join(plot_dir, f"{sample}.{rep_id}.posteriors.{method}.png"),
            lab_type=ref_label if ref_label in anns else qry_label,
        )

        ##################################################
        # plot heatmap
        for data_type in model.data_types:
            os.makedirs(os.path.join(plot_dir, f"{data_type}_heatmap"), exist_ok=True)
            sx_data: SX_Data = model.data_sources[data_type]
            for val in ["BAF", "pi_gk", "log2RDR"]:
                if val in ["log2RDR", "pi_gk"] and f"{data_type}-lambda" not in params:
                    continue
                for my_label in [qry_label, ref_label]:
                    if my_label not in anns:
                        continue
                    for agg in [1, agg_size]:
                        plot_cnv_heatmap(
                            sample,
                            data_type,
                            haplo_blocks,
                            sx_data,
                            anns,
                            wl_segments,
                            val=val,
                            base_props=params.get(f"{data_type}-lambda", None),
                            agg_size=agg,
                            lab_type=my_label,
                            filename=os.path.join(
                                plot_dir,
                                f"{data_type}_heatmap",
                                f"{sample}.{rep_id}.{val}_heatmap.{data_type}.agg{agg}.{my_label}.{img_type}",
                            ),
                            dpi=dpi,
                            figsize=(20, 6 if agg > 1 else 15),
                            title_info=title_info,
                            transparent=transparent,
                        )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Copytyping inference",
        description="copytyping inference",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_arguments_inference(parser)
    args = parser.parse_args()
    run(args)
