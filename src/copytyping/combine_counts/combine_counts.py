import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

from copytyping.copytyping_parser import add_arguments_combine_counts
from copytyping.utils import *
from copytyping.io_utils import *
from copytyping.combine_counts.combine_counts_utils import *
from copytyping.sx_data.sx_data import *
from copytyping.plot.plot_cell import plot_cnv_heatmap
from copytyping.plot.plot_common import plot_library_sizes


def run(args=None):
    print("run copytyping combine counts")
    if isinstance(args, argparse.Namespace):
        args = vars(args)

    out_dir = args["out_dir"]

    genome_segment = args["genome_segment"]
    genome_size = args["genome_size"]

    seg_ucn = args["seg_ucn"]
    snp_file = args["snp_file"]
    sample_file = args["sample_file"]

    min_cnv_length = args["min_cnv_length"]

    os.makedirs(out_dir, exist_ok=True)
    prep_dir = os.path.join(out_dir, "preprocess")
    plot_dir = os.path.join(prep_dir, "plot")
    tmp_dir = os.path.join(prep_dir, "tmp")
    for d in [prep_dir, plot_dir, tmp_dir]:
        os.makedirs(d, exist_ok=True)

    ##################################################
    wl_fragments = pd.read_table(
        genome_segment,
        sep="\t",
        header=None,
        names=["#CHR", "START", "END", "NAME"],
    )

    ##################################################
    # load copy-number profile
    haplo_blocks, cnv_Aallele, cnv_Ballele, cnv_mixBAF, clone_props = load_cnv_profile(
        seg_ucn, min_cnv_length=min_cnv_length
    )
    haplo_blocks["HB"] = haplo_blocks.index.to_numpy()
    haplo_block_file = os.path.join(prep_dir, "haplotype_blocks.tsv")
    haplo_blocks.to_csv(haplo_block_file, header=True, sep="\t", index=False)

    snp_info = pd.read_table(snp_file, sep="\t")
    snp_info = annotate_snps(haplo_blocks, snp_info, seg_id="HB")

    get_ref_label = lambda d: {
        "GEX": "cell_type",
        "ATAC": "cell_type",
        "VISIUM": "path_label",
    }[d]

    ##################################################
    sample_df = pd.read_table(sample_file, sep="\t", index_col=False).fillna("")
    for _, sample_row in sample_df.iterrows():
        sample, rep_id, data_type = sample_row[["SAMPLE", "REP_ID", "DATA_TYPE"]]
        print(f"process {sample} {rep_id} {data_type}")

        # output
        data_id = data_type if rep_id == "" else f"{data_type}_{rep_id}"
        mod_dir = os.path.join(prep_dir, data_id)
        os.makedirs(mod_dir, exist_ok=True)

        path2h5ad = sample_row["PATH_to_h5ad"]
        path2barcodes = sample_row["PATH_to_barcodes"]
        path2cellsnp = sample_row["PATH_to_cellsnp_lite"]
        path2ann = sample_row["PATH_to_annotation"] or None

        ref_label = get_ref_label(data_type)

        # TODO assumed barcodes are pre-sorted if multiome
        barcodes_df = read_barcodes_celltype_one_replicate(
            path2barcodes, data_type, path2ann, rep_id, ref_label=ref_label
        )
        barcodes = barcodes_df["BARCODE"].tolist()
        barcodes_df.to_csv(
            os.path.join(mod_dir, "Barcodes.tsv"), sep="\t", index=False, header=True
        )

        ##################################################
        # CNV segmentation, total features
        adata: AnnData = sc.read_h5ad(path2h5ad)
        adata, features = feature_to_haplo_blocks(adata, haplo_blocks, data_id)
        feature_bins = feature_binning(
            adata,
            features,
            tmp_dir,
            data_id,
            feature_may_overlap="ATAC" not in data_id,
            binning_strategy="segment",
            bin_colname="BIN_ID",
        )
        aggregate_var_counts(
            adata,
            feature_bins,
            data_id,
            agg_colname="BIN_ID",
            out_file=os.path.join(mod_dir, f"count.npz"),
        )
        adata.write_h5ad(os.path.join(mod_dir, f"{data_type}.h5ad"), compression="gzip")

        ##################################################
        # CNV segmentation, allele features
        allele_data = load_cellsnp_files(path2cellsnp, barcodes)
        [cell_snps, dp_mat, ref_mat, alt_mat] = annotate_snps_post2(
            snp_info, allele_data
        )
        cell_snps = snp_to_region(
            cell_snps, feature_bins, data_type, region_id="BIN_ID"
        )
        a_allele_mat, b_allele_mat, t_allele_mat = aggregate_allele_counts(
            feature_bins,
            cell_snps,
            dp_mat,
            ref_mat,
            alt_mat,
            data_type,
            agg_mode="cellsnp-lite",
            agg_colname="BIN_ID",
        )

        allele_bins = allele_binning(
            feature_bins,
            t_allele_mat,
            data_type,
            binning_strategy="segment",
            bin_colname="SNP_BIN_ID",
        )

        aggregate_allele_counts(
            allele_bins,
            feature_bins,
            t_allele_mat,
            b_allele_mat,
            a_allele_mat,
            data_type,
            agg_mode="regular",
            agg_colname="SNP_BIN_ID",
            out_dir=mod_dir,
            out_prefix="",
        )

        ##################################################
        # save book-keeping informations
        features.to_csv(
            os.path.join(mod_dir, f"raw_features.tsv.gz"),
            sep="\t",
            index=False,
            header=True,
        )
        features.loc[mark_imbalanced(features), :].to_csv(
            os.path.join(mod_dir, f"raw_features.imb.tsv.gz"),
            sep="\t",
            index=False,
            header=True,
        )

        feature_bins.to_csv(
            os.path.join(mod_dir, f"feature_bins.tsv"),
            sep="\t",
            index=False,
            header=True,
        )
        feature_bins.loc[mark_imbalanced(feature_bins), :].to_csv(
            os.path.join(mod_dir, f"feature_bins.imb.tsv"),
            sep="\t",
            index=False,
            header=True,
        )

        allele_bins.to_csv(
            os.path.join(mod_dir, "allele_bins.tsv"),
            sep="\t",
            index=False,
            header=True,
        )
        allele_bins.loc[mark_imbalanced(allele_bins), :].to_csv(
            os.path.join(mod_dir, "allele_bins.imb.tsv"),
            sep="\t",
            index=False,
            header=True,
        )

        ##################################################
        sx_data = SX_Data(len(barcodes), mod_dir, data_type)
        plot_cnv_heatmap(
            sample,
            data_id,
            haplo_blocks,
            sx_data,
            barcodes_df if ref_label in barcodes_df else None,
            wl_fragments,
            val="BAF",
            lab_type=ref_label,
            filename=os.path.join(plot_dir, f"{sample}.BAF_heatmap.{data_id}.pdf"),
            agg_size=10 if ref_label in barcodes_df else 1,
            dpi=300,
        )

        plot_library_sizes(
            sx_data,
            sample,
            data_id,
            os.path.join(plot_dir, f"{data_id}.library_size.png"),
            barcodes[ref_label].to_numpy() if ref_label in barcodes else None,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Copytyping combine_counts",
        description="combine counts",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_arguments_combine_counts(parser)
    args = parser.parse_args()
    run(args)
