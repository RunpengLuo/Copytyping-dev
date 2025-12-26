import os
import sys
import argparse

import numpy as np
from scanpy import AnnData
import snapatac2 as snap

from copytyping.utils import *

"""
This is a preprocessing script to build fragment count peak matrix for scATAC-seq
Output
# <unique_index> | <idx_col=gene_ids> feature_types <genome> pseudobulk_counts #CHR START END
"""


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        prog="Copytyping preprocessing - build tile matrix scATAC-seq",
        description="annotate Anndata",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--sample",
        required=True,
        type=str,
        help="sample name, used for output prefix",
    )
    parser.add_argument(
        "--ranger_dir",
        required=True,
        type=str,
        help="outs/ by cell-ranger",
    )
    parser.add_argument(
        "--barcodes",
        required=True,
        type=str,
        help="barcodes.txt",
    )
    parser.add_argument(
        "--genome_file",
        required=True,
        type=str,
        help="reference genome size file",
    )
    parser.add_argument(
        "--celltype_file", required=False, type=str, help="cell type file (optional)"
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        required=True,
        type=str,
        help="output directory",
    )
    args = parser.parse_args()
    return vars(args)


def read_barcodes(bc_file: str):
    barcodes = []
    with open(bc_file, "r") as fd:
        for line in fd:
            barcodes.append(line.strip().split("\t")[0])
        fd.close()
    return barcodes


if __name__ == "__main__":
    args = parse_arguments()
    sample = args["sample"]
    ranger_dir = args["ranger_dir"]
    barcode_file = args["barcodes"]
    genome_file = args["genome_file"]
    celltype_file = args["celltype_file"]
    out_dir = args["out_dir"]
    n_jobs = 4

    fragment_file = os.path.join(ranger_dir, "atac_fragments.tsv.gz")
    peak_file = os.path.join(ranger_dir, "atac_peaks.bed")
    assert os.path.exists(fragment_file) and os.path.exists(peak_file)
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    ##################################################
    barcodes = read_barcodes(barcode_file)
    print(f"#barcodes={len(barcodes)}")

    ##################################################
    chrom_sizes = get_chr_sizes(genome_file)

    ##################################################
    print(f"load anndata")
    adata: AnnData = snap.pp.import_fragments(
        fragment_file,
        chrom_sizes,
        whitelist=barcodes,
        min_num_fragments=0,
        sorted_by_barcode=False,
        tempdir=tmp_dir,
        n_jobs=n_jobs,
    )
    print(f"raw AnnData")
    print(adata)

    ##################################################
    print("build peak matrix")
    snap.pp.make_peak_matrix(
        adata,
        peak_file=peak_file,
        counting_strategy="fragment",
        inplace=True,
    )

    ##################################################
    if not celltype_file is None:
        print("append cell-type information")
        celltypes = read_celltypes(celltype_file)
        adata.obs_names = adata.obs_names.astype(str)
        print(
            "#obs_names&cell_types Overlap:",
            len(set(adata.obs_names) & set(celltypes["BARCODE"])),
        )
        adata.obs["cell_type"] = (
            celltypes.set_index("BARCODE")
            .reindex(adata.obs_names)["cell_type"]
            .fillna("Unknown")
            .values.astype(str)
        )

    ##################################################
    # filter no count bins
    adata.var["pseudobulk_counts"] = np.asarray(adata.X.sum(axis=0)).flatten()
    adata = adata[:, adata.var["pseudobulk_counts"] > 0].copy()

    ##################################################
    # annotate positions
    adata.var["#CHR"] = adata.var_names.str.split(":").str[0].astype(str)
    chs = sort_chroms(adata.var["#CHR"].unique().tolist())
    adata.var["#CHR"] = pd.Categorical(adata.var["#CHR"], categories=chs, ordered=True)
    adata.var["START"] = (
        adata.var_names.str.split(":").str[1].str.split("-").str[0].astype(int)
    )
    adata.var["END"] = (
        adata.var_names.str.split(":").str[1].str.split("-").str[1].astype(int)
    )

    assert adata.var_names.is_unique, "index is not unique!"
    sort_index = adata.var.sort_values(by=["#CHR", "START"]).index
    adata = adata[:, sort_index].copy()

    print(f"final processed AnnData")
    print(adata)
    print(adata.var.head(3))

    ##################################################
    out_file = os.path.join(out_dir, f"{sample}.atac_peak.h5ad")
    adata.write_h5ad(out_file, compression="gzip")
    print(f"final output={out_file}")
    sys.exit(0)
