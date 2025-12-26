import os
import sys
import argparse

import numpy as np
import pyranges as pr
import scanpy as sc

from copytyping.utils import *

"""
This is a preprocessing script to annotate anndata loaded from 
10x Genomics with coordinate annotations.
Any features without annotation are removed.

Output
# <unique_index> | <idx_col=gene_ids> feature_types <genome> pseudobulk_counts #CHR START END
"""


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        prog="Copytyping preprocessing - annotate scATAC-seq AnnData",
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
        help="output directory by cell-ranger",
    )
    parser.add_argument(
        "--barcodes",
        required=False,
        type=str,
        help="barcodes.txt",
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
    celltype_file = args["celltype_file"]
    out_dir = args["out_dir"]

    os.makedirs(out_dir, exist_ok=True)

    ##################################################
    barcodes = read_barcodes(barcode_file)
    print(f"#barcodes={len(barcodes)}")

    ##################################################
    print(f"load anndata")
    adata: sc.AnnData = sc.read_10x_mtx(
        os.path.join(ranger_dir, "filtered_feature_bc_matrix"),
        gex_only=False,
        make_unique=True,
    )
    adata = adata[:, adata.var["feature_types"] == "Peaks"][barcodes, :].copy()

    print(f"raw AnnData")
    print(adata)
    print(adata.var.head(3))

    ##################################################
    # filter barcodes
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
    # filter no count peaks
    adata.var["pseudobulk_counts"] = (
        np.asarray(adata.X.sum(axis=0)).flatten().astype(np.int32)
    )
    adata = adata[:, adata.var["pseudobulk_counts"] > 0].copy()

    ##################################################
    id_col = None  # 10x
    if id_col is None:
        candidates = ["gene_id", "gene_ids", "geneid", "GeneID"]
        candidates = [
            c
            for c in ["gene_id", "gene_ids", "geneid", "GeneID"]
            if c in adata.var.columns
        ]
        for cand in candidates:
            if cand in adata.var.columns:
                id_col = cand
    assert not id_col is None
    print(f"10x anndata id_col={id_col}")

    ##################################################
    print("preprocess coordinates")
    adata.var["#CHR"] = adata.var.index.str.split(":").str[0].astype(str)
    adata.var["START"] = (
        adata.var.index.str.split(":").str[1].str.split("-").str[0].astype(int)
    )
    adata.var["END"] = (
        adata.var.index.str.split(":").str[1].str.split("-").str[1].astype(int)
    )

    assert np.all(adata.var["START"].notna()), "invalid adata"
    print(f"#peaks={adata.n_vars}")

    chs = sort_chroms(adata.var["#CHR"].unique().tolist())
    adata.var["#CHR"] = pd.Categorical(adata.var["#CHR"], categories=chs, ordered=True)
    adata.var["START"] = adata.var["START"].astype(int)
    adata.var["END"] = adata.var["END"].astype(int)

    assert adata.var_names.is_unique, "index is not unique!"
    sort_index = adata.var.sort_values(by=["#CHR", "START"]).index
    adata = adata[:, sort_index].copy()

    print(f"final processed scATAC-seq AnnData")
    print(adata)
    print(adata.var.head(3))

    ##################################################
    out_file = os.path.join(out_dir, f"{sample}.peak.h5ad")
    adata.write_h5ad(out_file, compression="gzip")
    print(f"final output={out_file}")
    sys.exit(0)
