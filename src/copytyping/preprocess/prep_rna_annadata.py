import os
import sys
import argparse

import numpy as np
import pyranges as pr
import scanpy as sc
import squidpy as sq

from copytyping.utils import *

"""
This is a preprocessing script to annotate anndata loaded from 
10x Genomics with coordinate annotations.
Any features without annotation are removed.

Annotation file
* T2T-CHM13v2.0: 
    * ensembl https://ftp.ebi.ac.uk/pub/ensemblorganisms//Homo_sapiens/GCA_009914755.4/ensembl/geneset/2022_07/genes.gtf.gz
    * ncbi https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/009/914/755/GCF_009914755.1_T2T-CHM13v2.0/GCF_009914755.1_T2T-CHM13v2.0_genomic.gtf.gz
# GRCh38:
    * ensembl https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.basic.annotation.gtf.gz

Output
# <unique_index> | <idx_col=gene_ids> feature_types <genome> pseudobulk_counts #CHR START END
"""


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        prog="Copytyping preprocessing - annotate AnnData",
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
        "--library_id",
        required=False,
        type=str,
        help="library_id, visium data for concat multiple slice",
    )
    parser.add_argument(
        "--modality",
        required=True,
        type=str,
        default="rna",
        choices=["rna", "visium"],
        help="rna, visium",
    )
    parser.add_argument(
        "--ranger_dir",
        required=True,
        type=str,
        help="output directory by cell-ranger or space-ranger",
    )
    parser.add_argument(
        "--annotation_file", required=True, type=str, help="annotation gtf/bed file"
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
    modality = args["modality"]
    ranger_dir = args["ranger_dir"]
    ann_file = args["annotation_file"]
    barcode_file = args["barcodes"]
    celltype_file = args["celltype_file"]
    out_dir = args["out_dir"]

    assert any(ann_file.endswith(sfx) for sfx in [".bed.gz", ".bed", ".gtf", ".gtf.gz"])
    os.makedirs(out_dir, exist_ok=True)

    ##################################################
    barcodes = read_barcodes(barcode_file)
    print(f"#barcodes={len(barcodes)}")

    ##################################################
    print(f"load anndata")
    if modality in ["rna"]:
        # only load gene features
        adata: sc.AnnData = sc.read_10x_mtx(
            os.path.join(ranger_dir, "filtered_feature_bc_matrix"),
            gex_only=True,
            make_unique=True,
        )
    elif modality in ["visium"]:
        adata: sc.AnnData = sq.read.visium(ranger_dir, load_images=True)
        adata.var_names_make_unique()
    else:
        raise ValueError()

    print(f"raw AnnData")
    print(adata)
    print(adata.var.head(3))

    ##################################################
    # filter barcodes
    adata = adata[barcodes, :].copy()
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
    # filter no count genes
    adata.var["pseudobulk_counts"] = np.asarray(adata.X.sum(axis=0)).flatten()
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
    print(f"load annotation file={os.path.relpath(ann_file)}")
    if "gtf" in ann_file:
        gr = pr.read_gtf(ann_file, rename_attr=True)
        genes = (
            gr.df.query("Feature == 'gene'")[["Chromosome", "Start", "End", "gene_id"]]
            .drop_duplicates("gene_id", keep="first")
            .rename(
                columns={
                    "gene_id": id_col,
                    "Chromosome": "#CHR",
                    "Start": "START",
                    "End": "END",
                }
            )
        )
        genes[id_col] = genes[id_col].str.replace(
            r"\.\d+$", "", regex=True
        )  # discard .5 suffix
    elif "bed" in ann_file:
        gr = pr.read_bed(ann_file, as_df=True)
        genes = (
            gr[["Chromosome", "Start", "End", "Name"]]
            .drop_duplicates("Name", keep="first")
            .rename(
                columns={
                    "Name": id_col,
                    "Chromosome": "#CHR",
                    "Start": "START",
                    "End": "END",
                }
            )
        )
    else:
        raise ValueError()
    print(f"Loaded {len(genes)} unique genes from GTF.")

    ##################################################
    var_coords = adata.var.merge(
        genes, how="left", on="gene_ids", validate="m:1", sort=False
    )
    var_coords["index"] = adata.var.index.to_numpy()
    var_coords = var_coords.set_index(keys="index", drop=True)
    na_genes = var_coords["START"].isna().to_numpy()
    print(f"#features doesn't have annotation={na_genes.sum()}/{len(var_coords)}")
    var_coords = var_coords.loc[~na_genes, :]

    adata = adata[:, ~na_genes].copy()
    adata.var = var_coords

    print(f"#features (ignore na positions)={adata.shape[1]}")

    adata.var["#CHR"] = adata.var["#CHR"].astype(str)
    chs = sort_chroms(adata.var["#CHR"].unique().tolist())
    adata.var["#CHR"] = pd.Categorical(adata.var["#CHR"], categories=chs, ordered=True)
    adata.var["START"] = adata.var["START"].astype(int)
    adata.var["END"] = adata.var["END"].astype(int)

    assert adata.var_names.is_unique, "index is not unique!"
    sort_index = adata.var.sort_values(by=["#CHR", "START"]).index
    adata = adata[:, sort_index].copy()

    print(f"final processed AnnData")
    print(adata)
    print(adata.var.head(3))

    ##################################################
    out_file = os.path.join(out_dir, f"{sample}.{modality}.h5ad")
    adata.write_h5ad(out_file, compression="gzip")
    print(f"final output={out_file}")
    sys.exit(0)
