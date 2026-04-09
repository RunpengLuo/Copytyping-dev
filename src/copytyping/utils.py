import os
import pandas as pd
import numpy as np
import logging

from collections import OrderedDict

import subprocess
from io import StringIO

ALL_PLATFORMS = ["single_cell", "spatial"]
SPATIAL_PLATFORMS = {"spatial"}

TUMOR_LABELS = {"Tumor_cell", "tumor", "Tumor"}
TUMOR_PREFIXES = ("tumor", "clone", "clone_")

INVALID_LABELS = {"Doublet", "doublet", "Unknown", "NA"}
NA_CELLTYPE = {"Unknown", "NA"}


def is_tumor_label(label: str) -> bool:
    """Check if a label indicates a tumor cell/spot (tumor*, clone*, etc.)."""
    if label in TUMOR_LABELS:
        return True
    return label.lower().startswith(TUMOR_PREFIXES)


def is_normal_label(label: str) -> bool:
    """Check if a label indicates a normal cell/spot (not tumor, not invalid)."""
    return not is_tumor_label(label) and label not in INVALID_LABELS


def get_chr2ord(ch):
    chr2ord = {}
    for i in range(1, 23):
        chr2ord[f"{ch}{i}"] = i
    chr2ord[f"{ch}X"] = 23
    chr2ord[f"{ch}Y"] = 24
    chr2ord[f"{ch}M"] = 25
    return chr2ord


def sort_chroms(chromosomes: list):
    assert len(chromosomes) != 0
    ch = "chr" if str(chromosomes[0]).startswith("chr") else ""
    chr2ord = get_chr2ord(ch)
    return sorted(chromosomes, key=lambda x: chr2ord[x])


def read_barcodes(bc_file: str):
    barcodes = (
        pd.read_table(bc_file, sep="\t", header=None, dtype=str).iloc[:, 0].tolist()
    )
    return barcodes


def read_VCF_cellsnp_err_header(vcf_file: str):
    """cellsnp-lite has issue with its header"""
    fields = "%CHROM\t%POS\t%INFO"
    names = ["#CHR", "POS", "INFO"]
    cmd = ["bcftools", "query", "-f", fields, vcf_file]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    snps = pd.read_csv(StringIO(result.stdout), sep="\t", header=None, names=names)

    def extract_info_field(info_str, key):
        for field in info_str.split(";"):
            if field.startswith(f"{key}="):
                return int(field.split("=")[1])
        return pd.NA

    snps["DP"] = snps["INFO"].apply(lambda x: extract_info_field(x, "DP"))
    snps = snps.drop(columns="INFO")
    return snps


def get_chr_sizes(sz_file: str):
    chr_sizes = OrderedDict()
    with open(sz_file, "r") as rfd:
        for line in rfd.readlines():
            ch, sizes = line.strip().split()
            chr_sizes[ch] = int(sizes)
        rfd.close()
    return chr_sizes


def read_baf_file(baf_file: str):
    baf_df = pd.read_table(
        baf_file,
        names=["#CHR", "POS", "SAMPLE", "REF", "ALT"],
        dtype={
            "#CHR": object,
            "POS": np.uint32,
            "SAMPLE": object,
            "REF": np.uint32,
            "ALT": np.uint32,
        },
    )
    return sort_df_chr(baf_df)


def sort_df_chr(df: pd.DataFrame, ch="#CHR", pos="POS"):
    chs = sort_chroms(df[ch].unique().tolist())
    df[ch] = pd.Categorical(df[ch], categories=chs, ordered=True)
    df.sort_values(by=[ch, pos], inplace=True, ignore_index=True)
    return df


def read_seg_ucn_file(seg_ucn_file: str):
    segs_df = pd.read_table(seg_ucn_file, sep="\t")
    segs_df = sort_df_chr(segs_df, pos="START")

    n_clones = len([cname for cname in segs_df.columns if cname.startswith("cn_")])
    clones = ["normal"] + [f"clone{c}" for c in range(1, n_clones)]
    clone_props = segs_df[[f"u_{clone}" for clone in clones]].iloc[0].tolist()
    segs_df.loc[:, "CNP"] = segs_df.apply(
        func=lambda r: ";".join(r[f"cn_{c}"] for c in clones), axis=1
    )
    segs_df["PROPS"] = ";".join([str(p) for p in clone_props])
    return segs_df, clones, clone_props


def read_celltypes(celltype_file: str):
    # cell_id cell_types final_type
    celltypes = pd.read_table(celltype_file)
    celltypes = celltypes.rename(
        columns={"cell_id": "BARCODE", "cell_types": "cell_type"}
    )
    celltypes["BARCODE"] = celltypes["BARCODE"].astype(str)

    if "met_subcluster" in celltypes.columns.tolist():
        print("use column met_subcluster as final_type")
        celltypes["final_type"] = celltypes["met_subcluster"]

    if "final_type" not in celltypes.columns.tolist():
        assert "cell_type" in celltypes.columns.tolist(), (
            "cell_type column does not exist"
        )
        print("use column cell_type as final_type")
        celltypes["final_type"] = celltypes["cell_type"]
    return celltypes


def read_whitelist_segments(bed_file: str):
    wl_fragments = pd.read_table(
        bed_file,
        sep="\t",
        header=None,
        names=["#CHR", "START", "END", "NAME"],
    )
    return wl_fragments


def setup_logging(args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def add_file_logging(out_dir: str, command: str = "copytyping") -> None:
    """Attach a FileHandler to the root logger so logs are also written to *out_dir/<command>.log*."""
    os.makedirs(out_dir, exist_ok=True)
    level = (
        logging.root.level if logging.root.level != logging.WARNING else logging.INFO
    )
    fh = logging.FileHandler(os.path.join(out_dir, f"{command}.log"), mode="w")
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.root.addHandler(fh)
    if logging.root.level > level:
        logging.root.setLevel(level)
