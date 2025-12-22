import os
import sys
import subprocess

import numpy as np
import pandas as pd


def filter_vcf2bed(in_vcf_file: str, seg_bed_file: str, out_file: str, log_file: str):
    """
    only keep variants resides in seg_bed_file
    convert to BED format
    """
    if not os.path.exists(f"{in_vcf_file}.tbi"):
        subprocess.run(["tabix", in_vcf_file], check=True)

    view_proc = subprocess.run(
        ["bcftools", "view", "-R", seg_bed_file, in_vcf_file],
        stdout=subprocess.PIPE,
        stderr=open(log_file, "w"),
        check=True,
    )
    subprocess.run(
        ["bcftools", "query", "-f", "%CHROM\t%POS0\t%POS\n"],
        input=view_proc.stdout,
        stdout=open(out_file, "w"),
        check=True,
    )
    return


def sort_bed(in_bed_file: str, out_file):
    subprocess.run(
        ["sort", "-k1,1V", "-k2,2n", in_bed_file],
        stdout=open(out_file, "w"),
        check=True,
    )
    return


def run_bedtools_closest(
    seg_bed_file: str,
    in_vcf_file: str,
    in_bed_file: str,
    out_file: str,
    tmp_dir: str,
    load_df=False,
    usecols=[1, 3, 4, 5, 6, 7, 8],
    names=["POS", "#CHR", "START", "END", "feature_id", "mat_index", "dist"],
):
    print("run bedtools closest")
    tmp_bed_file = os.path.join(tmp_dir, "variants.filtered.bed")
    tmp_log_file = os.path.join(tmp_dir, "bcftools_filter.log")
    filter_vcf2bed(in_vcf_file, seg_bed_file, tmp_bed_file, tmp_log_file)

    tmp_sbed_file = os.path.join(tmp_dir, "variants.sorted.bed")
    sort_bed(tmp_bed_file, tmp_sbed_file)

    tmp_gbed_file = os.path.join(tmp_dir, "features.sorted.bed")
    sort_bed(in_bed_file, tmp_gbed_file)

    dist_proc = subprocess.run(
        [
            "bedtools",
            "closest",
            "-a",
            tmp_sbed_file,
            "-b",
            tmp_gbed_file,
            "-D",
            "ref",
            "-t",
            "all",
        ],
        stdout=open(out_file, "w"),
        check=True,
    )

    os.remove(tmp_bed_file)
    os.remove(tmp_log_file)
    os.remove(tmp_sbed_file)
    os.remove(tmp_gbed_file)

    if load_df:
        out_df = pd.read_table(out_file, sep="\t", header=None, usecols=usecols)
        out_df.columns = names
        return out_df
    return None


def run_bedtools_merge_clamp(
    seg_bed_file: str,
    in_vcf_file: str,
    out_file: str,
    tmp_dir: str,
    max_dist=500,
):
    """
    merge adjacent features based on <max_dist>
    set max_dist=0 for overlapping features only
    output
    report #CHR\tSTART\tEND\t#SNP
    """
    print("run bedtools merge")
    tmp_bed_file = os.path.join(tmp_dir, "variants.filtered.bed")
    tmp_log_file = os.path.join(tmp_dir, "bcftools_filter.log")
    filter_vcf2bed(in_vcf_file, seg_bed_file, tmp_bed_file, tmp_log_file)

    tmp_sbed_file = os.path.join(tmp_dir, "variants.sorted.bed")
    sort_bed(tmp_bed_file, tmp_sbed_file)

    subprocess.run(
        [
            "bedtools",
            "merge",
            "-i",
            tmp_sbed_file,
            "-d",
            str(max_dist),
            "-c",
            str(1),
            "-o",
            "count",
        ],
        stdout=open(out_file, "w"),
        check=True,
    )
    os.remove(tmp_bed_file)
    os.remove(tmp_log_file)

    out_df = pd.read_table(out_file, sep="\t", names=["#CHR", "START", "END", "#SNP"])
    return out_df


def run_bedtools_cluster(
    in_bed_file: str,
    out_file: str,
    tmp_dir: str,
    max_dist=0,
    load_df=False,
    usecols=[1, 2, 3, 4, 5, 6, 7, 8],
    names=[
        "#CHR",
        "START",
        "END",
        "#SNP",
        "DP",
        "feature_id",
        "mat_index",
        "cluster_id",
    ],
):
    """
    assign features with cluster id based on <max_dist>
    set max_dist=0 for overlapping features only
    input
        #CHR      START        END   #SNP DP  feature_id mat_index
    """
    print("run bedtools cluster")

    subprocess.run(
        [
            "bedtools",
            "cluster",
            "-i",
            in_bed_file,
            "-d",
            str(max_dist),
        ],
        stdout=open(out_file, "w"),
        check=True,
    )

    if load_df:
        out_df = pd.read_table(out_file, sep="\t", header=None, usecols=usecols)
        out_df.columns = names
        return out_df
    return None


def run_bedtools_merge(
    in_bed_file: str,
    out_file: str,
    tmp_dir: str,
    max_dist=0,
    load_df=False,
    merge_cols=[4, 5],
    usecols=[1, 2, 3, 4, 5],
    names=["#CHR", "START", "END", "feature_id", "mat_index"],
):
    """
    merge adjacent overlapping features
    """
    print("run bedtools merge")
    tmp_sbed_file = os.path.join(tmp_dir, "variants.sorted.bed")
    sort_bed(in_bed_file, tmp_sbed_file)

    subprocess.run(
        [
            "bedtools",
            "merge",
            "-i",
            tmp_sbed_file,
            "-d",
            str(max_dist),
            "-c",
            ",".join(str(c) for c in merge_cols),
            "-delim",
            ",",
            "-o",
            "collapse",
        ],
        stdout=open(out_file, "w"),
        check=True,
    )
    os.remove(tmp_sbed_file)

    if load_df:
        out_df = pd.read_table(out_file, sep="\t", header=None, usecols=usecols)
        out_df.columns = names
        return out_df
    return None


def run_bedtools_makewindows(
    in_bed_file: str,
    out_file: str,
    window_size=1e5,
):
    print("run bedtools makewindows")
    subprocess.run(
        [
            "bedtools",
            "makewindows",
            "-b",
            in_bed_file,
            "-w",
            str(window_size),
            "-i",
            "srcwinnum",
        ],
        stdout=open(out_file, "w"),
        check=True,
    )
    return


def run_bedtools_coverage(
    in_bed_file: str,
    in_win_file: str,
    genome_file: str,
    out_file: str,
    tmp_dir: str,
    load_df=False,
    ncols=5,
    names=["#CHR", "START", "END", "HB_WIN", "count"],
):
    """compute per-window counts"""
    print("run bedtools coverage")

    subprocess.run(
        [
            "bedtools",
            "coverage",
            "-a",
            in_win_file,
            "-b",
            in_bed_file,
            "-g",
            genome_file,
            "-counts",
            "-sorted",
        ],
        stdout=open(out_file, "w"),
        check=True,
    )

    if load_df:
        out_df = pd.read_table(out_file, sep="\t", header=None, usecols=range(ncols))
        out_df.columns = names
        return out_df
    return None

    return
