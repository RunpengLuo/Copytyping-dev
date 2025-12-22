import os
import sys
import gzip
import pandas as pd
import numpy as np
import pysam

from collections import OrderedDict
import pyranges as pr

import subprocess
from io import StringIO


def get_ord2chr(ch="chr"):
    return [f"{ch}{i}" for i in range(1, 23)] + [f"{ch}X", f"{ch}Y"]


def get_chr2ord(ch):
    chr2ord = {}
    for i in range(1, 23):
        chr2ord[f"{ch}{i}"] = i
    chr2ord[f"{ch}X"] = 23
    chr2ord[f"{ch}Y"] = 24
    chr2ord[f"{ch}M"] = 25
    return chr2ord


def count_comment_lines(filename: str, comment_symbol="#"):
    num_header = 0
    if filename.endswith(".gz"):
        fd = gzip.open(filename, "rt")
    else:
        fd = open(filename, "r")
    for line in fd:
        if line.startswith(comment_symbol):
            num_header += 1
        else:
            break
    fd.close()
    return num_header


def sort_chroms(chromosomes: list):
    assert len(chromosomes) != 0
    ch = "chr" if str(chromosomes[0]).startswith("chr") else ""
    chr2ord = get_chr2ord(ch)
    return sorted(chromosomes, key=lambda x: chr2ord[x])


def read_cn_profile(seg_ucn: str):
    """
    get copy-number profile per clone from HATCHet seg file
    """
    seg_df = pd.read_table(seg_ucn)
    # all samples share same copy-number states for each segment, just different purity
    samples = seg_df["SAMPLE"].unique().tolist()
    groups_sample = seg_df.groupby("SAMPLE", sort=False)
    seg_df = groups_sample.get_group(samples[0])

    n_clones = sum(1 for c in seg_df.columns if str.startswith(c, "cn_clone")) + 1
    clones = ["normal"] + [f"clone{i}" for i in range(1, n_clones)]
    chs = sort_chroms(seg_df["#CHR"].unique().tolist())
    seg_df["#CHR"] = pd.Categorical(seg_df["#CHR"], categories=chs, ordered=True)
    seg_df.sort_values(by=["#CHR", "START"], inplace=True, ignore_index=True)

    groups_ch = seg_df.groupby(by="#CHR", sort=False, observed=True)

    ch2segments = OrderedDict()  # ch -> position array
    ch2a_profile = OrderedDict()  # ch -> cn profile
    ch2b_profile = OrderedDict()  # ch -> cn profile
    for ch in chs:
        seg_df_ch = groups_ch.get_group(ch)
        num_segments_ch = len(seg_df_ch)
        ch2segments[ch] = seg_df_ch[["START", "END"]].to_numpy(dtype=np.int64)
        a_profile = np.zeros((num_segments_ch, n_clones), dtype=np.int8)
        b_profile = np.zeros((num_segments_ch, n_clones), dtype=np.int8)
        for j, clone in enumerate(clones):
            a_profile[:, j] = seg_df_ch.loc[:, f"cn_{clone}"].apply(
                func=lambda c: int(c.split("|")[0])
            )
            b_profile[:, j] = seg_df_ch.loc[:, f"cn_{clone}"].apply(
                func=lambda c: int(c.split("|")[1])
            )
        ch2a_profile[ch] = a_profile
        ch2b_profile[ch] = b_profile
    return chs, clones, ch2segments, ch2a_profile, ch2b_profile


# def get_cn_probability(
#     chs: list, ch2a_profile: dict, ch2b_profile: dict, laplace_alpha=0.01
# ):
#     """
#     compute laplace-smoothed copy-number probability
#     """
#     ch2probs = {}
#     ch2masks = {}
#     for ch in chs:
#         a_profile = ch2a_profile[ch]
#         b_profile = ch2b_profile[ch]
#         c_profile = a_profile + b_profile
#         cna_probabilities = np.divide(
#             b_profile + laplace_alpha,
#             c_profile + laplace_alpha * 2,
#             out=np.zeros_like(c_profile, dtype=np.float32),
#             where=(c_profile != 0),
#         )
#         ch2probs[ch] = cna_probabilities
#         # mask a segment if all clones are copy-neutral
#         ch2masks[ch] = np.all(a_profile == b_profile, axis=1)
#     return ch2probs, ch2masks


def read_barcodes(bc_file: str):
    barcodes = []
    with open(bc_file, "r") as fd:
        for line in fd:
            barcodes.append(line.strip().split("\t")[0])
        fd.close()
    return barcodes


def read_VCF(vcf_file: str, phased=False):
    """
    load vcf file as dataframe.
    If phased, parse GT[0] as USEREF, check PS
    """
    fields = "%CHROM\t%POS"
    names = ["#CHR", "POS"]
    format_tags = []
    if phased:
        vcf = pysam.VariantFile(vcf_file)
        format_tags.extend(["%GT"])
        names.extend(["GT"])
        if "PS" in vcf.header.formats:
            format_tags.extend(["%PS"])
            names.extend(["PS"])
        vcf.close()
        fields = fields + "\t[" + "\t".join(format_tags) + "]"
    fields += "\n"
    cmd = ["bcftools", "query", "-f", fields, vcf_file]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    snps = pd.read_csv(StringIO(result.stdout), sep="\t", header=None, names=names)
    assert not snps.duplicated().any(), f"{vcf_file} has duplicated rows"
    assert not snps.duplicated(subset=["#CHR", "POS"]).any(), (
        f"{vcf_file} has duplicated rows"
    )
    if phased:
        # Drop entries without phasing output
        if "PS" not in snps.columns:
            snps.loc[:, "PS"] = 1
        snps = snps[(~snps["GT"].isna()) & snps["GT"].str.contains(r"\|", na=False)]
        snps["GT"] = snps["GT"].apply(func=lambda v: v[0])
        snps.loc[:, "GT"] = snps["GT"].astype("Int8")
        snps.loc[~snps["GT"].isna(), "GT"] = (
            1 - snps.loc[~snps["GT"].isna(), "GT"]
        )  # USEREF
        snps.loc[:, "PS"] = snps["PS"].astype("Int64")
    snps = snps.reset_index(drop=True)
    return snps


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


def subset_baf(
    baf_df: pd.DataFrame, ch: str, start: int, end: int, is_last_block=False
):
    if ch != None:
        baf_ch = baf_df[baf_df["#CHR"] == ch]
    else:
        baf_ch = baf_df
    if baf_ch.index.name == "POS":
        pos = baf_ch.index
    else:
        pos = baf_ch["POS"]
    if is_last_block:
        return baf_ch[(pos >= start) & (pos <= end)]
    else:
        return baf_ch[(pos >= start) & (pos < end)]


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


def read_bbc_ucn_file(bbc_ucn_file: str):
    bbcs_df = pd.read_table(bbc_ucn_file, sep="\t")
    chs = sort_chroms(bbcs_df["#CHR"].unique().tolist())
    bbcs_df["#CHR"] = pd.Categorical(bbcs_df["#CHR"], categories=chs, ordered=True)
    bbcs_df.sort_values(by=["#CHR", "START"], inplace=True, ignore_index=True)
    # clones = [cname[3:] for cname in bbcs_df.columns if cname.startswith("cn_")]
    # bbcs_df["CNP"] = bbcs_df.apply(
    #     func=lambda r: ";".join(r[f"cn_{c}"] for c in clones), axis=1
    # )
    return bbcs_df


def BBC_segmentation(bbcs_df: pd.DataFrame):
    assert len(bbcs_df["SAMPLE"].unique()) == 1, ">1 samples"
    # segment BBC by chromosome and cluster ID
    group_name_to_indices = bbcs_df.groupby(
        (
            (bbcs_df["#CHR"] != bbcs_df["#CHR"].shift())
            | (bbcs_df["start"] != bbcs_df["end"].shift())
            | (bbcs_df["CLUSTER"] != bbcs_df["CLUSTER"].shift())
        ).cumsum(),
        # cumulative sum increments whenever a True is encountered, thus creating a series of monotonically
        # increasing values we can use as segment numbers
        sort=False,
    ).indices
    for group_name, indices in group_name_to_indices.items():
        bbcs_df.loc[indices, "segment"] = group_name

    aggregation_rules = {
        "#CHR": "first",
        "SAMPLE": "first",
        "start": "min",
        "end": "max",
        "#SNPS": "sum",
        "BAF": "mean",
        "RD": "mean",
        "COV": "mean",
        "ALPHA": "sum",
        "BETA": "sum",
        "CLUSTER": "first",
        "CNP": "first",
    }
    bbcs_df = bbcs_df.groupby(["segment", "SAMPLE"]).agg(aggregation_rules)
    return bbcs_df


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


def mark_co_type(assign_df: pd.DataFrame, a_col="simp_type", b_col="Decision"):
    """
    add a column to indicate consistency between assignments and `solution`
    """

    def co_type(d: str, t: str):
        if t == "tumor" and (d.lower().startswith("clone") or d.lower() == "tumor"):
            return "tumor"
        if t == "normal" and d == "normal":
            return "normal"
        return f"{t}_{d}"

    assign_df.loc[:, "co_type"] = assign_df.apply(
        func=lambda r: co_type(r[b_col], r[a_col]), axis=1
    )
    return assign_df


def index_states(chs: list, ch2segments: dict, ch2a_profile: dict, ch2b_profile: dict):
    state2id = {}
    id2state = []
    ch2sids = {}
    pos2states = {}
    sid = 0
    for ch in chs:
        ch2sids[ch] = []
        for i in range(len(ch2a_profile[ch])):
            aa = ch2a_profile[ch][i]
            bb = ch2b_profile[ch][i]
            [seg_s, seg_t] = ch2segments[ch][i]
            state = ";".join(f"{aa[j]}|{bb[j]}" for j in range(len(aa)))
            if state not in state2id:
                state2id[state] = sid
                id2state.append(state)
                sid += 1
            ch2sids[ch].append(state2id[state])
            pos2states[(ch, seg_s, seg_t)] = state
    return state2id, id2state, ch2sids, pos2states


def read_hair_file(hair_file: str, nsnps: int):
    """
    read raw hair file from HapCUT2
    hair=one read per row, list of supported alleles
    n=#SNPs
    hair array: shape (n, 4), first row is dummy.
    """
    hairs = np.zeros((nsnps, 4), dtype=np.int16)
    mapper = {"00": 0, "01": 1, "10": 2, "11": 3}
    mapper_rev = {"00": 3, "01": 2, "10": 1, "11": 0}
    ctr = 0
    with open(hair_file, "r") as hair_fd:
        for line in hair_fd:
            ctr += 1
            if ctr % 100000 == 0:
                print(ctr)
            fields = line.strip().split(" ")[:-1]
            nblocks = int(fields[0])
            for i in range(nblocks):
                var_start = int(fields[2 + (i * 2)])  # 1-based
                phases = fields[2 + (i * 2 + 1)]
                var_end = var_start + len(phases) - 1

                pvar_idx = [mapper[phases[j : j + 2]] for j in range(len(phases) - 1)]
                pvar_idx_rev = [
                    mapper_rev[phases[j : j + 2]] for j in range(len(phases) - 1)
                ]
                hairs[np.arange(var_start, var_end), pvar_idx] += 1
                hairs[np.arange(var_start, var_end), pvar_idx_rev] += 1
    print(f"total processed {ctr} reads")
    return hairs


def load_hairs(hair_file: str, smoothing=True, alpha=1):
    """
    load (nsnp, 4) hair tsv file, +alpha smoothing
    """
    hairs = None
    with gzip.open(hair_file, "rt") as f:
        hairs = np.loadtxt(f, delimiter="\t", dtype=np.int16)
    hairs = hairs.astype(np.float32)
    if smoothing:
        hairs[:, :] += alpha
    return hairs


def ordered_merge(ord_df, other_df, on=["#CHR", "POS"], how="left"):
    ord_df["_order"] = range(len(ord_df))
    ord_df = pd.merge(left=ord_df, right=other_df, on=on, how=how)
    ord_df = ord_df.sort_values("_order")
    ord_df = ord_df.drop(columns="_order")
    return ord_df


def annotate_snps_seg_idx(segs: pd.DataFrame, snps: pd.DataFrame, seg_id="SEG_IDX"):
    snps[seg_id] = pd.NA
    # annotate CNV profiles onto SNPs
    for ch in segs["#CHR"].unique():
        snps_ch = snps[snps["#CHR"] == ch]
        for seg_idx, seg in segs[segs["#CHR"] == ch].iterrows():
            seg_s, seg_t = seg["START"], seg["END"]
            seg_pos0 = snps_ch["POS0"]
            snp_seg = snps_ch[(seg_pos0 >= seg_s) & (seg_pos0 < seg_t)]
            snps.loc[snp_seg.index, seg_id] = seg_idx
    return snps


def adaptive_co_binning(
    parent_df: pd.DataFrame,
    child_df: pd.DataFrame,
    block_colname: str,
    feat_colnames: list,
    min_counts: np.ndarray,
    s_block_id=0,
):
    """
    multi-dimensionial adaptive binning
    """
    nfeatures = len(child_df)
    feat_counts = child_df[feat_colnames]
    feat_idxs = child_df.index
    block_ids = np.zeros(nfeatures, dtype=np.int64)

    block_id = s_block_id
    prev_start = 0
    prev_counts = feat_counts.loc[feat_idxs[0]]
    for i in range(1, nfeatures):
        idx = feat_idxs[i]
        if np.any(prev_counts < min_counts):
            # extend feature block
            prev_counts += feat_counts.loc[idx]
        else:
            block_ids[prev_start:i] = block_id
            block_id += 1
            prev_start = i
            prev_counts = feat_counts.loc[idx]
    # fill last block if any
    block_ids[prev_start:] = max(block_id - 1, s_block_id)
    parent_df.loc[feat_idxs, block_colname] = block_ids

    # if only a partial block is found, block_id is not incremented in the loop
    # we do it here in this case
    next_block_id = max(block_id, s_block_id + 1)
    return next_block_id


def adaptive_binning(
    parent_df: pd.DataFrame,
    child_df: pd.DataFrame,
    block_colname: str,
    feat_colname: str,
    min_count: int,
    s_block_id=0,
):
    """
    multi-dimensionial adaptive binning
    """
    nfeatures = len(child_df)
    feat_counts = child_df[feat_colname]
    feat_idxs = child_df.index
    block_ids = np.zeros(nfeatures, dtype=np.int64)

    block_id = s_block_id
    prev_start = 0
    acc_count = feat_counts.loc[feat_idxs[0]]
    for i in range(1, nfeatures):
        idx = feat_idxs[i]
        if acc_count < min_count:
            # extend feature block
            acc_count += feat_counts.loc[idx]
        else:
            block_ids[prev_start:i] = block_id
            block_id += 1
            prev_start = i
            acc_count = feat_counts.loc[idx]
    # fill last block if any
    block_ids[prev_start:] = max(block_id - 1, s_block_id)
    parent_df.loc[feat_idxs, block_colname] = block_ids

    # if only a partial block is found, block_id is not incremented in the loop
    # we do it here in this case
    next_block_id = max(block_id, s_block_id + 1)
    return next_block_id


def load_annotation_file_bed(ann_file: str):
    ann = pd.read_table(
        ann_file,
        sep="\t",
        header=None,
        usecols=range(4),
        names=["#CHR", "START", "END", "gene_ids"],
    )
    return ann


def load_annotation_file_gtf(ann_file: str):
    gtf = pd.read_csv(
        ann_file,
        sep="\t",
        comment="#",
        header=None,
        names=[
            "#CHR",
            "source",
            "feature_types",
            "START",
            "END",
            "score",
            "strand",
            "frame",
            "attribute",
        ],
    )
    # keep only gene records
    gtf = gtf[gtf["feature_types"] == "gene"].copy()

    # extract key attributes
    gtf["gene_ids"] = gtf["attribute"].str.extract('gene_id "([^"]+)"')
    gtf["gene_name"] = gtf["attribute"].str.extract('gene_name "([^"]+)"')
    gtf["gene_type"] = gtf["attribute"].str.extract('gene_type "([^"]+)"')
    ann = gtf[
        ["#CHR", "START", "END", "gene_ids", "gene_name", "gene_type"]
    ].drop_duplicates()

    ann["gene_id_base"] = ann["gene_ids"].str.replace(r"\.\d+$", "", regex=True)

    return ann


import pyranges as pr


def assign_largest_overlap(
    qry: pd.DataFrame, ref: pd.DataFrame, qry_id="SUPER_VAR_IDX", ref_id="HB"
) -> pd.DataFrame:
    """
    for each row in qry, assign the ID of the overlapping interval in ref
    that has largest overlap length
    """

    qry_gr = pr.PyRanges(
        qry.rename(columns={"#CHR": "Chromosome", "START": "Start", "END": "End"})
    )
    ref_gr = pr.PyRanges(
        ref.rename(columns={"#CHR": "Chromosome", "START": "Start", "END": "End"})
    )

    joined = qry_gr.join(ref_gr, suffix="_REF").as_df()
    joined["overlap_len"] = (
        np.minimum(joined["End"], joined["End_REF"])
        - np.maximum(joined["Start"], joined["Start_REF"])
    ).clip(lower=0)

    best = joined.loc[joined.groupby(joined.index)["overlap_len"].idxmax()].copy()

    best = best[[qry_id, ref_id, "overlap_len"]]
    qry = qry.merge(best, on=[qry_id], how="left", sort=False)
    max_ovlp_idx = qry.groupby(by=qry_id, sort=False)["overlap_len"].apply(
        lambda s: s.idxmax() if s.notna().any() else s.index[0]
    )
    qry_out = qry.loc[max_ovlp_idx].sort_values(by=qry_id).reset_index(drop=True)
    qry_out = qry_out.drop(columns="overlap_len")
    return qry_out


def assign_pos_to_range(
    qry: pd.DataFrame,
    ref: pd.DataFrame,
    ref_id="SUPER_VAR_IDX",
    pos_col="POS0",
    nodup=True,
):
    qry_pr = pr.PyRanges(
        chromosomes=qry["#CHR"], starts=qry[pos_col], ends=qry[pos_col] + 1
    )
    qry_pr = qry_pr.insert(pd.Series(data=qry.index.to_numpy(), name="qry_index"))
    ref_pr = pr.PyRanges(
        chromosomes=ref["#CHR"],
        starts=ref["START"],
        ends=ref["END"],
    )
    ref_pr = ref_pr.insert(ref[ref_id])

    if nodup:  # qry assigned to at most one ref.
        joined = qry_pr.join(ref_pr)
        qry[ref_id] = pd.NA
        qry.loc[joined.df["qry_index"], ref_id] = joined.df[ref_id].to_numpy()
        return qry
    else:
        hits = (
            qry_pr.join(ref_pr)
            .df[["Chromosome", "Start", "qry_index", ref_id]]
            .rename(columns={"Chromosome": "#CHR", "Start": "POS0"})
        )
        hits["POS"] = hits["POS0"] + 1
        return hits
