import os
import sys

import numpy as np
import pandas as pd

# import scanpy as sc
# from scanpy import AnnData
# import squidpy as sq
# import snapatac2 as snap

from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy import sparse

from copytyping.utils import *


##################################################
# preprocess IOs
##################################################


##################################################
def load_sample_file(sample_file: str):
    """
    one sample file, multiple trials.
    SAMPLE TRIAL_ID MODALITY DATA_TYPE paths
    """
    sample_df = pd.read_table(sample_file, sep="\t")

    return


def read_barcodes_celltype_one_replicate(
    barcode_file: str,
    modality: str,
    reference_file=None,
    replicate_id="",
    ref_label="cell_type",
):
    barcodes = read_barcodes(barcode_file)
    print(f"#barcodes={len(barcodes)}")

    barcodes_df = pd.DataFrame(data={"BARCODE": barcodes})
    if replicate_id != "":
        barcodes_df["REP_ID"] = replicate_id
        barcodes_df["BARCODE_FULL"] = (
            barcodes_df["BARCODE"].astype(str).str.rstrip() + f"_{replicate_id}"
        )

    if not reference_file is None:
        if modality == "VISIUM":
            barcodes_df = load_visium_path_annotation(
                reference_file, label=ref_label, anns=barcodes_df
            )
        else:
            celltypes = read_celltypes(reference_file)
            barcodes_df = pd.merge(
                left=barcodes_df, right=celltypes[["BARCODE", ref_label]], how="left"
            )
            barcodes_df[ref_label] = (
                barcodes_df[ref_label].fillna(value="Unknown").astype(str)
            )
    return barcodes_df


def load_seg_ucn(seg_ucn: str, min_cnv_length=1e6):
    """load CNV profile. Filter any unconfident ones."""
    print(f"load cnv profile, min_cnv_length={min_cnv_length}")
    segs, clones, clone_props = read_seg_ucn_file(seg_ucn)
    segs["SEG_BLOCKSIZE"] = segs["END"] - segs["START"]
    cnv_mask = segs["SEG_BLOCKSIZE"] < min_cnv_length
    # 2. CNP must contain at least one 1|0 or 2|0 anywhere in the string
    # has_het = (
    #     segs["CNP"].str.contains("1|0", regex=False) |
    #     segs["CNP"].str.contains("2|0", regex=False) |
    #     segs["CNP"].str.contains("3|0", regex=False) |
    #     segs["CNP"].str.contains("2|1", regex=False)
    # )
    # cnv_mask |= ~has_het

    # hard_list = [
    #     "1|1;1|0;1|0",
    #     "1|1;2|0;2|0",
    #     "1|1;1|1;2|1",
    #     "1|1;2|0;1|1"
    # ]
    # cnv_mask |= (~segs["CNP"].isin(hard_list))

    no_het = segs["CNP"].str.contains("1|2", regex=False)
    cnv_mask |= no_het

    if np.sum(cnv_mask) > 0:
        print("Following CNP entries are dropped: ")
        print(segs.loc[cnv_mask, ["#CHR", "START", "END", "SEG_BLOCKSIZE", "CNP"]])
        segs = segs.loc[~cnv_mask, :].reset_index(drop=True)

    n_clones = len(clones)
    cnv_Aallele = np.zeros((len(segs), n_clones), dtype=np.int32)
    cnv_Ballele = np.zeros((len(segs), n_clones), dtype=np.int32)
    for i, clone in enumerate(clones):
        cnv_Aallele[:, i] = segs.apply(
            func=lambda r: int(r[f"cn_{clone}"].split("|")[0]), axis=1
        ).to_numpy()
        cnv_Ballele[:, i] = segs.apply(
            func=lambda r: int(r[f"cn_{clone}"].split("|")[1]), axis=1
        ).to_numpy()
    cnv_mixBAF = (cnv_Ballele @ clone_props) / (
        (cnv_Aallele + cnv_Ballele) @ clone_props
    )
    segs["BAF"] = cnv_mixBAF
    segs["imbalanced"] = mark_imbalanced(segs)
    return segs, cnv_Aallele, cnv_Ballele, cnv_mixBAF, clone_props


def mark_imbalanced(segs: pd.DataFrame):
    def is_imbalanced(seg):
        cnp = seg["CNP"].split(";")[1:]
        cna = [int(cn.split("|")[0]) for cn in cnp]
        cnb = [int(cn.split("|")[1]) for cn in cnp]
        imb = any(a != b for (a, b) in zip(cna, cnb))
        return imb

    return segs.apply(is_imbalanced, axis=1).to_numpy()


def mark_cna(segs: pd.DataFrame):
    def is_cna(seg):
        cnp = seg["CNP"].split(";")[1:]
        cna = [int(cn.split("|")[0]) for cn in cnp]
        cnb = [int(cn.split("|")[1]) for cn in cnp]
        return any((a != 1) or (b != 1) for (a, b) in zip(cna, cnb))

    return segs.apply(is_cna, axis=1).to_numpy()


def mark_aneuploid(segs: pd.DataFrame):
    def is_aneuploid(seg):
        cnp = seg["CNP"].split(";")[1:]
        cna = [int(cn.split("|")[0]) for cn in cnp]
        cnb = [int(cn.split("|")[1]) for cn in cnp]
        return any((a + b) != 2 for (a, b) in zip(cna, cnb))

    return segs.apply(is_aneuploid, axis=1).to_numpy()


# Load DNA bulk data
def load_snps_HATCHet_old(
    phased_snps: pd.DataFrame,
    hatchet_files: list,
):
    [tumor_1bed_file] = hatchet_files
    snp_info = read_baf_file(tumor_1bed_file)
    snp_info = pd.merge(
        left=snp_info, right=phased_snps, on=["#CHR", "POS"], how="left"
    )
    snp_info.dropna(subset=["GT"], inplace=True)
    # assume one tumor sample for now
    ref_counts = snp_info["REF"].to_numpy().astype(np.int32)
    alt_counts = snp_info["ALT"].to_numpy().astype(np.int32)
    allale_counts = ref_counts + alt_counts
    return snp_info, allale_counts, ref_counts, alt_counts


def load_snps_HATCHet_new(
    phased_snps: pd.DataFrame,
    hatchet_files: list,
):
    [allele_dir] = hatchet_files
    snp_ifile = os.path.join(allele_dir, "snp_info.tsv.gz")
    ref_mfile = os.path.join(allele_dir, "snp_matrix.ref.npz")
    alt_mfile = os.path.join(allele_dir, "snp_matrix.alt.npz")
    snp_info = pd.read_table(snp_ifile, sep="\t")
    snp_info = pd.merge(
        left=snp_info, right=phased_snps, on=["#CHR", "POS"], how="left"
    )
    assert np.all(~pd.isna(snp_info["GT"])), (
        "invalid input, only phased SNPs should present here"
    )
    ref_mat = np.load(ref_mfile)["mat"].astype(np.int32)
    alt_mat = np.load(alt_mfile)["mat"].astype(np.int32)
    ref_counts = ref_mat[:, 1]
    alt_counts = alt_mat[:, 1]
    allele_counts = ref_counts + alt_counts
    phases = snp_info["GT"].astype(np.int8).to_numpy()
    b_counts = (phases * ref_counts + (1 - phases) * alt_counts).astype(
        allele_counts.dtype
    )
    baf_vals = b_counts / allele_counts
    return snp_info, allele_counts, ref_counts, alt_counts, b_counts, baf_vals


def load_snps_pseudobulk(
    barcodes_per_replicate: dict,
    allele_infos: dict,
    modality: str,
    ref_label="path_label",
):
    assert modality == "VISIUM", "todo"
    # all replicates share same snp info TODO clean later
    snp_info = allele_infos[list(allele_infos.keys())[0]][0].copy(deep=True)
    allele_counts = np.zeros(len(snp_info), dtype=np.int32)
    ref_counts = np.zeros(len(snp_info), dtype=np.int32)
    alt_counts = np.zeros(len(snp_info), dtype=np.int32)
    for rep_id, barcodes_df in barcodes_per_replicate.items():
        _, allele_counts_pb, ref_counts_pb, alt_counts_pb = allele_infos[rep_id]
        if ref_label in barcodes_df:
            tumor_mask = (barcodes_df[ref_label] == "tumor").to_numpy()
            allele_counts += (
                np.array(allele_counts_pb[:, tumor_mask].sum(axis=1))
                .ravel()
                .astype(np.int32)
            )
            ref_counts += (
                np.array(ref_counts_pb[:, tumor_mask].sum(axis=1))
                .ravel()
                .astype(np.int32)
            )
            alt_counts += (
                np.array(alt_counts_pb[:, tumor_mask].sum(axis=1))
                .ravel()
                .astype(np.int32)
            )
        else:
            allele_counts += (
                np.array(allele_counts_pb.sum(axis=1)).ravel().astype(np.int32)
            )
            ref_counts += np.array(ref_counts_pb.sum(axis=1)).ravel().astype(np.int32)
            alt_counts += np.array(alt_counts_pb.sum(axis=1)).ravel().astype(np.int32)
    return snp_info, allele_counts, ref_counts, alt_counts


##################################################
# Load Allele count data
def load_cellsnp_files(
    cellsnp_dir: str,
    barcodes: list,
):
    print(f"load cell-snp files from {cellsnp_dir}")
    barcode_file = os.path.join(cellsnp_dir, "cellSNP.samples.tsv")
    vcf_file = os.path.join(cellsnp_dir, "cellSNP.base.vcf.gz")
    dp_file = os.path.join(cellsnp_dir, "cellSNP.tag.DP.mtx")
    ad_file = os.path.join(cellsnp_dir, "cellSNP.tag.AD.mtx")

    raw_barcodes = read_barcodes(barcode_file)  # assume no header
    barcode_indices = np.array([raw_barcodes.index(x) for x in barcodes])
    dp_mat: csr_matrix = mmread(dp_file).tocsr()
    alt_mat: csr_matrix = mmread(ad_file).tocsr()
    ref_mat = dp_mat - alt_mat

    dp_mat = dp_mat[:, barcode_indices]
    alt_mat = alt_mat[:, barcode_indices]
    ref_mat = ref_mat[:, barcode_indices]

    cell_snps = read_VCF_cellsnp_err_header(vcf_file)
    cell_snps["RAW_SNP_IDX"] = np.arange(len(cell_snps))  # use to index matrix
    return cell_snps, dp_mat, ref_mat, alt_mat


def load_calicost_prep_data(calicost_prep_dir: str, barcodes: list):
    """
    Here the barcodes list does not have slice suffix, and we only
    load allele counts from the listed barcodes.
    """
    print(f"load allele count data from CalicoST preprocessed files")
    barcode_file = os.path.join(calicost_prep_dir, "barcodes.txt")
    a_mtx_file = os.path.join(calicost_prep_dir, "cell_snp_Aallele.npz")
    b_mtx_file = os.path.join(calicost_prep_dir, "cell_snp_Ballele.npz")
    usnp_id_file = os.path.join(calicost_prep_dir, "unique_snp_ids.npy")

    raw_barcodes = read_barcodes(barcode_file)
    barcode_indices = np.array([raw_barcodes.index(x) for x in barcodes])

    # transpose to match (SNP, barcodes)
    alt_mat = sparse.load_npz(a_mtx_file).T
    ref_mat = sparse.load_npz(b_mtx_file).T
    dp_mat = alt_mat + ref_mat

    dp_mat = dp_mat[:, barcode_indices]
    alt_mat = alt_mat[:, barcode_indices]
    ref_mat = ref_mat[:, barcode_indices]

    unique_snp_ids = np.load(usnp_id_file, allow_pickle=True)
    cell_snps = pd.DataFrame(
        [x.split("_") for x in unique_snp_ids], columns=["#CHR", "POS", "REF", "ALT"]
    )
    if not cell_snps["#CHR"].str.startswith("chr").any():
        cell_snps["#CHR"] = "chr" + cell_snps["#CHR"].astype(str)

    assert (len(cell_snps), len(barcodes)) == ref_mat.shape, (
        "even the matrix shape does not match!"
    )
    # check for duplicates, wired bug in CalicoST, fix later
    dup_mask = cell_snps.duplicated(subset=["#CHR", "POS"], keep="first")
    if np.any(dup_mask):
        print("found bug in CalicoST!!!!!!!!!!!!!!!!!!!!!")
        dup_idx = cell_snps.index[dup_mask]
        print("Duplicated rows (removed):")
        print(cell_snps.loc[dup_idx, ["#CHR", "POS"]])
        cell_snps = cell_snps.loc[~dup_mask].reset_index(drop=True)
        dp_mat = dp_mat[~dup_mask.to_numpy(), :]
        alt_mat = alt_mat[~dup_mask.to_numpy(), :]
        ref_mat = ref_mat[~dup_mask.to_numpy(), :]
        print(f"Removed {dup_mask.sum()} duplicated rows")

    cell_snps["POS"] = cell_snps["POS"].astype(int)
    cell_snps["RAW_SNP_IDX"] = np.arange(len(cell_snps))  # use to index matrix
    print(f"#SNPs={len(cell_snps)}")

    # placeholders
    cell_snps["GT"] = 1
    cell_snps["PS"] = 1
    cell_snps["DP"] = np.array(dp_mat.sum(axis=1), dtype=np.int32).ravel()
    return [cell_snps, dp_mat, ref_mat, alt_mat]


def load_genetic_map(genetic_map_file: str, mode="eagle2", sep=" "):
    assert mode == "eagle2"
    genetic_map = pd.read_table(genetic_map_file, sep=sep, index_col=None).rename(
        columns={
            "chr": "#CHR",
            "position": "POS",
            "COMBINED_rate(cM/Mb)": "recomb_rate",
            "Genetic_Map(cM)": "pos_cm",
        }
    )
    genetic_map["#CHR"] = genetic_map["#CHR"].astype(str)
    genetic_map.loc[genetic_map["#CHR"] == "23", "#CHR"] = "X"
    if not genetic_map["#CHR"].str.startswith("chr").any():
        genetic_map["#CHR"] = "chr" + genetic_map["#CHR"].astype(str)
    genetic_map = sort_df_chr(genetic_map, ch="#CHR", pos="POS")
    genetic_map["POS0"] = genetic_map["POS"] - 1
    return genetic_map


##################################################
# copytyping IOs
##################################################


def load_allele_input(mod_dir: str):
    bin_Aallele_file = os.path.join(mod_dir, "Aallele.npz")
    bin_Ballele_file = os.path.join(mod_dir, "Ballele.npz")
    bin_Tallele_file = os.path.join(mod_dir, "Tallele.npz")

    a_allele_mat: np.ndarray = (
        sparse.load_npz(bin_Aallele_file).toarray().astype(dtype=np.int32)
    )
    b_allele_mat: np.ndarray = (
        sparse.load_npz(bin_Ballele_file).toarray().astype(dtype=np.int32)
    )
    t_allele_mat: np.ndarray = (
        sparse.load_npz(bin_Tallele_file).toarray().astype(dtype=np.int32)
    )
    return a_allele_mat, b_allele_mat, t_allele_mat


def load_count_input(mod_dir: str):
    bin_count_file = os.path.join(mod_dir, f"count.npz")
    bin_count_mat: np.ndarray = (
        sparse.load_npz(bin_count_file).toarray().astype(dtype=np.int32)
    )
    return bin_count_mat


def load_spatial_coordinates(coord_file: str):
    pass


def parse_cnv_profile(haplo_blocks: pd.DataFrame, laplace=0.01):
    num_clones = len(str(haplo_blocks["CNP"].iloc[0]).split(";"))
    clones = ["normal"] + [f"clone{i}" for i in range(1, num_clones)]
    A = np.zeros((len(haplo_blocks), num_clones), dtype=np.int32)
    B = np.zeros((len(haplo_blocks), num_clones), dtype=np.int32)
    for i in range(num_clones):
        A[:, i] = haplo_blocks.apply(
            func=lambda r: int(r["CNP"].split(";")[i].split("|")[0]), axis=1
        ).to_numpy()
        B[:, i] = haplo_blocks.apply(
            func=lambda r: int(r["CNP"].split(";")[i].split("|")[1]), axis=1
        ).to_numpy()
    C = A + B
    BAF = np.divide(
        B,
        C,
        out=np.zeros_like(C, dtype=np.float32),
        where=(C > 0),
    )
    BAF = np.clip(BAF, laplace, 1 - laplace)

    # assign the CNP group id
    return clones, A, B, C, BAF


def parse_total_cnp(bin_info: pd.DataFrame):
    num_clones = len(str(bin_info["CNP"].iloc[0]).split(";"))
    clones = ["normal"] + [f"clone{i}" for i in range(1, num_clones)]
    A = np.zeros((len(bin_info), num_clones), dtype=np.int32)
    B = np.zeros((len(bin_info), num_clones), dtype=np.int32)
    for i in range(num_clones):
        A[:, i] = bin_info.apply(
            func=lambda r: int(r["CNP"].split(";")[i].split("|")[0]), axis=1
        ).to_numpy()
        B[:, i] = bin_info.apply(
            func=lambda r: int(r["CNP"].split(";")[i].split("|")[1]), axis=1
        ).to_numpy()
    C = A + B
    # assign the CNP group id
    return clones, A, B, C


##################################################
# validation IOs
##################################################


def load_visium_path_annotation(
    ann_file: str, raw_label="Microregion_annotation", label="path_label", anns=None
):
    def simplify_label(v):
        if v[0] == "T":
            if "_" not in v:
                return "tumor"
            else:
                return v[str(v).find("_") + 1 :]
        else:
            return v

    path_anns = pd.read_table(ann_file, sep="\t", keep_default_na=True).rename(
        columns={"Barcode": "BARCODE"}
    )

    # NA labels are non-tumor
    path_anns[raw_label].fillna("normal", inplace=True)
    path_anns[raw_label] = path_anns.apply(
        func=lambda r: simplify_label(r[raw_label]), axis=1
    )
    path_anns[raw_label] = path_anns[raw_label].astype("str")
    path_anns[label] = path_anns[raw_label]

    if not anns is None:
        anns = pd.merge(anns, path_anns, on="BARCODE", how="left").set_index(
            "BARCODE", drop=False
        )
        return anns
    return path_anns
