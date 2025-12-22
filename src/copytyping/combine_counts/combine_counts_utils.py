import os
import sys

import numpy as np
import pandas as pd

import scanpy as sc
from scanpy import AnnData
import squidpy as sq
import snapatac2 as snap

from scipy.io import mmread
from scipy import sparse
from scipy.sparse import issparse, csc_matrix, csr_matrix

from copytyping.utils import *
from copytyping.io_utils import *
from copytyping.external import *


##################################################
# def annotate_snps(
#     segs: pd.DataFrame,
#     snp_info: pd.DataFrame,
#     allele_counts: np.ndarray,
#     ref_counts: np.ndarray,
#     alt_counts: np.ndarray,
#     min_count=2,
# ):
#     """
#     annotate POS0, SEG_IDX, balanced state, START=POS0, END=POS
#              DP, PHASE_RAW, B_ALLELE_RAW
#     filter SNPs with bulk count<min_count
#     """
#     assert len(ref_counts) == len(alt_counts)
#     assert len(ref_counts) == len(snp_info)
#     snp_info["POS0"] = snp_info["POS"] - 1

#     snp_info = annotate_snps_seg_idx(segs, snp_info, "SEG_IDX")
#     na_snps = snp_info["SEG_IDX"].isna().to_numpy()
#     low_qual_snps = allele_counts < min_count
#     snp_mask = na_snps | low_qual_snps

#     num_masked_snps = np.sum(snp_mask)
#     print(
#         f"#SNPs outside CNV segments or low counts: {num_masked_snps}/{len(snp_info)}={num_masked_snps / len(snp_info):.3%}"
#     )
#     allele_counts = allele_counts[~snp_mask]
#     ref_counts = ref_counts[~snp_mask]
#     alt_counts = alt_counts[~snp_mask]
#     snp_info = snp_info.loc[~snp_mask].copy(deep=True)
#     snp_info["SEG_IDX"] = snp_info["SEG_IDX"].astype(segs.index.dtype)
#     snp_info = snp_info.reset_index(drop=True)

#     snp_info["imbalanced"] = segs.loc[
#         snp_info["SEG_IDX"].to_numpy(), "imbalanced"
#     ].to_numpy()

#     snp_info["START"] = snp_info["POS0"]
#     snp_info["END"] = snp_info["POS"]

#     snp_info["DP"] = allele_counts
#     snp_info["PHASE_RAW"] = snp_info["GT"].astype(np.int8)
#     phases = snp_info["PHASE_RAW"].to_numpy()
#     snp_info["B_ALLELE_RAW"] = (phases * ref_counts + (1 - phases) * alt_counts).astype(
#         allele_counts.dtype
#     )
#     snp_info["A_ALLELE_RAW"] = (snp_info["DP"] - snp_info["B_ALLELE_RAW"]).astype(
#         allele_counts.dtype
#     )
#     snp_info["BAF_RAW"] = (snp_info["B_ALLELE_RAW"] / snp_info["DP"]).round(3)
#     return snp_info, allele_counts, ref_counts, alt_counts


def annotate_snps(segs: pd.DataFrame, snp_info: pd.DataFrame, seg_id="SEG_IDX"):
    """
    annotate SEG_IDX, balanced state, START=POS0, END=POS
    filter SNPs with bulk count<min_count
    """
    snp_info["POS0"] = snp_info["POS"] - 1

    snp_info = annotate_snps_seg_idx(segs, snp_info, seg_id)
    snp_mask = snp_info[seg_id].isna().to_numpy()

    num_masked_snps = np.sum(snp_mask)
    print(
        f"#SNPs outside CNV segments or low counts: {num_masked_snps}/{len(snp_info)}={num_masked_snps / len(snp_info):.3%}"
    )
    snp_info = snp_info.loc[~snp_mask].copy(deep=True)
    snp_info[seg_id] = snp_info[seg_id].astype(segs.index.dtype)
    snp_info = snp_info.reset_index(drop=True)

    snp_info["imbalanced"] = segs.loc[
        snp_info[seg_id].to_numpy(), "imbalanced"
    ].to_numpy()

    snp_info["START"] = snp_info["POS0"]
    snp_info["END"] = snp_info["POS"]
    return snp_info


def annotate_snps_post(
    snp_info: pd.DataFrame, allele_infos: dict, min_phase_posterior=0.95
):
    """
    any SNPs that are unphased or has max(phase, 1-phase)<min_phase_posterior are discarded.
    """
    low_confidence_snps = (
        np.maximum(snp_info["PHASE"], 1 - snp_info["PHASE"]) < min_phase_posterior
    )
    print(f"#low-confidence phased SNPs={np.sum(low_confidence_snps)}/{len(snp_info)}")
    snp_info = snp_info.loc[~low_confidence_snps, :].reset_index(drop=True)

    for data_type in allele_infos.keys():
        cell_snps = allele_infos[data_type][0]
        cell_snps = (
            cell_snps.reset_index(drop=False)
            .merge(
                right=snp_info[["#CHR", "POS", "POS0", "PHASE", "HB"]],
                on=["#CHR", "POS"],
                how="left",
                sort=False,
            )
            .set_index("index")
        )

        # some SNPs may outside CNV segments or unphased
        num_filtered = np.sum(cell_snps["PHASE"].isna())
        print(f"{data_type}, #unphased SNPs={num_filtered}/{len(cell_snps)}")
        cell_snps = cell_snps.loc[cell_snps["PHASE"].notna(), :].copy(deep=True)

        cell_snps["PHASE"] = cell_snps["PHASE"].astype(snp_info["PHASE"].dtype)
        cell_snps["HB"] = cell_snps["HB"].astype(snp_info["HB"].dtype)
        cell_snps["POS0"] = cell_snps["POS0"].astype(snp_info["POS"].dtype)
        allele_infos[data_type][0] = cell_snps
    return allele_infos


def annotate_snps_post2(
    snp_info: pd.DataFrame,
    allele_data: list,
):
    cell_snps = allele_data[0]
    cell_snps = (
        allele_data[0]
        .reset_index(drop=False)
        .merge(
            right=snp_info[["#CHR", "POS", "POS0", "PHASE", "HB"]],
            on=["#CHR", "POS"],
            how="left",
            sort=False,
        )
        .set_index("index")
    )

    # some SNPs may outside CNV segments or unphased
    num_filtered = np.sum(cell_snps["PHASE"].isna())
    print(f"#unphased SNPs={num_filtered}/{len(cell_snps)}")
    cell_snps = cell_snps.loc[cell_snps["PHASE"].notna(), :].copy(deep=True)

    cell_snps["PHASE"] = cell_snps["PHASE"].astype(snp_info["PHASE"].dtype)
    cell_snps["HB"] = cell_snps["HB"].astype(snp_info["HB"].dtype)
    cell_snps["POS0"] = cell_snps["POS0"].astype(snp_info["POS"].dtype)
    allele_data[0] = cell_snps
    return allele_data


##################################################
def feature_to_haplo_blocks(
    adata: AnnData,
    haplo_blocks: pd.DataFrame,
    modality: str,
):
    """
    filter features not in haplotype blocks, likely masked regions include centromeres
    """
    print(f"assign features with HB ID for {modality}")
    adata.var["feature_ids"] = adata.var.index.astype(str)
    adata.var["FEATURE_ID"] = np.arange(len(adata.var))

    feature_df = adata.var.reset_index(drop=True)
    print(f"#{modality}-features (raw)={len(feature_df)}")

    feature_df = assign_largest_overlap(feature_df, haplo_blocks, "FEATURE_ID", "HB")
    isna_features = feature_df["HB"].isna()
    print(
        f"#{modality} feature outside any haplotype blocks={np.sum(isna_features) / len(feature_df):.3%}"
    )
    feature_df.dropna(subset="HB", inplace=True)
    feature_df["HB"] = feature_df["HB"].astype(np.int32)
    print(f"#{modality} feature (remain)={len(feature_df)}")

    ##################################################
    # map HB tag to adata.var, filter any out-of-range features
    adata.var = (
        adata.var.reset_index(drop=False)
        .merge(
            right=feature_df[["FEATURE_ID", "HB"]],
            on="FEATURE_ID",
            how="left",
        )
        .set_index("index")
    )

    adata = adata[:, adata.var["HB"].notna()].copy()
    adata.var["HB"] = adata.var["HB"].astype(feature_df["HB"].dtype)

    ##################################################
    idx_map = dict(zip(feature_df["FEATURE_ID"], np.arange(len(feature_df))))
    adata.var["FEATURE_ID"] = adata.var["FEATURE_ID"].map(idx_map)

    feature_df = feature_df.reset_index(drop=True)
    feature_df["FEATURE_ID"] = np.arange(len(feature_df))
    feature_df = pd.merge(
        left=feature_df,
        right=haplo_blocks[["HB", "CNP"]],
        on="HB",
        how="left",
        sort=False,
    )
    return adata, feature_df


def snp_to_region(
    snp_df: pd.DataFrame, region_df: pd.DataFrame, modality: str, region_id="BIN_ID"
):
    """
    region_df must contain 0-indexed non-ovlp intervals
    """
    # assign SNPs to var_super_df & filter non-super SNPs
    print(f"#{modality}-SNP (raw)={len(snp_df)}")
    snp_df = assign_pos_to_range(snp_df, region_df, ref_id=region_id, pos_col="POS0")
    isna_snp_df = snp_df[region_id].isna()
    print(
        f"#{modality}-SNPs outside any region={np.sum(isna_snp_df) / len(snp_df):.3%}"
    )
    snp_df.dropna(subset=region_id, inplace=True)
    snp_df[region_id] = snp_df[region_id].astype(region_df[region_id].dtype)
    print(f"#{modality}-SNP (remain)={len(snp_df)}")

    counts = snp_df[region_id].value_counts()
    region_df["#SNPS"] = region_df[region_id].map(counts).fillna(0).astype(int)
    return snp_df


def feature_binning(
    adata: AnnData,
    feature_df: pd.DataFrame,
    tmp_dir: str,
    modality: str,
    feature_may_overlap=True,
    binning_strategy="adaptive",
    bin_colname="BIN_ID",
    min_med_count=5,
    max_nfeature=50,
):
    """
    adata and feature_df has same var dim
    inplace modify adata and feature_df
    """
    if feature_may_overlap:
        print("detect overlapping features")
        tmp_feature_in_file = os.path.join(tmp_dir, f"tmp_{modality}.in.bed")
        tmp_feature_out_file = os.path.join(tmp_dir, f"tmp_{modality}.out.bed")
        feature_df.to_csv(
            tmp_feature_in_file,
            sep="\t",
            header=False,
            index=False,
            columns=["#CHR", "START", "END", "FEATURE_ID"],
        )
        var_df_clustered = run_bedtools_cluster(
            tmp_feature_in_file,
            tmp_feature_out_file,
            tmp_dir,
            max_dist=0,
            load_df=True,
            usecols=list(range(5)),
            names=["#CHR", "START", "END", "FEATURE_ID", "SUPER_VAR_IDX"],
        )
        var_df_clustered["SUPER_VAR_IDX"] -= 1  # convert to 0-based
        feature_df = pd.merge(
            left=feature_df,
            right=var_df_clustered[["FEATURE_ID", "SUPER_VAR_IDX"]],
            on="FEATURE_ID",
            how="left",
            sort=False,
        )
        num_ovlp = len(feature_df) - len(feature_df["SUPER_VAR_IDX"].unique())
        print(f"found {num_ovlp} overlapping features")
    else:
        feature_df["SUPER_VAR_IDX"] = feature_df["FEATURE_ID"]

    ##################################################
    # binning over HB and SUPER_VAR_IDX
    if issparse(adata.X) and not isinstance(adata.X, csc_matrix):
        adata.X = adata.X.tocsc()

    if binning_strategy == "adaptive":
        bin_id = 0
        feature_df[bin_colname] = -1
        for hb, hb_features in feature_df.groupby(by="HB", sort=False):
            hb_supers = hb_features.groupby(by="SUPER_VAR_IDX", sort=False).groups
            sup_ids = hb_features["SUPER_VAR_IDX"].unique()
            num_hb_supers = len(sup_ids)
            curr_idx = [0]
            acc_counts = adata.X[:, hb_supers[sup_ids[0]]].sum(1).A1
            acc_nfeature = len(hb_supers[sup_ids[0]])
            bin_id0 = bin_id
            for i in range(1, num_hb_supers):
                nxt_gene_ids = hb_supers[sup_ids[i]]
                nxt_counts = adata.X[:, nxt_gene_ids].sum(1).A1
                nxt_nfeature = len(nxt_gene_ids)
                if (np.median(acc_counts) >= min_med_count) or (
                    acc_nfeature >= max_nfeature
                ):
                    assign = np.concatenate([hb_supers[sup_ids[j]] for j in curr_idx])
                    feature_df.loc[assign, bin_colname] = bin_id

                    bin_id += 1
                    curr_idx = [i]
                    acc_counts = nxt_counts
                    acc_nfeature = nxt_nfeature
                else:
                    curr_idx.append(i)
                    acc_counts += nxt_counts
                    acc_nfeature += nxt_nfeature
            assign = np.concatenate([hb_supers[sup_ids[j]] for j in curr_idx])
            feature_df.loc[assign, bin_colname] = max(bin_id - 1, bin_id0)
            bin_id = max(bin_id, bin_id0 + 1)
    else:
        # ensure BIN_ID is contiguous
        feature_df[bin_colname], hb_levels = pd.factorize(feature_df["HB"], sort=True)

    ##################################################
    # merge BIN_ID with adata
    adata.var = (
        adata.var.reset_index(drop=False)
        .merge(
            right=feature_df[["FEATURE_ID", bin_colname]],
            on="FEATURE_ID",
            how="left",
        )
        .set_index("index")
    )

    ##################################################
    # construct bin_df
    var_feature_bins = feature_df.groupby(by=bin_colname, sort=False, as_index=True)
    feature_bins = var_feature_bins.agg(
        **{
            "#CHR": ("#CHR", "first"),
            "START": ("START", "min"),
            "END": ("END", "max"),
            "HB": ("HB", "first"),
            "CNP": ("CNP", "first"),
            "feature_ids": ("feature_ids", lambda x: "|".join(map(str, x.unique()))),
        }
    ).reset_index(drop=False)
    feature_bins.loc[:, "#features"] = var_feature_bins.size().reset_index(drop=True)
    feature_bins.loc[:, "BLOCKSIZE"] = feature_bins["END"] - feature_bins["START"]
    print(f"#bins={len(feature_bins)}")
    print(f"#cna-bins={len(feature_bins.loc[mark_cna(feature_bins)])}")
    return feature_bins


##################################################
def allele_binning(
    bin_df: pd.DataFrame,
    t_allele_mat: np.ndarray,
    data_type: str,
    binning_strategy="adaptive",
    med_allele_count=4,
    max_nfeature=50,
    bin_colname="BIN_ID",
):
    if binning_strategy == "adaptive":
        print(
            f"adaptive allele binning with med-allele-count={med_allele_count}\tmax #feats={max_nfeature} for {data_type}"
        )
        bin_id = 0
        bin_df[bin_colname] = -1
        for hb, hb_bins in bin_df.groupby(by="HB", sort=False):
            hb_bin_ids = hb_bins.index.to_numpy()
            nfeatures = len(hb_bin_ids)
            curr_idx = [0]
            acc_counts = t_allele_mat[hb_bin_ids[0], :].copy()
            acc_nfeature = hb_bins.loc[hb_bin_ids[0], "#features"]
            bin_id0 = bin_id
            for i in range(1, nfeatures):
                nxt_idx = hb_bin_ids[i]
                nxt_counts = t_allele_mat[nxt_idx, :].copy()
                nxt_nfeature = hb_bins.loc[nxt_idx, "#features"]
                if (np.median(acc_counts) >= med_allele_count) or (
                    acc_nfeature >= max_nfeature
                ):
                    bin_df.loc[hb_bin_ids[curr_idx], bin_colname] = bin_id

                    bin_id += 1
                    curr_idx = [i]
                    acc_counts = nxt_counts
                    acc_nfeature = nxt_nfeature
                else:
                    curr_idx.append(i)
                    acc_counts += nxt_counts
                    acc_nfeature += nxt_nfeature
            bin_df.loc[hb_bin_ids[curr_idx], bin_colname] = max(bin_id - 1, bin_id0)
            bin_id = max(bin_id, bin_id0 + 1)
    else:
        print(f"segment allele binning for {data_type}")
        bin_df[bin_colname], hb_levels = pd.factorize(bin_df["HB"], sort=True)

    allele_super_bins = bin_df.groupby(by=bin_colname, sort=False, as_index=True)
    allele_bins = allele_super_bins.agg(
        **{
            "#CHR": ("#CHR", "first"),
            "START": ("START", "min"),
            "END": ("END", "max"),
            "HB": ("HB", "first"),
            "CNP": ("CNP", "first"),
            "#features": ("#features", "sum"),
            "#SNPS": ("#SNPS", "sum"),
            "feature_ids": ("feature_ids", lambda x: "|".join(map(str, x.unique()))),
        }
    ).reset_index(drop=False)
    allele_bins.loc[:, "BLOCKSIZE"] = allele_bins["END"] - allele_bins["START"]
    print(f"#allele-bins={len(allele_bins)}")
    print(f"#imb-bins={len(allele_bins.loc[mark_imbalanced(allele_bins)])}")

    # stats per-bin statistics
    allele_bins[f"#D==0"] = 0
    # for allele_count in range(1, med_allele_count + 1):
    #     allele_bins[f"#D>={allele_count}"] = 0
    for bin_id in allele_bins[bin_colname].unique():
        bin_supers_index = allele_super_bins.get_group(bin_id).index.to_numpy()
        pcell_tcounts = np.sum(t_allele_mat[bin_supers_index, :], axis=0)
        allele_bins.loc[bin_id, "#D==0"] = np.sum(pcell_tcounts == 0)
        # for allele_count in range(1, med_allele_count + 1):
        #     allele_bins.loc[bin_id, f"#D>={allele_count}"] = np.sum(pcell_tcounts >= allele_count)
    return allele_bins


##################################################
def aggregate_allele_counts(
    var_bins: pd.DataFrame,
    cell_snps: pd.DataFrame,
    dp_mtx: csr_matrix | np.ndarray,
    ref_mtx: csr_matrix | np.ndarray,
    alt_mtx: csr_matrix | np.ndarray,
    data_type: str,
    agg_mode="cellsnp-lite",
    agg_colname="SUPER_VAR_IDX",
    out_dir=None,
    out_prefix="",
    verbose=1,
):
    print(f"aggregate allele counts per bin for {data_type}")
    num_bins = len(var_bins)
    num_barcodes = dp_mtx.shape[1]

    if agg_mode == "cellsnp-lite":
        # outputs
        b_allele_mat = np.zeros((num_bins, num_barcodes), dtype=np.int32)
        t_allele_mat = np.zeros((num_bins, num_barcodes), dtype=np.int32)
        cell_snps_bins = cell_snps.groupby(by=agg_colname, sort=False)
        for bin_id in cell_snps[agg_colname].unique():
            snps_bin = cell_snps_bins.get_group(bin_id)
            # original index in raw allele count matrix
            snp_indices = snps_bin["RAW_SNP_IDX"].to_numpy()  # (n, )
            snp_phases = snps_bin["PHASE"].to_numpy()[:, np.newaxis]  # (n, 1)

            # access allele-count matrix
            rows_dp = dp_mtx[snp_indices].toarray()
            rows_alt = alt_mtx[snp_indices].toarray()
            rows_ref = rows_dp - rows_alt

            # aggregate phased counts
            rows_beta = rows_alt * (1 - snp_phases) + rows_ref * snp_phases
            b_allele_mat[bin_id, :] = np.round(np.sum(rows_beta, axis=0))
            t_allele_mat[bin_id, :] = np.sum(rows_dp, axis=0)
        a_allele_mat = (t_allele_mat - b_allele_mat).astype(np.int32)
    else:
        assert agg_mode == "regular"
        bin_ids = cell_snps[agg_colname].to_numpy()
        a_allele_mat = aggregate_matrix(alt_mtx.T, bin_ids, num_bins)
        b_allele_mat = aggregate_matrix(ref_mtx.T, bin_ids, num_bins)
        t_allele_mat = aggregate_matrix(dp_mtx.T, bin_ids, num_bins)

    # append pseudobulk b-allele counts
    var_bins["D"] = np.sum(t_allele_mat, axis=1)
    var_bins["Y"] = np.sum(b_allele_mat, axis=1)
    var_bins["BAF"] = var_bins["Y"] / var_bins["D"]

    if verbose:
        sub_tmat = t_allele_mat[mark_imbalanced(var_bins), :]
        print(f"RAW allele matrix sparsity: {1 - np.mean(t_allele_mat != 0):.3%}")
        print(f"IMB allele matrix sparsity: {1 - np.mean(sub_tmat != 0):.3%}")
        # print(f"IMB allele matrix t=1: {np.mean(sub_tmat == 1):.3%}")
        # print(f"IMB allele matrix t=2: {np.mean(sub_tmat == 2):.3%}")
        # print(f"IMB allele matrix t=3: {np.mean(sub_tmat == 3):.3%}")
        # print(f"IMB allele matrix t>=4: {np.mean(sub_tmat >= 4):.3%}")

    if not out_dir is None:
        bin_Aallele_file = os.path.join(out_dir, f"{out_prefix}Aallele.npz")
        bin_Ballele_file = os.path.join(out_dir, f"{out_prefix}Ballele.npz")
        bin_Tallele_file = os.path.join(out_dir, f"{out_prefix}Tallele.npz")

        sparse.save_npz(bin_Aallele_file, csr_matrix(a_allele_mat))
        sparse.save_npz(bin_Ballele_file, csr_matrix(b_allele_mat))
        sparse.save_npz(bin_Tallele_file, csr_matrix(t_allele_mat))
    return a_allele_mat, b_allele_mat, t_allele_mat


def aggregate_var_counts(
    adata: AnnData,
    var_bins: pd.DataFrame,
    data_type: str,
    agg_colname="BIN_ID",
    out_file=None,
    verbose=1,
):
    """
    assumes both var_bins adata.var has <agg_colname> with consecutive and ordered values.
    """
    print(f"aggregate {data_type} total counts by {agg_colname}")
    n_bins = len(var_bins)
    feat_bin = adata.var[agg_colname].to_numpy()
    var_count_mat = aggregate_matrix(adata.X, feat_bin, n_bins)
    var_bins["VAR_DP"] = np.sum(var_count_mat, axis=1)
    var_bins["#expressed_cells"] = np.sum(var_count_mat > 0, axis=1)

    if verbose:
        print(f"feature matrix sparsity: {1 - np.mean(var_count_mat != 0):.3%}")

    if not out_file is None:
        sparse.save_npz(out_file, var_count_mat.tocsr())
    return var_count_mat


def aggregate_matrix(X, bin_ids, n_uniq_bins):
    """
    X: (nobs, nvars)
    return: (nbins, nobs)
    """
    n_feats = X.shape[1]
    indicator_matrix = sparse.csr_matrix(
        (np.ones_like(bin_ids, dtype=np.int8), (np.arange(n_feats), bin_ids)),
        shape=(n_feats, n_uniq_bins),
    )
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)

    bin_count_mat = X @ indicator_matrix  # shape: (n_cells × n_bins)
    bin_count_mat = bin_count_mat.T  # shape: (n_bins × n_cells)
    return bin_count_mat
