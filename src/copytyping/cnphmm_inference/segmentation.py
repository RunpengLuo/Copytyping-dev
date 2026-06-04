"""Segmentation utilities: BBC→seg adaptive aggregation, phase-corrected
seg-level matrices, and CNP-breakpoint primitives for the divisive loop.
"""

import logging

import numpy as np
import pandas as pd
from scipy import sparse


# ============================== pre-segmentation ==============================


def adaptive_segmentation(
    bbc_df: pd.DataFrame,
    C_bbc: sparse.spmatrix,
    segmentation_prior: np.ndarray,
    min_snp_count: int = 300,
    max_bin_length: int = 5_000_000,
) -> tuple[pd.DataFrame, sparse.csr_matrix]:
    """Merge adjacent BBC bins per chromosome until each segment reaches
    ``min_snp_count`` pseudobulk allele count or ``max_bin_length`` bp.

    Splits at ``segmentation_prior`` boundaries. The CNP-tuple-based prior
    already separates all-diploid bins from bins with at least one
    non-diploid clone (their CNP tuples differ), so no separate phased /
    unphased boundary check is needed.

    Returns:
        agg_df: DataFrame with #CHR, START, END, seg_id columns.
        agg_mat: (G_agg, G_bbc) sparse CSR one-hot aggregation matrix.
    """
    C = C_bbc.tocsr() if sparse.issparse(C_bbc) else sparse.csr_matrix(C_bbc)
    C_total = np.asarray(C.sum(axis=1)).ravel()
    bbc_chr = bbc_df["#CHR"].to_numpy()
    starts = bbc_df["START"].to_numpy()
    ends = bbc_df["END"].to_numpy()
    n_bbc = len(bbc_df)

    groups = []  # (chr, start, end, seg_id, [bbc_indices])
    for chrom in pd.unique(bbc_chr):
        chr_idx = np.where(bbc_chr == chrom)[0]
        if chr_idx.size == 0:
            continue
        order = chr_idx[np.argsort(starts[chr_idx])]

        cur_seg = segmentation_prior[order[0]]
        cur_start = starts[order[0]]
        cur_end = ends[order[0]]
        cur_indices = [order[0]]
        cur_c = C_total[order[0]]

        for i in range(1, len(order)):
            bi = order[i]
            same_seg = segmentation_prior[bi] == cur_seg
            fits_length = (ends[bi] - cur_start) <= max_bin_length
            needs_more = cur_c < min_snp_count

            if same_seg and fits_length and needs_more:
                cur_indices.append(bi)
                cur_end = ends[bi]
                cur_c += C_total[bi]
            else:
                groups.append((chrom, cur_start, cur_end, cur_seg, cur_indices))
                cur_seg = segmentation_prior[bi]
                cur_start = starts[bi]
                cur_end = ends[bi]
                cur_indices = [bi]
                cur_c = C_total[bi]

        groups.append((chrom, cur_start, cur_end, cur_seg, cur_indices))

    n_agg = len(groups)
    row_idx = []
    col_idx = []
    rows = []
    for gi, (chrom, start, end, sid, indices) in enumerate(groups):
        row_idx.extend([gi] * len(indices))
        col_idx.extend(indices)
        rows.append({"#CHR": chrom, "START": start, "END": end, "seg_id": sid})

    agg_mat = sparse.csr_matrix(
        (np.ones(len(row_idx), dtype=np.int32), (row_idx, col_idx)),
        shape=(n_agg, n_bbc),
    )
    agg_df = pd.DataFrame(rows)

    lengths = (agg_df["END"] - agg_df["START"]).to_numpy()
    logging.info(
        f"adaptive_segmentation: {n_bbc} -> {n_agg} bins, "
        f"length median={np.median(lengths):.0f} mean={np.mean(lengths):.0f}"
    )

    return agg_df, agg_mat


def perform_segmentation(
    agg_mat: sparse.csr_matrix,
    X_bbc: sparse.spmatrix,
    B_bbc: sparse.spmatrix,
    C_bbc: sparse.spmatrix,
    cna_profile: np.ndarray,
    phase_bbc: np.ndarray,
    switchprobs_bbc: np.ndarray,
) -> tuple[
    sparse.csr_matrix,
    sparse.csr_matrix,
    sparse.csr_matrix,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Apply bulk phase correction and aggregate BBC-level matrices to segments.

    ``B_corr`` (B-allele after applying the BBC-level ``phase_bbc``) is
    aggregated into ``B_seg``, so at seg level the data is in the phase-
    corrected orientation by construction — the per-seg phase label is 1
    everywhere.

    Returns ``(X_seg, B_seg, C_seg, cna_profile_seg, phase_seg,
    switchprobs_seg)``.
    """
    B = B_bbc.tocsr() if sparse.issparse(B_bbc) else sparse.csr_matrix(B_bbc)
    C = C_bbc.tocsr() if sparse.issparse(C_bbc) else sparse.csr_matrix(C_bbc)
    X = X_bbc.tocsr() if sparse.issparse(X_bbc) else sparse.csr_matrix(X_bbc)

    phases = phase_bbc.astype(np.float64)[:, None]
    B_corr = B.multiply(phases) + (C - B).multiply(1 - phases)

    X_seg = agg_mat @ X
    B_seg = agg_mat @ B_corr
    C_seg = agg_mat @ C

    n_seg = agg_mat.shape[0]
    first_bbc_per_group = np.array([agg_mat[g].indices[0] for g in range(n_seg)])
    cna_profile_seg = cna_profile[first_bbc_per_group]
    phase_seg = np.ones(n_seg, dtype=np.int8)
    switchprobs_seg = switchprobs_bbc[first_bbc_per_group]

    n_entries = X_seg.shape[0] * X_seg.shape[1]
    logging.info(
        f"perform_segmentation: X={X_seg.shape}, B={B_seg.shape}, C={C_seg.shape}, "
        f"cna_profile={cna_profile_seg.shape}, n_seg={n_seg}"
    )
    logging.info(
        f"  X sparsity: {1 - X_seg.nnz / n_entries:.3%} ({X_seg.nnz}/{n_entries} nonzero), "
        f"C sparsity: {1 - C_seg.nnz / n_entries:.3%} ({C_seg.nnz}/{n_entries} nonzero)"
    )
    return X_seg, B_seg, C_seg, cna_profile_seg, phase_seg, switchprobs_seg
