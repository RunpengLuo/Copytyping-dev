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
    cna_mirrored: np.ndarray,
    phase_bbc: np.ndarray,
    switchprobs_bbc: np.ndarray,
) -> tuple[
    sparse.csr_matrix,
    sparse.csr_matrix,
    sparse.csr_matrix,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Apply bulk phase correction and aggregate BBC-level matrices to segments.

    ``B_corr`` (B-allele after applying the BBC-level ``phase_bbc``) is
    aggregated into ``B_seg``, so at seg level the data is in the phase-
    corrected orientation by construction — the per-seg phase label is 1
    everywhere. The returned ``phase_seg`` reflects that baseline; downstream
    models (e.g. the divisive loop's ``phase_chain``) start from these 1s
    and flip entries as accepted splits propose phase changes.

    Per-segment CNP / switchprob attributes take the first BBC in each
    group; the segment's left-boundary switchprob is therefore preserved
    while internal-BBC switchprobs are absorbed by the merge.

    Returns ``(X_seg, B_seg, C_seg, cna_profile_seg, cna_mirrored_seg,
    phase_seg, switchprobs_seg)``.
    """
    B = B_bbc.tocsr() if sparse.issparse(B_bbc) else sparse.csr_matrix(B_bbc)
    C = C_bbc.tocsr() if sparse.issparse(C_bbc) else sparse.csr_matrix(C_bbc)
    X = X_bbc.tocsr() if sparse.issparse(X_bbc) else sparse.csr_matrix(X_bbc)

    # phase correction at BBC level: B_corr = B if phase=1 else A = C - B.
    # Caller folds the diploid → phase=1 rule into ``phase_bbc`` upfront,
    # so no per-bin mask is needed here.
    phases = phase_bbc.astype(np.float64)[:, None]
    B_corr = B.multiply(phases) + (C - B).multiply(1 - phases)

    X_seg = agg_mat @ X
    B_seg = agg_mat @ B_corr
    C_seg = agg_mat @ C

    n_seg = agg_mat.shape[0]
    first_bbc_per_group = np.array([agg_mat[g].indices[0] for g in range(n_seg)])
    cna_profile_seg = cna_profile[first_bbc_per_group]
    cna_mirrored_seg = cna_mirrored[first_bbc_per_group]
    # B_seg is already phase-corrected → the per-seg phase label is 1
    # everywhere by construction.
    phase_seg = np.ones(n_seg, dtype=np.int8)
    switchprobs_seg = switchprobs_bbc[first_bbc_per_group]

    logging.info(
        f"perform_segmentation: X={X_seg.shape}, B={B_seg.shape}, C={C_seg.shape}, "
        f"cna_profile={cna_profile_seg.shape}, n_seg={n_seg}"
    )
    return (
        X_seg,
        B_seg,
        C_seg,
        cna_profile_seg,
        cna_mirrored_seg,
        phase_seg,
        switchprobs_seg,
    )


# =========================== CNP-breakpoint helpers ===========================


def derive_cnp_segments(
    genome_coords: pd.DataFrame,
    cna_profile_seg: np.ndarray,
    cna_mirrored_seg: np.ndarray,
    cna_int_states: np.ndarray,
    clone_ids: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Group bin-level segs into candidate segs by current CNP layout.

    A candidate seg is a maximal contiguous run of bin-level segs that share
    the same ``(state, mirror)`` tuple across all clones AND lie on the same
    chromosome.

    Returns:
        cand_idx: (G_seg,) int array mapping each bin-seg to its candidate-seg
            index in [0, n_cand).
        segments_df: one row per candidate seg with columns ``#CHR``, ``START``,
            ``END``, and ``cn_<clone>`` per clone — the effective ``A|B`` copy
            number (mirror flag folded; ``mirror=1`` swaps canonical to
            ``B|A``).
    """
    cn_cols = [f"cn_{name}" for name in clone_ids]
    G = cna_profile_seg.shape[0]
    if G == 0:
        return np.zeros(0, dtype=np.int64), pd.DataFrame(
            columns=["#CHR", "START", "END", "LENGTH", *cn_cols]
        )

    chr_arr = genome_coords["#CHR"].to_numpy()
    starts = genome_coords["START"].to_numpy()
    ends = genome_coords["END"].to_numpy()
    chr_change = chr_arr[1:] != chr_arr[:-1]
    combined = (cna_profile_seg.astype(np.int64) << 8) | (
        cna_mirrored_seg.astype(np.int64) & 0xFF
    )
    cnp_change = np.any(combined[1:] != combined[:-1], axis=1)
    breakpoints = np.r_[True, chr_change | cnp_change]

    cand_idx = breakpoints.cumsum() - 1
    first_g = np.flatnonzero(breakpoints)  # (n_cand,) — cand-seg starts
    last_g = np.r_[first_g[1:] - 1, G - 1]  # (n_cand,) — cand-seg ends
    seg_starts = starts[first_g]
    seg_ends = ends[last_g]
    seg_lengths = seg_ends - seg_starts

    rows: dict[str, np.ndarray | list[str]] = {
        "#CHR": chr_arr[first_g],
        "START": seg_starts,
        "END": seg_ends,
        "LENGTH": seg_lengths,
    }
    for m, name in enumerate(clone_ids):
        state_idx = cna_profile_seg[first_g, m]
        mirror = cna_mirrored_seg[first_g, m]
        a_copy = cna_int_states[state_idx, 0]
        b_copy = cna_int_states[state_idx, 1]
        major = np.where(mirror == 0, a_copy, b_copy)
        minor = np.where(mirror == 0, b_copy, a_copy)
        rows[f"cn_{name}"] = [f"{a}|{b}" for a, b in zip(major, minor)]

    return cand_idx, pd.DataFrame(rows)
