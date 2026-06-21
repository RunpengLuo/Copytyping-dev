import logging

import numpy as np
import pandas as pd
from scipy import sparse

from copytyping.sx_data.sx_data import SX_Data
from copytyping.utils import NA_CELLTYPE


def merge_celltype_into_barcodes(barcodes_df, cell_type_df, ref_label, assay_type):
    """Merge cell_type annotations into barcodes DataFrame.

    Drops the ref_label column if all labels are uninformative (NA_CELLTYPE).
    """
    merge_cols = ["BARCODE", ref_label]
    ref_purity_col = f"{ref_label}-tumor_purity"
    if ref_purity_col in cell_type_df.columns:
        merge_cols.append(ref_purity_col)
    barcodes_df = pd.merge(
        left=barcodes_df,
        right=cell_type_df[merge_cols],
        on="BARCODE",
        how="left",
        validate="1:1",
        sort=False,
    )
    barcodes_df[ref_label] = barcodes_df[ref_label].fillna("Unknown").astype(str)
    if barcodes_df[ref_label].isin(NA_CELLTYPE).all():
        logging.warning(
            f"all {assay_type} barcodes have "
            f"uninformative {ref_label} labels "
            f"(all in NA_CELLTYPE={NA_CELLTYPE})"
        )
        barcodes_df = barcodes_df.drop(columns=[ref_label])
    return barcodes_df


def annotate_adata_celltype(adata, cell_type_df, ref_label, assay_type):
    """Add cell_type annotations to adata.obs from cell_type_df."""
    ct_map = cell_type_df.set_index("BARCODE")[ref_label]
    if ref_label in adata.obs.columns:
        logging.warning(
            f"overwriting existing '{ref_label}' column "
            f"in {assay_type} h5ad obs with cell_type_df"
        )
    adata.obs[ref_label] = (
        adata.obs_names.to_series().map(ct_map).fillna("Unknown").values
    )


def adaptive_bin_bbc(
    bbc_df,
    X_bbc,
    Y_bbc,
    D_bbc,
    seg_sx,
    min_snp_count=300,
    max_bin_length=5_000_000,
):
    """Merge adjacent BBC bins within the same segment to reduce sparsity.

    Walks BBC bins in genomic order per chromosome, grouping consecutive bins
    that share the same seg_id until the pseudobulk SNP count reaches
    min_snp_count or the combined length exceeds max_bin_length.

    Args:
        bbc_df: BBC-level DataFrame with #CHR, START, END, seg_id, CNP columns.
        X_bbc, Y_bbc, D_bbc: (G_bbc, N) sparse or dense count matrices.
        seg_sx: segment-level SX_Data (for barcodes).
        min_snp_count: minimum pseudobulk D sum per merged bin.
        max_bin_length: maximum merged bin length in bp.

    Returns:
        SX_Data with aggregated bins.
    """
    bbc_df = bbc_df.reset_index(drop=True)

    # Keep X/Y/D sparse (CSR) — avoids ~30 GB densification of 66K × 38K matrices.
    # Per-group sum on sparse row slices is cheap; final aggregated matrix is small.
    X = X_bbc.tocsr() if sparse.issparse(X_bbc) else sparse.csr_matrix(X_bbc)
    Y = Y_bbc.tocsr() if sparse.issparse(Y_bbc) else sparse.csr_matrix(Y_bbc)
    D = D_bbc.tocsr() if sparse.issparse(D_bbc) else sparse.csr_matrix(D_bbc)

    D_total = np.asarray(D.sum(axis=1)).ravel()
    # Pre-extract as numpy arrays so the walk loop avoids pandas iloc per access.
    bbc_chr = bbc_df["#CHR"].to_numpy()
    seg_ids = bbc_df["seg_id"].to_numpy()
    starts = bbc_df["START"].to_numpy()
    ends = bbc_df["END"].to_numpy()
    cnps = bbc_df["CNP"].to_numpy()

    # bbc_df is already chr-pos sorted (HATCHet writes it that way); preserve that
    # row order via pd.unique rather than np.unique (which would lex-sort).
    groups = []  # list of (chr, start, end, seg_id, cnp, [bbc_indices])
    for chrom in pd.unique(bbc_chr):
        chr_idx = np.where(bbc_chr == chrom)[0]
        if chr_idx.size == 0:
            continue
        order = chr_idx[np.argsort(starts[chr_idx])]

        cur_seg = seg_ids[order[0]]
        cur_start = starts[order[0]]
        cur_end = ends[order[0]]
        cur_cnp = cnps[order[0]]
        cur_indices = [order[0]]
        cur_d = D_total[order[0]]

        for i in range(1, len(order)):
            bi = order[i]
            bi_seg = seg_ids[bi]
            bi_start = starts[bi]
            bi_end = ends[bi]
            bi_length = bi_end - cur_start

            same_seg = bi_seg == cur_seg and cur_seg >= 0
            fits_length = bi_length <= max_bin_length
            needs_more = cur_d < min_snp_count

            if same_seg and fits_length and needs_more:
                cur_indices.append(bi)
                cur_end = bi_end
                cur_d += D_total[bi]
            else:
                groups.append(
                    (chrom, cur_start, cur_end, cur_seg, cur_cnp, cur_indices)
                )
                cur_seg = bi_seg
                cur_start = bi_start
                cur_end = bi_end
                cur_cnp = cnps[bi]
                cur_indices = [bi]
                cur_d = D_total[bi]

        groups.append((chrom, cur_start, cur_end, cur_seg, cur_cnp, cur_indices))

    # build aggregated arrays — sum sparse rows per group, then densify the result
    n_agg = len(groups)
    N = X.shape[1]
    X_agg = np.zeros((n_agg, N), dtype=np.int32)
    Y_agg = np.zeros((n_agg, N), dtype=np.int32)
    D_agg = np.zeros((n_agg, N), dtype=np.int32)
    rows = []

    for gi, (chrom, start, end, sid, cnp, indices) in enumerate(groups):
        idx = np.asarray(indices)
        X_agg[gi] = np.asarray(X[idx].sum(axis=0)).ravel()
        Y_agg[gi] = np.asarray(Y[idx].sum(axis=0)).ravel()
        D_agg[gi] = np.asarray(D[idx].sum(axis=0)).ravel()
        rows.append(
            {"#CHR": chrom, "START": start, "END": end, "seg_id": sid, "CNP": cnp}
        )

    agg_df = pd.DataFrame(rows)

    # Drop unmapped bins (seg_id=-1, no CNP)
    mapped = agg_df["seg_id"] >= 0
    if not mapped.all():
        n_drop = (~mapped).sum()
        logging.warning(f"adaptive_bin_bbc: dropping {n_drop} unmapped bins")
        keep = mapped.to_numpy()
        agg_df = agg_df[keep].reset_index(drop=True)
        X_agg = X_agg[keep]
        Y_agg = Y_agg[keep]
        D_agg = D_agg[keep]

    lengths = (agg_df["END"] - agg_df["START"]).to_numpy()
    d_sums = D_agg.sum(axis=1)
    x_sums = X_agg.sum(axis=1)
    logging.info(
        f"adaptive_bin_bbc: {len(bbc_df)} -> {n_agg} bins, "
        f"length median={np.median(lengths):.0f} mean={np.mean(lengths):.0f}, "
        f"snp_count median={np.median(d_sums):.0f} mean={np.mean(d_sums):.0f}, "
        f"total_count median={np.median(x_sums):.0f} mean={np.mean(x_sums):.0f}"
    )

    return SX_Data(seg_sx.barcodes, agg_df, X_agg, Y_agg, D_agg)


def compute_loh_baf(
    ballele_counts: np.ndarray,
    total_allele_counts: np.ndarray,
    cn_A: np.ndarray,
    cn_B: np.ndarray,
    clones: list[str],
):
    """Per-spot aggregated BAF over clone-specific LOH clusters.

    Args:
        ballele_counts: (G, N) cluster-level B-allele counts.
        total_allele_counts: (G, N) cluster-level total-allele counts (A + B).
        cn_A/cn_B: (G, K) per-clone copy numbers.
        clones: clone names, length K.

    Returns (baf_array, loh_info) where:
        baf_array: float (N, K_tumor) — per-spot BAF aggregated over LOH clusters of each tumor clone.
            NaN if no allele coverage or no LOH clusters for that clone.
        loh_info: list of (clone_name, list of "cluster <tab> clone states") per clone with LOH.
    """
    num_clones = len(clones)
    num_cells = ballele_counts.shape[1]
    K_tumor = num_clones - 1
    baf = np.full((num_cells, K_tumor), np.nan)
    loh_info = []

    for ki in range(K_tumor):
        k = ki + 1  # skip normal
        clone = clones[k]
        loh_mask = (cn_B[:, k] == 0) & (cn_A[:, k] > 0)
        if loh_mask.sum() == 0:
            continue

        entries = []
        for gi in np.where(loh_mask)[0]:
            cn_parts = [
                f"{clones[j]}={cn_A[gi, j]}|{cn_B[gi, j]}" for j in range(num_clones)
            ]
            entries.append(f"cluster{gi}\t{', '.join(cn_parts)}")
        loh_info.append((clone, entries))

        Y_loh = ballele_counts[loh_mask].sum(axis=0).astype(float)
        D_loh = total_allele_counts[loh_mask].sum(axis=0).astype(float)
        valid = D_loh > 0
        baf[valid, ki] = Y_loh[valid] / D_loh[valid]

        logging.info(f"LOH clusters for {clone} ({int(loh_mask.sum())} clusters):")
        for entry in entries:
            logging.info(f"  {entry}")

    return baf, loh_info
