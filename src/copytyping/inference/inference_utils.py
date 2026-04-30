import logging

import numpy as np
import pandas as pd
from scipy import sparse

from copytyping.sx_data.sx_data import SX_Data
from copytyping.utils import NA_CELLTYPE


def merge_celltype_into_barcodes(barcodes_df, cell_type_df, ref_label, data_type):
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
            f"all {data_type} barcodes have "
            f"uninformative {ref_label} labels "
            f"(all in NA_CELLTYPE={NA_CELLTYPE})"
        )
        barcodes_df = barcodes_df.drop(columns=[ref_label])
    return barcodes_df


def annotate_adata_celltype(adata, cell_type_df, ref_label, data_type):
    """Add cell_type annotations to adata.obs from cell_type_df."""
    ct_map = cell_type_df.set_index("BARCODE")[ref_label]
    if ref_label in adata.obs.columns:
        logging.warning(
            f"overwriting existing '{ref_label}' column "
            f"in {data_type} h5ad obs with cell_type_df"
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

    def _to_dense(m):
        return (
            m.toarray().astype(np.int32)
            if sparse.issparse(m)
            else np.asarray(m, dtype=np.int32)
        )

    X = _to_dense(X_bbc)
    Y = _to_dense(Y_bbc)
    D = _to_dense(D_bbc)

    D_total = D.sum(axis=1)
    bbc_chr = bbc_df["#CHR"].to_numpy()
    seg_ids = bbc_df["seg_id"].to_numpy()

    # walk and merge
    groups = []  # list of (chr, start, end, seg_id, cnp, [bbc_indices])
    chroms = bbc_df["#CHR"].unique()
    for chrom in chroms:
        chr_mask = bbc_chr == chrom
        chr_idx = np.where(chr_mask)[0]
        if len(chr_idx) == 0:
            continue
        order = chr_idx[np.argsort(bbc_df["START"].to_numpy()[chr_idx])]

        cur_seg = seg_ids[order[0]]
        cur_start = bbc_df["START"].iloc[order[0]]
        cur_end = bbc_df["END"].iloc[order[0]]
        cur_cnp = bbc_df["CNP"].iloc[order[0]]
        cur_indices = [order[0]]
        cur_d = D_total[order[0]]

        for i in range(1, len(order)):
            bi = order[i]
            bi_seg = seg_ids[bi]
            bi_start = bbc_df["START"].iloc[bi]
            bi_end = bbc_df["END"].iloc[bi]
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
                cur_cnp = bbc_df["CNP"].iloc[bi]
                cur_indices = [bi]
                cur_d = D_total[bi]

        groups.append((chrom, cur_start, cur_end, cur_seg, cur_cnp, cur_indices))

    # build aggregated arrays
    n_agg = len(groups)
    N = X.shape[1]
    X_agg = np.zeros((n_agg, N), dtype=np.int32)
    Y_agg = np.zeros((n_agg, N), dtype=np.int32)
    D_agg = np.zeros((n_agg, N), dtype=np.int32)
    rows = []

    for gi, (chrom, start, end, sid, cnp, indices) in enumerate(groups):
        idx = np.array(indices)
        X_agg[gi] = X[idx].sum(axis=0)
        Y_agg[gi] = Y[idx].sum(axis=0)
        D_agg[gi] = D[idx].sum(axis=0)
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
