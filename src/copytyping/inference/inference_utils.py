import logging

import numpy as np
import pandas as pd
from scipy import sparse

from copytyping.utils import NA_CELLTYPE


def merge_celltype_into_barcodes(barcodes_df, cell_type_df, ref_label, data_type):
    """Merge cell_type annotations into barcodes DataFrame.

    Drops the ref_label column if all labels are uninformative (NA_CELLTYPE).
    """
    barcodes_df = pd.merge(
        left=barcodes_df,
        right=cell_type_df[["BARCODE", ref_label]],
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


def adaptive_bin_bbc(bbc_data, seg_blocks, min_snp_count=300, max_bin_length=5_000_000):
    """Merge adjacent BBC bins within the same segment to reduce sparsity.

    Walks BBC bins in genomic order per chromosome, grouping consecutive bins
    that map to the same segment until the pseudobulk SNP count reaches
    min_snp_count or the combined length exceeds max_bin_length.

    Args:
        bbc_data: dict with 'bbc_df', 'X', 'Y', 'D' (sparse or dense).
        seg_blocks: segment-level DataFrame with #CHR, START, END columns.
        min_snp_count: minimum pseudobulk D sum per merged bin.
        max_bin_length: maximum merged bin length in bp.

    Returns:
        dict with same keys as bbc_data but fewer, larger bins (dense arrays).
    """
    bbc_df = bbc_data["bbc_df"].reset_index(drop=True)
    X = bbc_data["X"].toarray() if sparse.issparse(bbc_data["X"]) else bbc_data["X"]
    Y = bbc_data["Y"].toarray() if sparse.issparse(bbc_data["Y"]) else bbc_data["Y"]
    D = bbc_data["D"].toarray() if sparse.issparse(bbc_data["D"]) else bbc_data["D"]
    X = X.astype(np.int32)
    Y = Y.astype(np.int32)
    D = D.astype(np.int32)

    # pseudobulk D for threshold check
    D_total = D.sum(axis=1)

    # map BBC bins to segments via midpoint containment
    bbc_mid = ((bbc_df["START"] + bbc_df["END"]) / 2).astype(np.int64).to_numpy()
    bbc_chr = bbc_df["#CHR"].to_numpy()
    seg_id = np.full(len(bbc_df), -1, dtype=np.int64)

    for chrom in seg_blocks["#CHR"].unique():
        bm = bbc_chr == chrom
        if not bm.any():
            continue
        seg_ch = seg_blocks[seg_blocks["#CHR"] == chrom].sort_values("START")
        starts = seg_ch["START"].to_numpy()
        ends = seg_ch["END"].to_numpy()
        sids = seg_ch.index.to_numpy()
        idx = np.searchsorted(starts, bbc_mid[bm], side="right") - 1
        safe = idx.clip(min=0)
        ok = (idx >= 0) & (bbc_mid[bm] < ends[safe])
        seg_id[np.where(bm)[0][ok]] = sids[idx[ok]]

    # walk and merge
    groups = []  # list of (chr, start, end, [bbc_indices])
    chroms = bbc_df["#CHR"].unique()
    for chrom in chroms:
        chr_mask = bbc_chr == chrom
        chr_idx = np.where(chr_mask)[0]
        if len(chr_idx) == 0:
            continue
        # sort by START within chromosome
        order = chr_idx[np.argsort(bbc_df["START"].to_numpy()[chr_idx])]

        cur_seg = seg_id[order[0]]
        cur_start = bbc_df["START"].iloc[order[0]]
        cur_end = bbc_df["END"].iloc[order[0]]
        cur_indices = [order[0]]
        cur_d = D_total[order[0]]

        for i in range(1, len(order)):
            bi = order[i]
            bi_seg = seg_id[bi]
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
                groups.append((chrom, cur_start, cur_end, cur_indices))
                cur_seg = bi_seg
                cur_start = bi_start
                cur_end = bi_end
                cur_indices = [bi]
                cur_d = D_total[bi]

        groups.append((chrom, cur_start, cur_end, cur_indices))

    # build aggregated arrays
    n_agg = len(groups)
    N = X.shape[1]
    X_agg = np.zeros((n_agg, N), dtype=np.int32)
    Y_agg = np.zeros((n_agg, N), dtype=np.int32)
    D_agg = np.zeros((n_agg, N), dtype=np.int32)
    rows = []

    for gi, (chrom, start, end, indices) in enumerate(groups):
        idx = np.array(indices)
        X_agg[gi] = X[idx].sum(axis=0)
        Y_agg[gi] = Y[idx].sum(axis=0)
        D_agg[gi] = D[idx].sum(axis=0)
        rows.append({"#CHR": chrom, "START": start, "END": end})

    agg_df = pd.DataFrame(rows)

    # log stats
    lengths = (agg_df["END"] - agg_df["START"]).to_numpy()
    d_sums = D_agg.sum(axis=1)
    x_sums = X_agg.sum(axis=1)
    logging.info(
        f"adaptive_bin_bbc: {len(bbc_df)} -> {n_agg} bins, "
        f"length median={np.median(lengths):.0f} mean={np.mean(lengths):.0f}, "
        f"snp_count median={np.median(d_sums):.0f} mean={np.mean(d_sums):.0f}, "
        f"total_count median={np.median(x_sums):.0f} mean={np.mean(x_sums):.0f}"
    )

    return {"bbc_df": agg_df, "X": X_agg, "Y": Y_agg, "D": D_agg}
