import logging

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix

from copytyping.utils import read_seg_ucn_file

##################################################
# preprocess IOs
##################################################


def load_modality_data(
    bc_file: str,
    bbc_file: str,
    x_count_file: str,
    a_allele_file: str,
    b_allele_file: str,
    bbc_phases_file: str,
    data_type: str,
    seg_ucn_file: str,
    solfile=None,
):
    """Load bbc count matrices, apply phase correction, and aggregate.

    Returns:
        barcodes_df: DataFrame with BARCODE, REP_ID columns.
        seg_df: Segment-level DataFrame with CNP, PROPS columns.
        X_seg, Y_seg, D_seg: Dense int32 count matrices (G_seg, N).
        bbc_df: BBC-level DataFrame with seg_id and CNP columns.
        X_bbc, Y_bbc, D_bbc: Sparse count matrices (G_bbc, N).
    """
    barcodes_df = pd.read_table(
        bc_file, sep="\t", header=None, names=["BARCODE"], dtype=str
    )
    barcodes_df["REP_ID"] = barcodes_df["BARCODE"].str.rsplit("_", n=1).str[-1]

    X_bbc = sparse.load_npz(x_count_file)
    A_bbc = sparse.load_npz(a_allele_file)
    B_bbc = sparse.load_npz(b_allele_file)

    bbc_df = pd.read_table(bbc_file, sep="\t")
    assert X_bbc.shape[0] == len(bbc_df), (
        f"X rows ({X_bbc.shape[0]}) != bbc bins ({len(bbc_df)})"
    )

    # Load BBC phases and merge PHASE column into bbc_df
    phases_df = pd.read_table(bbc_phases_file, sep="\t")
    bbc_df = pd.merge(
        bbc_df,
        phases_df[["#CHR", "START", "END", "PHASE"]],
        on=["#CHR", "START", "END"],
        how="left",
    )
    assert bbc_df["PHASE"].notna().all(), (
        "some BBC blocks have no matching phase in --bbc_phases"
    )

    # Apply phase correction: PHASE=1 -> B-allele is B; PHASE=0 -> B-allele is A (swap)
    phases = bbc_df["PHASE"].to_numpy()[:, None]
    Y_bbc = A_bbc.multiply(1 - phases) + B_bbc.multiply(phases)
    Y_bbc.data = np.rint(Y_bbc.data).astype(np.int32)
    D_bbc = A_bbc + B_bbc

    logging.info(
        f"[{data_type}] phase correction applied: "
        f"{int(bbc_df['PHASE'].sum())}/{len(bbc_df)} BBC blocks flipped"
    )

    # Load segments and apply solfile before any BBC mapping
    seg_df, clones, clone_props = read_seg_ucn_file(seg_ucn_file)
    if "SAMPLE" in seg_df.columns:
        first_sample = seg_df["SAMPLE"].iloc[0]
        seg_df = seg_df[seg_df["SAMPLE"] == first_sample].reset_index(drop=True)
    if solfile is not None:
        seg_df, clones, clone_props = _apply_solfile(seg_df, solfile)

    seg_df, X_sp, Y_sp, D_sp = aggregate_bbc_to_seg(
        bbc_df, seg_df, X_bbc, Y_bbc, D_bbc, data_type=data_type
    )
    X_seg = X_sp.toarray().astype(np.int32)
    Y_seg = Y_sp.toarray().astype(np.int32)
    D_seg = D_sp.toarray().astype(np.int32)

    return barcodes_df, seg_df, X_seg, Y_seg, D_seg, bbc_df, X_bbc, Y_bbc, D_bbc


def union_align_barcodes(data_dict, data_types):
    """Compute union barcodes across modalities and realign matrices.

    Each entry in data_dict has attributes: barcodes (DataFrame with
    BARCODE, REP_ID), X, Y, D (G, N) arrays, T (N,), N (int).
    After alignment, all objects share the same N_union columns.

    Args:
        data_dict: dict mapping data_type -> SX_Data or SimpleNamespace.
        data_types: ordered list of data_type keys.

    Returns:
        union_barcodes_df: DataFrame with BARCODE, REP_ID for union.
        modality_masks: dict[str, ndarray bool (N_union,)].
    """
    # 1. Compute ordered union of barcodes, carrying all columns
    seen = {}
    union_rows = []
    for dt in data_types:
        obj = data_dict[dt]
        for _, row in obj.barcodes.iterrows():
            bc = row["BARCODE"]
            if bc not in seen:
                seen[bc] = len(union_rows)
                union_rows.append(row.to_dict())
    N_union = len(union_rows)
    union_barcodes_df = pd.DataFrame(union_rows)

    # 2. Realign each modality and build masks
    modality_masks = {}
    for dt in data_types:
        obj = data_dict[dt]
        bc_arr = obj.barcodes["BARCODE"].to_numpy()
        N_orig = len(bc_arr)
        mask = np.zeros(N_union, dtype=bool)
        idx_map = np.empty(N_orig, dtype=np.intp)
        for i, bc in enumerate(bc_arr):
            j = seen[bc]
            mask[j] = True
            idx_map[i] = j
        modality_masks[dt] = mask

        if N_orig == N_union and np.all(idx_map == np.arange(N_union)):
            # Already aligned, skip reallocation
            obj.barcodes = union_barcodes_df
            continue

        # Realign (G, N_orig) -> (G, N_union) with zero-fill
        G = obj.X.shape[0]
        X_new = np.zeros((G, N_union), dtype=obj.X.dtype)
        Y_new = np.zeros((G, N_union), dtype=obj.Y.dtype)
        D_new = np.zeros((G, N_union), dtype=obj.D.dtype)
        X_new[:, idx_map] = obj.X
        Y_new[:, idx_map] = obj.Y
        D_new[:, idx_map] = obj.D
        obj.X = X_new
        obj.Y = Y_new
        obj.D = D_new
        obj.T = np.sum(X_new, axis=0)
        obj.N = N_union
        obj.barcodes = union_barcodes_df

    logging.debug(
        f"union_align_barcodes: N_union={N_union}, "
        + ", ".join(f"{dt}={int(modality_masks[dt].sum())}" for dt in data_types)
    )
    return union_barcodes_df, modality_masks


##################################################
# bbc -> segment aggregation
##################################################


def _apply_solfile(seg_df, solfile):
    """Override cn_* and u_* columns in seg_df using a HATCHet solution file.

    Matches rows by CLUSTER ID and rebuilds CNP/PROPS columns.
    """
    sol_df = pd.read_table(solfile, sep="\t")
    if "SAMPLE" in sol_df.columns:
        sol_df = sol_df[sol_df["SAMPLE"] == sol_df["SAMPLE"].iloc[0]]

    cn_cols = [c for c in sol_df.columns if c.startswith("cn_")]
    u_cols = [c for c in sol_df.columns if c.startswith("u_")]
    clones = [c.replace("cn_", "") for c in cn_cols]

    logging.info(
        f"applying solfile: {solfile}, "
        f"{len(cn_cols)} clones ({clones}), {len(sol_df)} clusters"
    )

    # build cluster -> (cn, u) mapping
    sol_map = sol_df.set_index("CLUSTER")[cn_cols + u_cols].to_dict("index")

    # drop old cn_/u_ columns that don't match the solution's clone count
    old_cn = [c for c in seg_df.columns if c.startswith("cn_")]
    old_u = [c for c in seg_df.columns if c.startswith("u_")]
    seg_df = seg_df.drop(columns=old_cn + old_u)

    # add new columns from solution
    for col in cn_cols + u_cols:
        seg_df[col] = seg_df["CLUSTER"].map(
            lambda cid, c=col: sol_map.get(cid, {}).get(c)
        )

    # segments with clusters not in solfile → default diploid (1|1)
    unmapped = seg_df[cn_cols[0]].isna()
    if unmapped.any():
        logging.warning(
            f"{unmapped.sum()} segments have CLUSTER IDs not in solfile, "
            "defaulting to diploid 1|1"
        )
        for col in cn_cols:
            seg_df.loc[unmapped, col] = "1|1"
        for col in u_cols:
            seg_df.loc[unmapped, col] = 0.0

    # rebuild CNP and PROPS
    seg_df["CNP"] = seg_df.apply(
        lambda r: ";".join(str(r[f"cn_{c}"]) for c in clones), axis=1
    )
    clone_props = [float(seg_df[f"u_{c}"].iloc[0]) for c in clones]
    seg_df["PROPS"] = ";".join(str(p) for p in clone_props)
    return seg_df, clones, clone_props


def aggregate_bbc_to_seg(bbc_df, seg_df, X_bbc, Y_bbc, D_bbc, data_type=""):
    """Map bbc-level bins to segments and aggregate count matrices.

    Args:
        bbc_df: DataFrame with bbc-level cnv_segments (#CHR, START, END, bbc_id).
        seg_df: Segment-level DataFrame (already loaded, with solfile applied).
        X_bbc: (G_bbc, N) sparse or dense count matrix (read depth).
        Y_bbc: (G_bbc, N) sparse or dense count matrix (B-allele).
        D_bbc: (G_bbc, N) sparse or dense count matrix (total allele).
        data_type: Modality tag used to prefix log messages (e.g. "gex").

    Returns:
        seg_df: Segment-level DataFrame with CNP, PROPS, seg_id columns.
        X_seg, Y_seg, D_seg: (G_seg, N) aggregated count matrices (sparse).
    """
    tag = f"[{data_type}] " if data_type else ""
    seg_df["seg_id"] = np.arange(len(seg_df))
    n_seg = len(seg_df)
    n_bbc = len(bbc_df)
    logging.info(f"{tag}aggregate_bbc_to_seg: {n_bbc} bbc bins -> {n_seg} segments")

    # map each bbc bin to a segment by coordinate containment
    # bbc bins are subsets of segments, so use midpoint for assignment
    bbc_mid = ((bbc_df["START"].to_numpy() + bbc_df["END"].to_numpy()) / 2).astype(
        np.int64
    )
    bbc_chr = bbc_df["#CHR"].to_numpy()
    seg_ids = np.full(n_bbc, -1, dtype=np.int64)

    for chrom in seg_df["#CHR"].unique():
        bbc_mask = bbc_chr == chrom
        if not bbc_mask.any():
            continue
        seg_chrom = seg_df[seg_df["#CHR"] == chrom].sort_values("START")
        starts = seg_chrom["START"].to_numpy()
        ends = seg_chrom["END"].to_numpy()
        ids = seg_chrom["seg_id"].to_numpy()
        mids = bbc_mid[bbc_mask]

        idx = np.searchsorted(starts, mids, side="right") - 1
        safe_idx = idx.clip(min=0)
        valid = (idx >= 0) & (mids < ends[safe_idx])
        bbc_indices = np.where(bbc_mask)[0]
        seg_ids[bbc_indices[valid]] = ids[idx[valid]]

    n_unmapped = (seg_ids < 0).sum()
    if n_unmapped > 0:
        logging.warning(f"{tag}{n_unmapped}/{n_bbc} bbc bins not mapped to any segment")

    mapped = seg_ids >= 0
    logging.info(f"{tag}mapped {mapped.sum()}/{n_bbc} bbc bins to {n_seg} segments")

    # Store segment mapping and CNP on bbc_df for downstream use
    bbc_df["seg_id"] = seg_ids
    cnp_map = dict(zip(seg_df["seg_id"], seg_df["CNP"]))
    bbc_df["CNP"] = bbc_df["seg_id"].map(cnp_map).fillna("")

    # aggregate counts using one-hot matrix multiplication
    mapped_idx = np.where(mapped)[0]
    mapped_seg_ids = seg_ids[mapped]

    agg_matrix = csr_matrix(
        (np.ones(len(mapped_idx), dtype=np.int8), (mapped_seg_ids, mapped_idx)),
        shape=(n_seg, n_bbc),
    )

    if sparse.issparse(X_bbc):
        X_seg = agg_matrix @ X_bbc
        Y_seg = agg_matrix @ Y_bbc
        D_seg = agg_matrix @ D_bbc
    else:
        X_seg = sparse.csr_matrix(agg_matrix @ X_bbc)
        Y_seg = sparse.csr_matrix(agg_matrix @ Y_bbc)
        D_seg = sparse.csr_matrix(agg_matrix @ D_bbc)

    # count BBC bins per segment
    bbc_per_seg = pd.Series(mapped_seg_ids).value_counts()
    seg_df["#BBC"] = bbc_per_seg.reindex(range(n_seg)).fillna(0).astype(int).values

    # aggregate #SNPS and #gene from bbc if available
    for col in ("#SNPS", "#gene"):
        if col in bbc_df.columns:
            vals = bbc_df[col].to_numpy()
            per_seg = (
                pd.Series(vals[mapped_idx], dtype=int).groupby(mapped_seg_ids).sum()
            )
            seg_df[col] = per_seg.reindex(range(n_seg)).fillna(0).astype(int).values

    logging.info(
        f"{tag}segment-level matrices: "
        f"X={X_seg.shape}, Y={Y_seg.shape}, D={D_seg.shape}"
    )
    return seg_df, X_seg, Y_seg, D_seg


##################################################
# spatial coordinates
##################################################


def load_spatial_neighbors(h5ad_path, n_neighs=6):
    """Load spatial coordinates from h5ad and compute per-rep neighbor graphs.

    Args:
        h5ad_path: path to h5ad file with spatial coordinates in obsm['spatial'].
        n_neighs: number of neighbors (default 6 for Visium hexagonal).

    Returns:
        dict[rep_id -> {"BARCODE": array, "coords": (N,2), "W": sparse}]
    """
    import scanpy as sc
    import squidpy as sq

    adata = sc.read_h5ad(h5ad_path)
    assert "spatial" in adata.obsm, f"no spatial coordinates in {h5ad_path}"

    # extract rep_id from barcode suffix
    rep_ids = np.array([bc.rsplit("_", 1)[-1] for bc in adata.obs_names])
    result = {}
    for rep_id in np.unique(rep_ids):
        mask = rep_ids == rep_id
        adata_rep = adata[mask].copy()
        sq.gr.spatial_neighbors(adata_rep, n_neighs=n_neighs, coord_type="generic")
        W = adata_rep.obsp["spatial_connectivities"]
        row_sums = np.asarray(W.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        W = W.multiply(1.0 / row_sums[:, None])
        result[rep_id] = {
            "BARCODE": adata_rep.obs_names.to_numpy(),
            "coords": adata_rep.obsm["spatial"],
            "W": W,
        }
        logging.info(
            f"spatial neighbors rep={rep_id}: {mask.sum()} spots, "
            f"n_neighs={n_neighs}, edges={W.nnz}"
        )
    return result


##################################################
# copytyping IOs
##################################################


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
