import logging

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix

from copytyping.utils import read_seg_ucn_file, sort_df_chr

##################################################
# preprocess IOs
##################################################


def read_cell_types(ct_file: str, req_cols: set):
    if ct_file is None:
        return None
    cell_type_df = pd.read_table(ct_file)
    for req_col in req_cols:
        assert req_col in cell_type_df.columns, (
            f"read_cell_types {ct_file}: {req_col} is missing"
        )
    return cell_type_df


def read_bbc_phases(
    bbc_phases_file: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load BBC-level bulk WGS phasing. Returns
    ``(bbc_df, phase_post_bbc, phase_bbc, switchprobs_bbc)``;
    ``phase_bbc`` is 1=keep B, 0=swap A/B; ``phase_post_bbc`` starts equal
    to it.
    """
    bbc_df = pd.read_table(bbc_phases_file, sep="\t")
    bbc_df = sort_df_chr(bbc_df, pos="START")
    logging.info(f"loaded bulk phases: {len(bbc_df)} BBC bins.")
    return bbc_df


def load_bulk_cnprofile(
    seg_ucn_file: str,
    solfile: str | None = None,
    baf_clip: float = 0.01,
) -> tuple[
    pd.DataFrame,
    list[str],
    list[float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load bulk HATCHet seg.ucn CN profile. Returns ``(seg_df, clones,
    clone_props, cn_A, cn_B, cn_C, BAF)``; keeps the first SAMPLE and applies
    ``solfile`` if given. ``cn_A/cn_B/cn_C/BAF`` are per-segment per-clone copy
    numbers parsed from the CNP column.
    """
    seg_df, clones, clone_props = read_seg_ucn_file(seg_ucn_file)
    if "SAMPLE" in seg_df.columns:
        first_sample = seg_df["SAMPLE"].iloc[0]
        seg_df = seg_df[seg_df["SAMPLE"] == first_sample].reset_index(drop=True)
    if solfile is not None:
        seg_df, clones, clone_props = _apply_solfile(seg_df, solfile)
    _, cn_A, cn_B, cn_C, BAF = parse_cnv_profile(seg_df, baf_clip=baf_clip)
    logging.info(f"loaded bulk CN profile: {len(seg_df)} segments, clones={clones}")
    return seg_df, clones, clone_props, cn_A, cn_B, cn_C, BAF


def load_bbc_modality(
    barcodes_path: str,
    x_count_path: str,
    a_allele_path: str,
    b_allele_path: str,
    cnv_segments_path: str,
    assay_type: str,
    cell_type_df: pd.DataFrame | None = None,
    ref_label: str | None = None,
    exclude_labels: set | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    sparse.csr_matrix,
    sparse.csr_matrix,
    sparse.csr_matrix,
]:
    """Load one modality's BBC-level counts + barcodes; merge cell types and
    drop excluded labels.

    Returns ``(barcodes_df, bbc_df, X_bbc, a_bbc, b_bbc)``: barcodes carry
    BARCODE, REP_ID (+ ref_label); a_bbc/b_bbc are raw pre-phasing alleles.
    """
    barcodes_df = pd.read_table(
        barcodes_path, sep="\t", header=None, names=["BARCODE"], dtype=str
    )
    # REP_ID is everything after the first underscore ("ACGT-1_U1" -> "U1").
    barcodes_df["REP_ID"] = barcodes_df["BARCODE"].str.split("_", n=1).str[1]

    X_bbc = sparse.load_npz(x_count_path)
    a_bbc = sparse.load_npz(a_allele_path)
    b_bbc = sparse.load_npz(b_allele_path)

    bbc_df = pd.read_table(cnv_segments_path, sep="\t")
    assert X_bbc.shape[0] == len(bbc_df), (
        f"X rows ({X_bbc.shape[0]}) != bbc bins ({len(bbc_df)})"
    )

    if cell_type_df is not None and ref_label is not None:
        from copytyping.inference.inference_utils import merge_celltype_into_barcodes

        barcodes_df = merge_celltype_into_barcodes(
            barcodes_df, cell_type_df, ref_label, assay_type
        )
        if exclude_labels:
            barcodes_df, X_bbc, a_bbc, b_bbc = exclude_barcodes(
                barcodes_df, exclude_labels, ref_label, X_bbc, a_bbc, b_bbc
            )
    return barcodes_df, bbc_df, X_bbc, a_bbc, b_bbc


def apply_bbc_phasing(
    bbc_df: pd.DataFrame,
    bbc_phases: pd.DataFrame,
    a_bbc: sparse.csr_matrix,
    b_bbc: sparse.csr_matrix,
    assay_type: str = "",
) -> tuple[pd.DataFrame, sparse.csr_matrix, sparse.csr_matrix]:
    """Merge bulk PHASE into bbc_df and phase-correct allele counts.

    Returns ``(bbc_df, B_bbc, N_bbc)``: B_bbc = phase-corrected B-allele
    (PHASE=1 keeps B, PHASE=0 swaps A/B); N_bbc = total A+B.
    """
    bbc_df = pd.merge(
        bbc_df,
        bbc_phases[["#CHR", "START", "END", "PHASE"]],
        on=["#CHR", "START", "END"],
        how="left",
    )
    assert bbc_df["PHASE"].notna().all(), (
        "some BBC blocks have no matching phase in --bbc_phases"
    )

    phases = bbc_df["PHASE"].to_numpy()[:, None]
    B_bbc = a_bbc.multiply(1 - phases) + b_bbc.multiply(phases)
    B_bbc.data = np.rint(B_bbc.data).astype(np.int32)
    N_bbc = a_bbc + b_bbc

    tag = f"[{assay_type}] " if assay_type else ""
    logging.info(
        f"{tag}phase correction applied: "
        f"{int(bbc_df['PHASE'].sum())}/{len(bbc_df)} BBC blocks flipped"
    )
    return bbc_df, B_bbc, N_bbc


def exclude_barcodes(barcodes_df, exclude_labels, ref_label, *mats):
    """Drop barcodes whose ref_label is in exclude_labels and column-subset
    each aligned matrix in ``mats``.

    Returns ``(barcodes_df, *mats)`` filtered to kept cells.
    """
    if ref_label not in barcodes_df.columns:
        return (barcodes_df, *mats)
    keep = ~barcodes_df[ref_label].isin(exclude_labels)
    n_excl = int((~keep).sum())
    if n_excl == 0:
        return (barcodes_df, *mats)
    idx = np.where(keep.values)[0]
    barcodes_df = barcodes_df[keep].reset_index(drop=True)
    mats = tuple(m[:, idx] for m in mats)
    logging.info(
        f"excluded {n_excl} cells with ref_label in {exclude_labels}, "
        f"{len(barcodes_df)} remaining"
    )
    return (barcodes_df, *mats)


def union_align_barcodes(data_dict, assay_types):
    """Compute union barcodes across modalities and realign matrices.

    Each entry in data_dict has attributes: barcodes (DataFrame with
    BARCODE, REP_ID), X, Y, D (G, N) arrays, T (N,), N (int).
    After alignment, all objects share the same N_union columns.

    Args:
        data_dict: dict mapping assay_type -> SX_Data or SimpleNamespace.
        assay_types: ordered list of assay_type keys.

    Returns:
        union_barcodes_df: DataFrame with BARCODE, REP_ID for union.
        modality_masks: dict[str, ndarray bool (N_union,)].
    """
    # 1. Compute ordered union of barcodes, carrying all columns
    seen = {}
    union_rows = []
    for assay in assay_types:
        obj = data_dict[assay]
        for _, row in obj.barcodes.iterrows():
            bc = row["BARCODE"]
            if bc not in seen:
                seen[bc] = len(union_rows)
                union_rows.append(row.to_dict())
    N_union = len(union_rows)
    union_barcodes_df = pd.DataFrame(union_rows)

    # 2. Realign each modality and build masks
    modality_masks = {}
    for assay in assay_types:
        obj = data_dict[assay]
        bc_arr = obj.barcodes["BARCODE"].to_numpy()
        N_orig = len(bc_arr)
        mask = np.zeros(N_union, dtype=bool)
        idx_map = np.empty(N_orig, dtype=np.intp)
        for i, bc in enumerate(bc_arr):
            j = seen[bc]
            mask[j] = True
            idx_map[i] = j
        modality_masks[assay] = mask

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
        + ", ".join(
            f"{assay}={int(modality_masks[assay].sum())}" for assay in assay_types
        )
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


def aggregate_bbc_to_seg(bbc_df, seg_df, X_bbc, B_bbc, N_bbc, assay_type=""):
    """Map bbc-level bins to segments and aggregate count matrices.

    Args:
        bbc_df: DataFrame with bbc-level cnv_segments (#CHR, START, END, bbc_id).
        seg_df: Segment-level DataFrame (already loaded, with solfile applied).
        X_bbc: (G_bbc, N) sparse or dense count matrix (read depth).
        B_bbc: (G_bbc, N) sparse or dense count matrix (B-allele).
        N_bbc: (G_bbc, N) sparse or dense count matrix (total allele).
        assay_type: Modality tag used to prefix log messages (e.g. "gex").

    Returns:
        seg_df: Segment-level DataFrame with CNP, PROPS, seg_id, LENGTH columns
            (a copy; the input bulk seg_df is left unmutated).
        X_seg, B_seg, N_seg: (G_seg, N) dense int32 aggregated count matrices.
    """
    tag = f"[{assay_type}] " if assay_type else ""
    seg_df = seg_df.copy()  # don't mutate the shared bulk seg_df across modalities
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

    X_seg = (agg_matrix @ X_bbc).toarray().astype(np.int32)
    B_seg = (agg_matrix @ B_bbc).toarray().astype(np.int32)
    N_seg = (agg_matrix @ N_bbc).toarray().astype(np.int32)
    seg_df["LENGTH"] = seg_df["END"] - seg_df["START"]

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
        f"X={X_seg.shape}, Y={B_seg.shape}, D={N_seg.shape}"
    )
    return seg_df, X_seg, B_seg, N_seg


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

    # extract rep_id from barcode suffix (everything after first underscore)
    rep_ids = np.array([bc.split("_", 1)[1] for bc in adata.obs_names])
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


def build_spatial_graphs(
    assay_types: list[str],
    h5ad_paths: dict[str, str | None],
    n_neighs: int,
) -> dict[str, dict]:
    """Build per-modality spatial neighbor graphs from h5ad spatial coords.

    Skips assays without an h5ad or without ``obsm['spatial']``. Each value is
    a per-rep graph dict from ``load_spatial_neighbors``.
    """
    import scanpy as sc

    spatial_graphs = {}
    for assay_type in assay_types:
        h5ad_path = h5ad_paths[assay_type]
        if h5ad_path is not None:
            adata = sc.read_h5ad(h5ad_path)
            if "spatial" in adata.obsm:
                spatial_graphs[assay_type] = load_spatial_neighbors(
                    h5ad_path, n_neighs=n_neighs
                )
    return spatial_graphs


##################################################
# copytyping IOs
##################################################


def parse_cnv_profile(haplo_blocks: pd.DataFrame, baf_clip=0.01):
    num_clones = len(str(haplo_blocks["CNP"].iloc[0]).split(";"))
    clones = ["normal"] + [f"clone{i}" for i in range(1, num_clones)]
    cn_A = np.zeros((len(haplo_blocks), num_clones), dtype=np.int32)
    cn_B = np.zeros((len(haplo_blocks), num_clones), dtype=np.int32)
    for i in range(num_clones):
        cn_A[:, i] = haplo_blocks.apply(
            func=lambda r: int(r["CNP"].split(";")[i].split("|")[0]), axis=1
        ).to_numpy()
        cn_B[:, i] = haplo_blocks.apply(
            func=lambda r: int(r["CNP"].split(";")[i].split("|")[1]), axis=1
        ).to_numpy()
    cn_C = cn_A + cn_B
    BAF = np.divide(
        cn_B,
        cn_C,
        out=np.zeros_like(cn_C, dtype=np.float32),
        where=(cn_C > 0),
    )
    BAF = np.clip(BAF, baf_clip, 1 - baf_clip)

    # assign the CNP group id
    return clones, cn_A, cn_B, cn_C, BAF
