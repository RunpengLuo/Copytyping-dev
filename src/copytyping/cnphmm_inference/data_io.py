import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from copytyping.utils import read_seg_ucn_file, sort_df_chr
from copytyping.io_utils import _apply_solfile


def load_single_cell_data(
    args: dict[str, Any],
    data_types: list[str],
    cell_type_df: pd.DataFrame | None = None,
    celltype_col: str | None = None,
) -> tuple[pd.DataFrame, sparse.csc_matrix, sparse.csc_matrix, sparse.csc_matrix]:
    """Load + hstack per-modality BBC-level counts. Returns
    ``(barcodes_df, X_bbc, B_bbc, C_bbc)``; matrices are (G_bbc, N) sparse CSC,
    with ``C = A + B``. If ``cell_type_df`` / ``celltype_col`` are given, that
    column is left-joined onto ``barcodes_df``.
    """
    bc_parts = []
    X_parts = []
    B_parts = []
    C_parts = []
    for data_type in data_types:
        bc_dt = pd.read_table(
            args[f"{data_type}_barcodes"],
            sep="\t",
            header=None,
            names=["BARCODE"],
            dtype=str,
        )
        bc_dt["REP_ID"] = bc_dt["BARCODE"].str.split("_", n=1).str[1]
        bc_dt["DATA_TYPE"] = data_type

        X_dt = sparse.load_npz(args[f"{data_type}_X_count"])
        A_dt = sparse.load_npz(args[f"{data_type}_A_allele"])
        B_dt = sparse.load_npz(args[f"{data_type}_B_allele"])

        bbc_df = pd.read_table(args[f"{data_type}_cnv_segments"], sep="\t")
        assert X_dt.shape[0] == len(bbc_df), (
            f"[{data_type}] X rows ({X_dt.shape[0]}) != bbc bins ({len(bbc_df)})"
        )

        C_dt = A_dt + B_dt
        bc_parts.append(bc_dt)
        X_parts.append(X_dt)
        B_parts.append(B_dt)
        C_parts.append(C_dt)
        logging.info(
            f"[{data_type}] loaded {X_dt.shape[1]} barcodes, {X_dt.shape[0]} BBC bins"
        )

    barcodes_df = pd.concat(bc_parts, ignore_index=True)

    # optionally merge cell-type annotations (left join keeps row order, so the
    # alignment with the count-matrix columns is preserved)
    if cell_type_df is not None and celltype_col is not None:
        barcodes_df = barcodes_df.merge(
            cell_type_df[["BARCODE", celltype_col]].drop_duplicates("BARCODE"),
            on="BARCODE",
            how="left",
        )
        n_typed = int(barcodes_df[celltype_col].notna().sum())
        logging.info(
            f"merged cell types: {n_typed}/{len(barcodes_df)} barcodes "
            f"labeled by '{celltype_col}'"
        )

    X_bbc = sparse.hstack(X_parts, format="csc")
    B_bbc = sparse.hstack(B_parts, format="csc")
    C_bbc = sparse.hstack(C_parts, format="csc")
    logging.info(
        f"loaded: {len(barcodes_df)} barcodes, "
        f"X={X_bbc.shape}, B={B_bbc.shape}, C={C_bbc.shape}"
    )
    return barcodes_df, X_bbc, B_bbc, C_bbc


def load_bulk_phases(
    bbc_phases_file: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load BBC-level bulk WGS phasing. Returns
    ``(bbc_df, phase_post_bbc, phase_bbc, switchprobs_bbc)``;
    ``phase_bbc`` is 1=keep B, 0=swap A/B; ``phase_post_bbc`` starts equal
    to it.
    """
    bbc_df = pd.read_table(bbc_phases_file, sep="\t")
    bbc_df = sort_df_chr(bbc_df, pos="START")

    phase_bbc = bbc_df["PHASE"].to_numpy().astype(np.int32)
    # posterior phase equals the bulk phase map initially
    phase_post_bbc = phase_bbc.copy()

    switchprobs_bbc = np.zeros(len(bbc_df), dtype=np.float64)
    if "switchprobs" in bbc_df.columns:
        switchprobs_bbc = bbc_df["switchprobs"].to_numpy().astype(np.float64)

    logging.info(
        f"loaded bulk phases: {len(bbc_df)} BBC bins, "
        f"{int(phase_bbc.sum())}/{len(bbc_df)} phase=1"
    )
    return bbc_df, phase_post_bbc, phase_bbc, switchprobs_bbc


def load_bulk_cnp(
    seg_ucn_file: str,
    bbc_df: pd.DataFrame,
    solfile: str | None = None,
    baf_clip: float = 1e-3,
    no_normal: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Map HATCHet seg.ucn.tsv to BBC-level. Returns
    ``(cna_int_states, rdr_baf_states, cna_profile, bulk_segmentation,
    clone_ids)``.

    States are ``(A, B)`` tuples directly — both ``(A, B)`` and ``(B, A)``
    are separate states if either appears in the bulk CNP (no
    major-minor canonicalization, no mirror flag). ``rdr_baf_states[k] =
    (rdr=total/2, baf=B/total clipped to [baf_clip, 1-baf_clip])``.
    ``bulk_segmentation`` IDs BBC bins by identical state-CNP across all
    clones. ``solfile``, if given, overrides the CN profiles.

    ``no_normal=True`` drops the normal clone column before
    ``bulk_segmentation`` is built.
    """
    seg_df, clones, clone_props = read_seg_ucn_file(seg_ucn_file)
    if "SAMPLE" in seg_df.columns:
        first_sample = seg_df["SAMPLE"].iloc[0]
        seg_df = seg_df[seg_df["SAMPLE"] == first_sample].reset_index(drop=True)
    if solfile is not None:
        seg_df, clones, clone_props = _apply_solfile(seg_df, solfile)

    n_clones = len(clones)
    n_seg = len(seg_df)
    n_bbc = len(bbc_df)

    # --- map each BBC bin to a segment by midpoint containment ---
    seg_df["seg_id"] = np.arange(n_seg)
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
        logging.warning(f"{n_unmapped}/{n_bbc} BBC bins not mapped to any segment")

    # --- parse per-segment CN profiles into A, B arrays ---
    A_seg = np.zeros((n_seg, n_clones), dtype=np.int32)
    B_seg = np.zeros((n_seg, n_clones), dtype=np.int32)
    for i, clone in enumerate(clones):
        A_seg[:, i] = (
            seg_df[f"cn_{clone}"].apply(lambda x: int(str(x).split("|")[0])).to_numpy()
        )
        B_seg[:, i] = (
            seg_df[f"cn_{clone}"].apply(lambda x: int(str(x).split("|")[1])).to_numpy()
        )

    # --- collect unique (A, B) pairs directly (orientation-aware) ---
    state_to_idx: dict[tuple[int, int], int] = {}
    for s in range(n_seg):
        for m in range(n_clones):
            key = (int(A_seg[s, m]), int(B_seg[s, m]))
            if key not in state_to_idx:
                state_to_idx[key] = len(state_to_idx)
    n_states = len(state_to_idx)

    cna_int_states = np.zeros((n_states, 2), dtype=np.int32)
    for (a, b), idx in state_to_idx.items():
        cna_int_states[idx] = (a, b)

    # rdr_baf_states[k] = (rdr=total/2, baf=B/total). B/total is the canonical
    # BAF — no mirror flip needed since state encodes orientation directly.
    totals = cna_int_states[:, 0] + cna_int_states[:, 1]
    rdrs = totals / 2.0
    bafs = np.where(
        totals > 0,
        np.clip(cna_int_states[:, 1] / totals, baf_clip, 1 - baf_clip),
        0.5,
    )
    rdr_baf_states = np.column_stack([rdrs, bafs])

    # seg-level state index
    seg_profile = np.zeros((n_seg, n_clones), dtype=np.int32)
    for s in range(n_seg):
        for m in range(n_clones):
            seg_profile[s, m] = state_to_idx[(int(A_seg[s, m]), int(B_seg[s, m]))]

    cna_profile = np.zeros((n_bbc, n_clones), dtype=np.int32)
    mapped = seg_ids >= 0
    cna_profile[mapped] = seg_profile[seg_ids[mapped]]

    clone_ids = ["normal"] + [f"clone{i}" for i in range(1, n_clones)]

    if no_normal:
        assert clone_ids[0] == "normal", clone_ids
        clone_ids = clone_ids[1:]
        cna_profile = cna_profile[:, 1:]
        n_clones -= 1
        logging.info(f"no_normal: dropped normal clone -> {clone_ids}")

    # bulk_segmentation: group BBC bins by identical state CNP across clones.
    unique_cnps: dict[tuple, int] = {}
    seg_counter = 0
    bulk_segmentation = np.zeros(n_bbc, dtype=np.int32)
    for g in range(n_bbc):
        key = tuple(cna_profile[g])
        if key not in unique_cnps:
            unique_cnps[key] = seg_counter
            seg_counter += 1
        bulk_segmentation[g] = unique_cnps[key]

    logging.info(
        f"loaded bulk CNP: {n_states} CN states, {seg_counter} bulk segments, "
        f"{n_clones} clones {clone_ids}"
    )
    return (
        cna_int_states,
        rdr_baf_states,
        cna_profile,
        bulk_segmentation,
        clone_ids,
    )
