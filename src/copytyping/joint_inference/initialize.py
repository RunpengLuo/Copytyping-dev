"""Everything that runs before the clone-EM (``bulk_cnp_copytyping``):
adaptive segmentation, reference-cell detection, baseline, spot purity (TODO),
and the ``initialize_bulk_cnp_copytyping`` top wrapper.
"""

import logging

import numpy as np
import pandas as pd
from scipy import sparse

from copytyping.joint_inference.optimize import block_coordinate_ascent_fixed_cnp


# ============================== pre-segmentation ==============================


def adaptive_segmentation(
    bbc_df,
    C_bbc,
    segmentation_prior,
    bulk_phased=None,
    min_snp_count=300,
    max_bin_length=5_000_000,
):
    """Merge adjacent BBC bins per chromosome until each segment reaches
    ``min_snp_count`` pseudobulk allele count or ``max_bin_length`` bp.

    Splits at ``segmentation_prior`` boundaries, and (if given) at ``bulk_phased``
    boundaries so confidently-phased bins are never merged with unphased ones.

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
        cur_phased = bulk_phased[order[0]] if bulk_phased is not None else None

        for i in range(1, len(order)):
            bi = order[i]
            same_seg = segmentation_prior[bi] == cur_seg
            fits_length = (ends[bi] - cur_start) <= max_bin_length
            needs_more = cur_c < min_snp_count
            if bulk_phased is not None:
                same_phased = bulk_phased[bi] == cur_phased
            else:
                same_phased = True

            if same_seg and fits_length and needs_more and same_phased:
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
                cur_phased = bulk_phased[bi] if bulk_phased is not None else None

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
    agg_mat,
    X_bbc,
    B_bbc,
    C_bbc,
    cna_profile,
    cna_mirrored,
    bulk_phased,
    phase_map_bbc,
):
    """Apply bulk phase correction and aggregate BBC-level matrices to segments.

    Phase correction on bulk_phased bins: ``B_corr = B`` if phase=1 else
    ``A = C - B``; non-phased bins keep B as-is. Per-segment CN attributes
    (``cna_profile_seg``, ``cna_mirrored_seg``, ``bulk_phased_seg``) are taken
    from the first BBC in each group.

    Returns ``(X_seg, B_seg, C_seg, cna_profile_seg, cna_mirrored_seg,
    bulk_phased_seg)``; the count matrices are (G_seg, N) sparse CSR.
    """
    B = B_bbc.tocsr() if sparse.issparse(B_bbc) else sparse.csr_matrix(B_bbc)
    C = C_bbc.tocsr() if sparse.issparse(C_bbc) else sparse.csr_matrix(C_bbc)
    X = X_bbc.tocsr() if sparse.issparse(X_bbc) else sparse.csr_matrix(X_bbc)

    # phase correction at BBC level
    # B_corr = phase-corrected B-allele: for bulk_phased bins, apply phase_map;
    # for non-bulk_phased bins, keep B as-is
    phases = phase_map_bbc.copy().astype(np.float64)
    phases[~bulk_phased] = 1.0  # no correction for diploid bins
    phases = phases[:, None]  # (G_bbc, 1) for broadcasting

    # B_corr = B * phase + (C - B) * (1 - phase)
    #        = B * phase + A * (1 - phase)
    # when phase=1: B_corr = B; when phase=0: B_corr = A = C - B
    B_corr = B.multiply(phases) + (C - B).multiply(1 - phases)

    # aggregate
    X_seg = agg_mat @ X
    B_seg = agg_mat @ B_corr
    C_seg = agg_mat @ C

    # per-segment attributes: pick from the first BBC in each group
    n_seg = agg_mat.shape[0]
    first_bbc_per_group = np.array([agg_mat[g].indices[0] for g in range(n_seg)])
    cna_profile_seg = cna_profile[first_bbc_per_group]
    cna_mirrored_seg = cna_mirrored[first_bbc_per_group]
    bulk_phased_seg = bulk_phased[first_bbc_per_group]

    logging.info(
        f"perform_segmentation: X={X_seg.shape}, B={B_seg.shape}, C={C_seg.shape}, "
        f"cna_profile={cna_profile_seg.shape}, "
        f"bulk_phased={int(bulk_phased_seg.sum())}/{n_seg}"
    )
    return X_seg, B_seg, C_seg, cna_profile_seg, cna_mirrored_seg, bulk_phased_seg


# =========================== reference-cell detection ========================


def estimate_normal_cells(
    B_seg,
    C_seg,
    X_seg,
    T_seg,
    cna_profile_seg,
    cna_mirrored_seg,
    rdr_baf_states,
    rdr_baf_params,
    masks,
    em_kwargs,
):
    """BB-only EM over CLONAL_IMBALANCED segs splitting normal (0) from tumor (1).

    Profile is collapsed to one normal + one representative tumor column (tumor
    clones agree on CLONAL_IMBALANCED segs). Returns ``(labels_init, normal_cells)``.
    ``rdr_baf_params`` is mutated in place by the BB tau MLE.
    """
    n_cells = X_seg.shape[1]
    n_clones_all = cna_profile_seg.shape[1]
    tumor_col = 1 if n_clones_all > 1 else 0
    profile_nt = cna_profile_seg[:, [0, tumor_col]]
    mirror_nt = cna_mirrored_seg[:, [0, tumor_col]]
    logging.info(
        f"estimate_normal_cells: BB-only EM on "
        f"{int(masks['CLONAL_IMBALANCED'].sum())} clonal imbalanced segs"
    )
    labels_init, _, _ = block_coordinate_ascent_fixed_cnp(
        B_seg,
        C_seg,
        X_seg,
        T_seg,
        profile_nt,
        mirror_nt,
        rdr_baf_states,
        rdr_baf_params,
        base_props=None,
        clone_norm=None,
        bb_mask=masks["CLONAL_IMBALANCED"],
        nb_mask=None,
        **em_kwargs,
    )
    normal_cells = np.where(labels_init == 0)[0]
    logging.info(
        f"estimate_normal_cells: {normal_cells.size}/{n_cells} normal, "
        f"{n_cells - normal_cells.size}/{n_cells} tumor"
    )
    return labels_init, normal_cells


def estimate_major_clone_cells(
    B_seg,
    C_seg,
    X_seg,
    T_seg,
    cna_profile_seg,
    cna_mirrored_seg,
    rdr_baf_states,
    rdr_baf_params,
    masks,
    clone_ids,
    em_kwargs,
):
    """BB-only EM over IMBALANCED segs across all tumor clones (no-normal samples).

    Picks the most-assigned clone as the "major" clone. Returns
    ``(labels_init, ref_cells, major_clone)``. ``rdr_baf_params`` is mutated in
    place by the BB tau MLE.
    """
    n_cells = X_seg.shape[1]
    n_clones = cna_profile_seg.shape[1]
    logging.info(
        f"estimate_major_clone_cells: BB-only EM on "
        f"{int(masks['IMBALANCED'].sum())} imbalanced segs over {n_clones} tumor clones"
    )
    labels_init, _, _ = block_coordinate_ascent_fixed_cnp(
        B_seg,
        C_seg,
        X_seg,
        T_seg,
        cna_profile_seg,
        cna_mirrored_seg,
        rdr_baf_states,
        rdr_baf_params,
        base_props=None,
        clone_norm=None,
        bb_mask=masks["IMBALANCED"],
        nb_mask=None,
        **em_kwargs,
    )
    counts = np.bincount(labels_init, minlength=n_clones)
    major_clone = int(counts.argmax())
    ref_cells = np.where(labels_init == major_clone)[0]
    logging.info(
        f"estimate_major_clone_cells: counts={counts.tolist()}, "
        f"major={clone_ids[major_clone]} ({counts[major_clone]}/{n_cells} cells)"
    )
    return labels_init, ref_cells, major_clone


# ================================== baseline ==================================


def estimate_baseline(
    X_seg,
    clone_cells,
    clone_cnp,
    rdr_baf_states,
    eps=1e-12,
):
    """Per-seg read-depth baseline (sums to 1) from a single-clone pseudobulk.

    Method-of-moments with mu_{m,g} = rdr_{m,g} (clone m's per-seg RDR from
    ``clone_cnp``):

        lambda_g = (sum_{n in C_m} x_{n,g} / mu_{m,g}) / (sum over g')
    """
    G = X_seg.shape[0]
    X_clone = np.asarray(X_seg[:, clone_cells].sum(axis=1)).ravel().astype(np.float64)
    mu_g = rdr_baf_states[clone_cnp, 0]  # (G,) clone's per-seg RDR
    base = X_clone / np.clip(mu_g, eps, None)
    total = base.sum()
    return base / total if total > 0 else np.ones(G, dtype=np.float64) / G


def get_clone_norm(base_props, state_idx, rdr_baf_states):
    """Genome-wide per-clone RDR normalizer S_m = sum_g base_props[g] * rdr[g, m]."""
    rdr_gm = rdr_baf_states[state_idx, 0]  # (G, M) rdr per bin per clone
    return (base_props[:, None] * rdr_gm).sum(axis=0)  # (M,)


# ================================ spot purity =================================


def estimate_spot_purity(
    X_seg,
    B_seg,
    C_seg,
    T_seg,
    cna_profile_seg,
    cna_mirrored_seg,
    rdr_baf_states,
    base_props,
    clone_norm,
):
    """Per-spot tumor purity θ_n ∈ [0, 1]. TODO: not implemented; returns None,
    which makes the spot-model emissions fall back to single-cell (θ ≡ 1).
    """
    logging.warning(
        "estimate_spot_purity: not implemented; falling back to single-cell "
        "emissions (θ ≡ 1)"
    )
    return None


# ================================ top wrapper =================================


def initialize_bulk_cnp_copytyping(
    X_seg,
    B_seg,
    C_seg,
    T_seg,
    cna_profile_seg,
    cna_mirrored_seg,
    rdr_baf_states,
    rdr_baf_params,
    masks,
    clone_ids,
    is_spot=False,
    em_kwargs=None,
):
    """Reference-cell detection -> baseline -> (spatial) spot purity.

    ``masks`` is the bin-level CN-state mask dict from
    ``get_masks_from_cna_profile``, computed once by the caller and shared
    downstream. Returns a dict with:
        rdr_baf_params: (S, 2) (invphi, tau); a mutated copy of the input.
        base_props:     (G,) per-seg baseline (sums to 1).
        spot_purities:  (N,) θ_n, or None for single-cell.
        labels_init:    (N,) reference-cell EM labels — normal(0)/tumor(1) if
                        a normal clone is present, else per-tumor-clone.
        ref_clone:      int index of the baseline reference clone.
    """
    em_kwargs = em_kwargs or {}
    assert len(clone_ids) == cna_profile_seg.shape[1], (
        len(clone_ids),
        cna_profile_seg.shape[1],
    )
    has_normal = clone_ids[0] == "normal"
    rdr_baf_params = rdr_baf_params.copy()

    # 1) + 2) reference cells -> per-seg baseline
    if has_normal:
        labels_init, ref_cells = estimate_normal_cells(
            B_seg,
            C_seg,
            X_seg,
            T_seg,
            cna_profile_seg,
            cna_mirrored_seg,
            rdr_baf_states,
            rdr_baf_params,
            masks,
            em_kwargs,
        )
        ref_clone = 0  # normal clone (1|1 everywhere)
    else:
        labels_init, ref_cells, ref_clone = estimate_major_clone_cells(
            B_seg,
            C_seg,
            X_seg,
            T_seg,
            cna_profile_seg,
            cna_mirrored_seg,
            rdr_baf_states,
            rdr_baf_params,
            masks,
            clone_ids,
            em_kwargs,
        )
    base_props = estimate_baseline(
        X_seg,
        ref_cells,
        cna_profile_seg[:, ref_clone],
        rdr_baf_states,
    )

    # 3) spot purity (spatial only). ``estimate_spot_purity`` needs the
    # per-clone S_m, so compute it just for this call — we do not store it.
    spot_purities = None
    if is_spot:
        spot_purities = estimate_spot_purity(
            X_seg,
            B_seg,
            C_seg,
            T_seg,
            cna_profile_seg,
            cna_mirrored_seg,
            rdr_baf_states,
            base_props,
            get_clone_norm(base_props, cna_profile_seg, rdr_baf_states),
        )

    return {
        "rdr_baf_params": rdr_baf_params,
        "base_props": base_props,
        "spot_purities": spot_purities,
        "labels_init": labels_init,
        "ref_clone": ref_clone,
    }
