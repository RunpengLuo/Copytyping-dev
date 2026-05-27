"""Reference-cell detection, per-seg baseline, spot purity (TODO), and the
``initialize_copytyping`` top wrapper. Segmentation primitives moved to
``segmentation.py``.
"""

import logging
from typing import Any

import numpy as np
from scipy import sparse

from copytyping.joint_inference.optimize import block_coordinate_ascent_fixed_cnp
from copytyping.joint_inference.tmp_funcs import get_masks_from_cna_profile


# =========================== reference-cell detection ========================


def estimate_reference_cells(
    B_seg: sparse.csr_matrix,
    C_seg: sparse.csr_matrix,
    X_seg: sparse.csr_matrix,
    T_seg: np.ndarray,
    cna_profile_seg: np.ndarray,
    cna_mirrored_seg: np.ndarray,
    rdr_baf_states: np.ndarray,
    clone_ids: list[str],
    model_params: dict[str, Any],
    em_kwargs: dict[str, Any],
) -> tuple[np.ndarray, int, np.ndarray]:
    """BB-only EM to pick the per-dataset reference-cell set. Has-normal
    case: collapse to (normal, one tumor) over ``CLONAL_IMBALANCED`` segs,
    ref_clone=0. No-normal case: full tumor-clone profile over
    ``IMBALANCED`` segs, ref_clone=argmax(label counts).

    Reads ``masks`` and ``rdr_baf_params`` from ``model_params`` and
    forwards the dict to ``block_coordinate_ascent_fixed_cnp`` (which
    mutates ``rdr_baf_params`` in place via the τ MLE and writes a
    transient BB-only ``pi`` that the next EM pass overwrites).

    Returns ``(labels_init, ref_clone, ref_cells)``.
    """
    n_cells = X_seg.shape[1]
    n_clones = cna_profile_seg.shape[1]
    has_normal = clone_ids[0] == "normal"
    masks = model_params["masks"]

    if has_normal:
        tumor_col = 1 if n_clones > 1 else 0
        profile_used = cna_profile_seg[:, [0, tumor_col]]
        mirror_used = cna_mirrored_seg[:, [0, tumor_col]]
        bb_mask = masks["CLONAL_IMBALANCED"]
        mask_label = "CLONAL_IMBALANCED"
    else:
        profile_used = cna_profile_seg
        mirror_used = cna_mirrored_seg
        bb_mask = masks["IMBALANCED"]
        mask_label = "IMBALANCED"

    logging.info(
        f"estimate_reference_cells: BB-only EM on {int(bb_mask.sum())} "
        f"{mask_label} segs over {profile_used.shape[1]} clones"
    )
    labels_init, _ = block_coordinate_ascent_fixed_cnp(
        B_seg,
        C_seg,
        X_seg,
        T_seg,
        profile_used,
        mirror_used,
        rdr_baf_states,
        model_params,
        clone_norm=None,
        bb_mask=bb_mask,
        nb_mask=None,
        em_kwargs=em_kwargs,
    )

    if has_normal:
        ref_clone = 0  # normal clone (1|1 everywhere)
    else:
        counts = np.bincount(labels_init, minlength=n_clones)
        ref_clone = int(counts.argmax())
    ref_cells = np.where(labels_init == ref_clone)[0]
    logging.info(
        f"estimate_reference_cells: ref_clone={clone_ids[ref_clone]} "
        f"(idx={ref_clone}), {ref_cells.size}/{n_cells} ref cells"
    )
    return labels_init, ref_clone, ref_cells


# ================================== baseline ==================================


def estimate_baseline(
    X_seg: sparse.csr_matrix,
    clone_cells: np.ndarray,
    clone_cnp: np.ndarray,
    rdr_baf_states: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-seg read-depth baseline (sums to 1) from a single-clone pseudobulk:

        lambda_g = (sum_{n in C_m} x_{n,g} / mu_{m,g}) / (sum over g'),

    where ``mu_{m,g} = rdr_{m,g}`` is clone m's per-seg RDR.
    """
    G = X_seg.shape[0]
    X_clone = np.asarray(X_seg[:, clone_cells].sum(axis=1)).ravel().astype(np.float64)
    mu_g = rdr_baf_states[clone_cnp, 0]  # (G,) clone's per-seg RDR
    base = X_clone / np.clip(mu_g, eps, None)
    total = base.sum()
    return base / total if total > 0 else np.ones(G, dtype=np.float64) / G


def get_clone_norm(
    base_props: np.ndarray,
    state_idx: np.ndarray,
    rdr_baf_states: np.ndarray,
) -> np.ndarray:
    """Genome-wide per-clone RDR normalizer S_m = sum_g base_props[g] * rdr[g, m]."""
    rdr_gm = rdr_baf_states[state_idx, 0]  # (G, M) rdr per bin per clone
    return (base_props[:, None] * rdr_gm).sum(axis=0)  # (M,)


# ================================ spot purity =================================


def estimate_spot_purity(
    X_seg: sparse.csr_matrix,
    B_seg: sparse.csr_matrix,
    C_seg: sparse.csr_matrix,
    T_seg: np.ndarray,
    cna_profile_seg: np.ndarray,
    cna_mirrored_seg: np.ndarray,
    rdr_baf_states: np.ndarray,
    base_props: np.ndarray,
    clone_norm: np.ndarray,
) -> np.ndarray | None:
    """Per-spot tumor purity θ_n ∈ [0, 1]. TODO: not implemented; returns None,
    which makes the spot-model emissions fall back to single-cell (θ ≡ 1).
    """
    logging.warning(
        "estimate_spot_purity: not implemented; falling back to single-cell "
        "emissions (θ ≡ 1)"
    )
    return None


# ================================ top wrapper =================================


def initialize_copytyping(
    X_seg: sparse.csr_matrix,
    B_seg: sparse.csr_matrix,
    C_seg: sparse.csr_matrix,
    T_seg: np.ndarray,
    cna_int_states: np.ndarray,
    cna_profile_seg: np.ndarray,
    cna_mirrored_seg: np.ndarray,
    rdr_baf_states: np.ndarray,
    clone_ids: list[str],
    em_kwargs: dict[str, Any],
) -> tuple[np.ndarray, int, dict[str, Any]]:
    assert len(clone_ids) == cna_profile_seg.shape[1], (
        len(clone_ids),
        cna_profile_seg.shape[1],
    )
    n_states = len(rdr_baf_states)

    model_params: dict[str, Any] = {
        "rdr_baf_params": np.column_stack(
            [
                np.full(n_states, em_kwargs["max_invphi"]),
                np.full(n_states, em_kwargs["max_tau"]),
            ]
        ),
        "masks": get_masks_from_cna_profile(
            cna_int_states, cna_profile_seg, cna_mirrored_seg
        ),
        "base_props": None,
        "spot_purities": None,
    }

    labels_init, ref_clone, ref_cells = estimate_reference_cells(
        B_seg,
        C_seg,
        X_seg,
        T_seg,
        cna_profile_seg,
        cna_mirrored_seg,
        rdr_baf_states,
        clone_ids,
        model_params,
        em_kwargs,
    )

    model_params["base_props"] = estimate_baseline(
        X_seg,
        ref_cells,
        cna_profile_seg[:, ref_clone],
        rdr_baf_states,
    )

    if em_kwargs["is_spot"]:
        model_params["spot_purities"] = estimate_spot_purity(
            X_seg,
            B_seg,
            C_seg,
            T_seg,
            cna_profile_seg,
            cna_mirrored_seg,
            rdr_baf_states,
            model_params["base_props"],
            get_clone_norm(model_params["base_props"], cna_profile_seg, rdr_baf_states),
        )

    # ``labels_init`` and ``ref_clone`` are not model parameters — they're
    # inference outputs of the BB-only seeding pass. They travel back to the
    # caller via the explicit triple return below.
    return labels_init, ref_clone, model_params
