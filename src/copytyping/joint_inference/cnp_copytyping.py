"""Bulk-CNP-fixed clone-EM. Pure EM — caller supplies ``model_params`` from
``initialize_copytyping`` and an ``em_kwargs`` config bundle.
"""

import logging
from typing import Any

import numpy as np
from scipy import sparse

from copytyping.joint_inference.initialize import get_clone_norm
from copytyping.joint_inference.optimize import block_coordinate_ascent_fixed_cnp


def cnp_copytyping(
    X_seg: sparse.csr_matrix,
    B_seg: sparse.csr_matrix,
    C_seg: sparse.csr_matrix,
    T_seg: np.ndarray,
    cna_profile_seg: np.ndarray,
    cna_mirrored_seg: np.ndarray,
    rdr_baf_states: np.ndarray,
    clone_ids: list[str],
    phase: np.ndarray,
    model_params: dict[str, Any],
    em_kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """copy-typing with fixed CNPs.

    ``phase`` is the per-seg phase chain (int8, length G_seg). ``phase[g]=1``
    means "B in its bulk-corrected orientation"; ``phase[g]=0`` means "swap
    A/B at seg g". The decision is folded into ``cna_mirrored_seg`` via XOR
    before the EM call (no other changes to the emission math).

    Returns ``(labels, resp)``: MAP per-cell clone assignments (N,) and the
    posterior responsibility matrix (N, M). Every other model-side quantity
    lives on the caller-owned ``model_params``.
    """
    n_clones = cna_profile_seg.shape[1]
    fit_mode = em_kwargs["fit_mode"]
    base_props = model_params["base_props"]
    masks = model_params["masks"]

    clone_norm = get_clone_norm(base_props, cna_profile_seg, rdr_baf_states)
    bb_mask = masks["IMBALANCED"] if fit_mode in ("hybrid", "allele_only") else None
    nb_mask = masks["ANEUPLOID"] if fit_mode in ("hybrid", "total_only") else None
    bb_n = int(bb_mask.sum()) if bb_mask is not None else 0
    nb_n = int(nb_mask.sum()) if nb_mask is not None else 0
    logging.info(
        f"clone EM (fit_mode={fit_mode}, n_clones={n_clones}): "
        f"BB on {bb_n} imbalanced, NB on {nb_n} aneuploid segs"
    )
    # fold phase into cna_mirrored: phase=0 means swap A/B at this seg, i.e.
    # flip the mirror flag across all clones at that row.
    swap = (1 - phase).astype(cna_mirrored_seg.dtype)[:, None]
    effective_mirrored = cna_mirrored_seg ^ swap
    labels, resp = block_coordinate_ascent_fixed_cnp(
        B_seg,
        C_seg,
        X_seg,
        T_seg,
        cna_profile_seg,
        effective_mirrored,
        rdr_baf_states,
        model_params,
        clone_norm=clone_norm,
        bb_mask=bb_mask,
        nb_mask=nb_mask,
        em_kwargs=em_kwargs,
    )

    pi = model_params["pi"]
    logging.info(
        f"final: pi=[{', '.join(f'{p:.3f}' for p in pi)}], "
        f"labels={np.bincount(labels, minlength=n_clones).tolist()}, clone_ids={clone_ids}"
    )
    return labels, resp
