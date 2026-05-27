import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import gammaln

from copytyping.joint_inference.cnp_copytyping import cnp_copytyping
from copytyping.joint_inference.emissions import (
    bb_logpmf_nz,
    nb_logpmf_nz_adjustment,
    nb_logpmf_zero,
)
from copytyping.joint_inference.initialize import get_clone_norm
from copytyping.joint_inference.segmentation import derive_cnp_segments
from copytyping.joint_inference.tmp_funcs import (
    get_masks_from_cna_profile,
    plot_clone_rdr_baf,
    plot_cnp_segments_nll_probs,
)


def bulk_cnp_anchored_copytyping(
    X: sparse.csr_matrix,
    B: sparse.csr_matrix,
    C: sparse.csr_matrix,
    T: np.ndarray,
    genome_coords_seg: pd.DataFrame,
    cna_int_states: np.ndarray,
    cna_profile: np.ndarray,
    cna_mirrored: np.ndarray,
    rdr_baf_states: np.ndarray,
    clone_ids: list[str],
    phase: np.ndarray,
    switchprobs: np.ndarray,
    model_params: dict[str, Any],
    em_kwargs: dict[str, Any],
    workdir: str,
    region_bed: str,
    sample: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, list[int]]:
    max_clones = em_kwargs["max_clones"]
    anchored_tol = em_kwargs["anchored_tol"]
    rng = np.random.default_rng(em_kwargs["rng_seed"])
    _, m_0 = cna_profile.shape
    clone_ids = list(clone_ids)
    parent_map = [0] * m_0
    os.makedirs(workdir, exist_ok=True)

    clone_labels, _resp = cnp_copytyping(
        X,
        B,
        C,
        T,
        cna_profile,
        cna_mirrored,
        rdr_baf_states,
        clone_ids,
        phase=phase,
        model_params=model_params,
        em_kwargs=em_kwargs,
    )

    for m in range(m_0 + 1, max_clones + 1):
        logging.info(f"bulk-anchored iter: proposing clone {m} (current M={m - 1})")

        seg_inds, cnp_segments_df = derive_cnp_segments(
            genome_coords_seg, cna_profile, cna_mirrored, cna_int_states, clone_ids
        )
        logging.info(
            f"  {cnp_segments_df.shape[0]} candidate segs from {seg_inds.size} bin-segs"
        )

        sampling_probs, nll_cell_by_seg, cnp_segments_df = compute_sampling_probs_seg(
            X,
            B,
            C,
            T,
            cna_profile,
            cna_mirrored,
            phase,
            rdr_baf_states,
            model_params,
            em_kwargs,
            clone_labels,
            seg_inds,
            cnp_segments_df,
            clone_ids,
        )
        cnp_segments_df.to_csv(
            os.path.join(workdir, f"cnp_segments.iter{m}.tsv"), sep="\t", index=False
        )
        plot_cnp_segments_nll_probs(
            nll_cell_by_seg,
            sampling_probs,
            clone_labels,
            cnp_segments_df,
            clone_ids,
            region_bed,
            out_path=os.path.join(workdir, f"cnp_segments.iter{m}.pdf"),
            sample=sample,
        )
        plot_clone_rdr_baf(
            genome_coords_seg,
            X,
            B,
            C,
            T,
            clone_labels,
            model_params["base_props"],
            cnp_segments_df,
            None,
            region_bed=region_bed,
            sample=sample,
            out_dir=workdir,
            out_prefix=f"cnp_segments.iter{m}",
            clone_names=clone_ids,
        )

        # candidate seg indices ranked by total NLL (descending), top-K.
        sorted_seg_inds = np.argsort(cnp_segments_df["total_NLL"].to_numpy())[::-1]
        for seg_idx in sorted_seg_inds[: em_kwargs["top_segments"]]:
            logging.info(cnp_segments_df.iloc[seg_idx])
            # TODO Phases 3-4: propose (k_tilde, h_tilde, parent) at seg_idx,
            # score Δ vs anchored objective, update `result` if best so far.
            pass
        raise ValueError("TODO")

        if result is None or result[-1] <= anchored_tol:
            logging.info(
                f"  no split beats anchored_tol={anchored_tol}; stopping at M={m - 1}"
            )
            break
        seg_idx, k_tilde, h_tilde, parent_index, delta = result

        underlying = np.flatnonzero(seg_inds == seg_idx)
        logging.info(
            f"  accepted candidate seg {seg_idx} ({underlying.size} bin-segs), "
            f"k_tilde={k_tilde}, h_tilde={h_tilde}, "
            f"parent_index={clone_ids[parent_index]}, Δ={delta:.3f}"
        )
        cna_profile, cna_mirrored = append_clone(
            m,
            parent_index,
            k_tilde,
            h_tilde,
            underlying,
            cna_profile,
            cna_mirrored,
            phase,
            clone_ids,
            parent_map,
        )
        model_params["masks"] = get_masks_from_cna_profile(
            cna_int_states, cna_profile, cna_mirrored
        )
        clone_labels, _resp = cnp_copytyping(
            X,
            B,
            C,
            T,
            cna_profile,
            cna_mirrored,
            rdr_baf_states,
            clone_ids,
            phase=phase,
            model_params=model_params,
            em_kwargs=em_kwargs,
        )

    return clone_labels, cna_profile, cna_mirrored, clone_ids, phase, parent_map


def compute_sampling_probs_seg(
    X: sparse.csr_matrix,
    B: sparse.csr_matrix,
    C: sparse.csr_matrix,
    T: np.ndarray,
    cna_profile: np.ndarray,
    cna_mirrored: np.ndarray,
    phase: np.ndarray,
    rdr_baf_states: np.ndarray,
    model_params: dict[str, Any],
    em_kwargs: dict[str, Any],
    clone_labels: np.ndarray,
    seg_inds: np.ndarray,
    cnp_segments_df: pd.DataFrame,
    clone_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    fit_mode = em_kwargs["fit_mode"]
    eps = em_kwargs["eps"]
    base_props = model_params["base_props"]
    rdr_baf_params = model_params["rdr_baf_params"]
    clone_norm = get_clone_norm(base_props, cna_profile, rdr_baf_states)
    swap = (1 - phase).astype(cna_mirrored.dtype)[:, None]
    effective_mirrored = cna_mirrored ^ swap

    use_bb = fit_mode in ("hybrid", "allele_only")
    use_nb = fit_mode in ("hybrid", "total_only")

    n_cells = X.shape[1]
    n_cand = int(seg_inds.max()) + 1
    chunk_size = em_kwargs["chunk_size"]
    nll = np.zeros((n_cand, n_cells), dtype=np.float64)

    if use_bb:
        C_coo = C.tocoo()
        nz_g = C_coo.row
        nz_n = C_coo.col
        total = C_coo.data.astype(np.float64)
        b_nz = np.asarray(B.tocsr()[nz_g, nz_n]).ravel().astype(np.float64)
        a_nz = total - b_nz
        comb_nz = gammaln(total + 1) - gammaln(b_nz + 1) - gammaln(a_nz + 1)
        c_assigned = clone_labels[nz_n]
        state_idx = cna_profile[nz_g, c_assigned]
        mirror = effective_mirrored[nz_g, c_assigned]
        baf_state_col = rdr_baf_states[:, 1]
        baf = np.where(
            mirror == 1, 1.0 - baf_state_col[state_idx], baf_state_col[state_idx]
        )
        tau = rdr_baf_params[state_idx, 1]
        bb_ll = bb_logpmf_nz(b_nz, a_nz, comb_nz, tau * baf, tau * (1.0 - baf))
        np.add.at(nll, (seg_inds[nz_g], nz_n), -bb_ll)

    if use_nb:
        rdr_state_col = rdr_baf_states[:, 0]

        # dense baseline: log P(x=0 | mu, invphi) for every (g, n),
        # scattered to (cand, cell). Chunked over cells for memory.
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            c_chunk = clone_labels[start:end]  # (cs,)
            state_chunk = cna_profile[:, c_chunk]  # (G, cs)
            invphi_chunk = rdr_baf_params[state_chunk, 0]
            rdr_chunk = rdr_state_col[state_chunk]
            cn_chunk = np.clip(clone_norm[c_chunk], eps, None)
            T_chunk = T[start:end]
            mu_chunk = (
                base_props[:, None] * rdr_chunk / cn_chunk[None, :] * T_chunk[None, :]
            )
            mu_chunk = np.clip(mu_chunk, eps, None)
            zero_ll = nb_logpmf_zero(invphi_chunk, mu_chunk)  # (G, cs)
            np.add.at(nll[:, start:end], seg_inds, -zero_ll)

        # sparse adjustment: log P(x | mu, invphi) - log P(x=0 | mu, invphi) at x>0
        X_coo = X.tocoo()
        nz_g = X_coo.row
        nz_n = X_coo.col
        x_nz = X_coo.data.astype(np.float64)
        logfact_nz = gammaln(x_nz + 1)
        c_assigned = clone_labels[nz_n]
        state_idx = cna_profile[nz_g, c_assigned]
        invphi = rdr_baf_params[state_idx, 0]
        mu = (
            base_props[nz_g]
            * rdr_state_col[state_idx]
            / np.clip(clone_norm[c_assigned], eps, None)
            * T[nz_n]
        )
        mu = np.clip(mu, eps, None)
        adj_ll = nb_logpmf_nz_adjustment(x_nz, logfact_nz, invphi, mu)
        np.add.at(nll, (seg_inds[nz_g], nz_n), -adj_ll)

    # Per-clone NLL per cand seg: group nll's cell-columns by clone_labels.
    n_clones = len(clone_ids)
    nll_per_clone = np.zeros((n_cand, n_clones), dtype=np.float64)
    np.add.at(nll_per_clone, (slice(None), clone_labels), nll)

    for m, name in enumerate(clone_ids):
        cnp_segments_df[f"NLL_{name}"] = nll_per_clone[:, m]
    cnp_segments_df["total_NLL"] = nll.sum(axis=1)

    nll_cell_by_seg = nll.T  # (N, n_cand)
    # Sampling prob = excess NLL above per-seg best-fit cell, column-normalized.
    excess = nll_cell_by_seg - nll_cell_by_seg.min(axis=0, keepdims=True)
    col_sums = excess.sum(axis=0, keepdims=True)
    sampling_probs = np.where(
        col_sums > 0,
        excess / np.clip(col_sums, eps, None),
        1.0 / n_cells,
    )
    return sampling_probs, nll_cell_by_seg, cnp_segments_df


def append_clone(
    m: int,
    parent_index: int,
    k_tilde: int,
    h_tilde: int,
    underlying: np.ndarray,
    cna_profile: np.ndarray,
    cna_mirrored: np.ndarray,
    phase: np.ndarray,
    clone_ids: list[str],
    parent_map: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    new_cnp_col = cna_profile[:, parent_index].copy()
    new_mirror_col = cna_mirrored[:, parent_index].copy()
    new_cnp_col[underlying] = k_tilde
    if h_tilde == 0:
        new_mirror_col[underlying] = 1 - new_mirror_col[underlying]
        phase[underlying] = 0
    cna_profile = np.column_stack([cna_profile, new_cnp_col])
    cna_mirrored = np.column_stack([cna_mirrored, new_mirror_col])
    clone_ids.append(f"clone{m}")
    parent_map.append(parent_index)
    return cna_profile, cna_mirrored
