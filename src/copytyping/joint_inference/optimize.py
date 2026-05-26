"""EM loop (``block_coordinate_ascent_fixed_cnp``) and M-step per-state dispersion MLE
(``update_nb_bb_dispersion``); the E-step lives in ``emissions``.
"""

import logging

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import betaln, gammaln

from copytyping.joint_inference.emissions import do_estep_clone_label

# memory budget (in array elements) for the invphi grid sub-blocks
_GRID_BUDGET = 2_000_000


def update_nb_bb_dispersion(
    labels,
    bb_args=None,
    nb_args=None,
    min_tau=50.0,
    max_tau=5000.0,
    min_invphi=20.0,
    max_invphi=5000.0,
    update_tau=True,
    update_invphi=True,
    eps=1e-12,
):
    """Per-canonical-state 1-D bounded MLE for BB ``tau`` (col 1 of
    ``rdr_baf_params``) and NB ``invphi`` (col 0); mutates in place.

    Skeleton: assign each nonzero to its clone's canonical CN state, then
    minimize the negative Q per state. Reuses the E-step kwargs bundles;
    either may be None (BB-only or NB-only EM), leaving that param untouched.
    """

    def neg_Q_bb(log_tau, buckets):
        """Negative BetaBinomial Q(tau) over unique (B, A) buckets per effective BAF."""
        tau = np.exp(log_tau)
        total = 0.0
        for b_u, a_u, counts, logcomb_u, baf in buckets:
            alpha, beta = tau * baf, tau * (1.0 - baf)
            total += float(
                (
                    counts
                    * (
                        logcomb_u
                        + betaln(b_u + alpha, a_u + beta)
                        - betaln(alpha, beta)
                    )
                ).sum()
            )
        return -total

    def neg_Q_nb(invphi, grids, count_vals, count_freq, nz_counts, nz_mu, n_nz, n_grid):
        """Negative NegBinomial Q(invphi); the grid term spans every (seg, cell)
        pair of the state but is accumulated in bounded-memory blocks."""
        grid_logsum = 0.0
        for seg_coef, cell_lib in grids:
            chunk = max(1, _GRID_BUDGET // max(cell_lib.size, 1))
            for s in range(0, seg_coef.size, chunk):
                mu_block = np.clip(
                    seg_coef[s : s + chunk, None] * cell_lib[None, :], eps, None
                )
                grid_logsum += float(np.log(invphi + mu_block).sum())
        q = (
            float((count_freq * gammaln(count_vals + invphi)).sum())
            - n_nz * gammaln(invphi)
            + n_grid * invphi * np.log(invphi)
            - invphi * grid_logsum
            - float((nz_counts * np.log(invphi + nz_mu)).sum())
        )
        return -q

    # --- BetaBinomial tau, per canonical state from allele nonzeros ---
    if update_tau and bb_args is not None:
        nz_seg = bb_args["nz_seg"]
        nz_cell = bb_args["nz_cell"]
        B_nz, A_nz = bb_args["B_nz"], bb_args["A_nz"]
        state_idx, mirror_idx = bb_args["cna_profile"], bb_args["cna_mirrored"]
        rdr_baf_states, rdr_baf_params = (
            bb_args["rdr_baf_states"],
            bb_args["rdr_baf_params"],
        )

        assigned_clone = labels[nz_cell]
        assigned_state = state_idx[nz_seg, assigned_clone]
        assigned_mirror = mirror_idx[nz_seg, assigned_clone]
        baf_s = rdr_baf_states[:, 1]
        assigned_baf = np.where(
            assigned_mirror == 1, 1.0 - baf_s[assigned_state], baf_s[assigned_state]
        )
        log_bounds = (np.log(min_tau), np.log(max_tau))

        for state in np.unique(assigned_state):
            in_state = assigned_state == state
            b_grp, a_grp, baf_grp = (
                B_nz[in_state],
                A_nz[in_state],
                assigned_baf[in_state],
            )

            # collapse to unique (B, A) per effective BAF with multiplicities
            buckets = []
            for baf in np.unique(np.round(baf_grp, 8)):
                same_baf = np.abs(baf_grp - baf) < 1e-7
                b_vals, a_vals = b_grp[same_baf], a_grp[same_baf]
                radix = int(a_vals.max()) + 1 if a_vals.size else 1
                keys_u, inv = np.unique(
                    (b_vals * radix + a_vals).astype(np.int64), return_inverse=True
                )
                counts = np.bincount(inv).astype(np.float64)
                b_u = (keys_u // radix).astype(np.float64)
                a_u = (keys_u % radix).astype(np.float64)
                logcomb_u = gammaln(b_u + a_u + 1) - gammaln(b_u + 1) - gammaln(a_u + 1)
                buckets.append((b_u, a_u, counts, logcomb_u, float(baf)))

            res = minimize_scalar(
                lambda log_tau: neg_Q_bb(log_tau, buckets),
                bounds=log_bounds,
                method="bounded",
            )
            rdr_baf_params[state, 1] = float(np.exp(np.clip(res.x, *log_bounds)))

    # --- NegBinomial invphi, per canonical state from the read-count grid ---
    if update_invphi and nb_args is not None:
        nz_seg = nb_args["nz_seg"]
        nz_cell = nb_args["nz_cell"]
        X_nz, T_seg = nb_args["X_nz"], nb_args["T"]
        state_idx = nb_args["cna_profile"]
        base_props, clone_norm = nb_args["base_props"], nb_args["clone_norm"]
        rdr_baf_states, rdr_baf_params = (
            nb_args["rdr_baf_states"],
            nb_args["rdr_baf_params"],
        )

        n_clones = state_idx.shape[1]
        cells_of_clone = [np.where(labels == clone)[0] for clone in range(n_clones)]
        nz_clone = labels[nz_cell]
        nz_state = state_idx[nz_seg, nz_clone]

        for state in np.unique(state_idx):
            state_rdr = float(rdr_baf_states[state, 0])
            if state_rdr == 0.0:
                continue

            # per-clone (seg, cell) grid for the dense normalization term
            grids = []
            n_grid = 0
            for clone in range(n_clones):
                cells = cells_of_clone[clone]
                if cells.size == 0:
                    continue
                state_segs = np.where(state_idx[:, clone] == state)[0]
                if state_segs.size == 0:
                    continue
                seg_coef = (
                    base_props[state_segs] * state_rdr / max(clone_norm[clone], eps)
                )
                grids.append((seg_coef, T_seg[cells]))
                n_grid += state_segs.size * cells.size
            if n_grid == 0:
                continue

            # sparse read-count points (x>0) for this state
            in_state = nz_state == state
            nz_counts = X_nz[in_state]
            nz_mu = np.clip(
                base_props[nz_seg[in_state]]
                * state_rdr
                / np.clip(clone_norm[nz_clone[in_state]], eps, None)
                * T_seg[nz_cell[in_state]],
                eps,
                None,
            )
            n_nz = nz_counts.size
            count_vals, count_freq = np.unique(nz_counts, return_counts=True)
            count_freq = count_freq.astype(np.float64)

            res = minimize_scalar(
                lambda invphi: neg_Q_nb(
                    invphi,
                    grids,
                    count_vals,
                    count_freq,
                    nz_counts,
                    nz_mu,
                    n_nz,
                    n_grid,
                ),
                bounds=(min_invphi, max_invphi),
                method="bounded",
            )
            rdr_baf_params[state, 0] = float(np.clip(res.x, min_invphi, max_invphi))


def block_coordinate_ascent_fixed_cnp(
    B_seg,
    C_seg,
    X_seg,
    T_seg,
    cna_profile_seg,
    cna_mirrored_seg,
    rdr_baf_states,
    rdr_baf_params,
    base_props,
    clone_norm,
    bb_mask,
    nb_mask,
    spot_purities=None,
    niters=100,
    tol=1e-4,
    min_tau=50.0,
    max_tau=5000.0,
    min_invphi=20.0,
    max_invphi=5000.0,
    update_tau=True,
    update_invphi=True,
    chunk_size=1000,
):
    """EM over bulk-derived clones. Returns ``(labels, pi, rdr_baf_params)``.

    Caller controls which bins and clones via ``bb_mask`` / ``nb_mask`` (either
    may be None — at least one required) and via pre-subsetting
    ``cna_profile_seg`` / ``cna_mirrored_seg`` columns. ``base_props`` is
    restricted to the masked bins internally; ``clone_norm`` stays genome-wide
    and must match the number of clones in the passed profile.
    """
    n_cells = B_seg.shape[1]
    n_clones = cna_profile_seg.shape[1]
    run_allele = bb_mask is not None
    run_depth = nb_mask is not None
    if not (run_allele or run_depth):
        raise ValueError(
            "block_coordinate_ascent_fixed_cnp: at least one of bb_mask, nb_mask must be given"
        )

    # allele (BetaBinomial) inputs: nonzeros of C over the allele bins
    bb_args = None
    if run_allele:
        B_al = B_seg[bb_mask]
        C_al = C_seg[bb_mask]
        state_al = cna_profile_seg[bb_mask]
        mirror_al = cna_mirrored_seg[bb_mask]
        base_al = None if base_props is None else base_props[bb_mask]
        C_coo = C_al.tocoo()
        allele_bin, allele_cell = C_coo.row, C_coo.col
        allele_total = C_coo.data.astype(np.float64)
        b_allele = (
            np.asarray(B_al.tocsr()[allele_bin, allele_cell]).ravel().astype(np.float64)
        )
        a_allele = allele_total - b_allele
        allele_logcomb = (
            gammaln(allele_total + 1) - gammaln(b_allele + 1) - gammaln(a_allele + 1)
        )
        bb_args = dict(
            B=B_al,
            C=C_al,
            cna_profile=state_al,
            cna_mirrored=mirror_al,
            rdr_baf_states=rdr_baf_states,
            rdr_baf_params=rdr_baf_params,
            nz_seg=allele_bin,
            nz_cell=allele_cell,
            B_nz=b_allele,
            A_nz=a_allele,
            comb_nz=allele_logcomb,
            base_props=base_al,
            clone_norm=clone_norm,
        )

    # depth (NegBinomial) inputs: nonzeros of X over the depth bins
    nb_args = None
    if run_depth:
        X_dp = X_seg[nb_mask]
        state_dp = cna_profile_seg[nb_mask]
        base_dp = base_props[nb_mask]
        X_coo = X_dp.tocoo()
        depth_bin, depth_cell = X_coo.row, X_coo.col
        depth_count = X_coo.data.astype(np.float64)
        depth_logfact = gammaln(depth_count + 1)
        nb_args = dict(
            X=X_dp,
            T=T_seg,
            cna_profile=state_dp,
            rdr_baf_states=rdr_baf_states,
            rdr_baf_params=rdr_baf_params,
            base_props=base_dp,
            clone_norm=clone_norm,
            nz_seg=depth_bin,
            nz_cell=depth_cell,
            X_nz=depth_count,
            logfact_nz=depth_logfact,
        )

    pi = np.ones(n_clones, dtype=np.float64) / n_clones
    log_pi = np.log(pi)
    prev_ll = -np.inf

    for t in range(niters):
        # E-step: posterior responsibilities and marginal log-likelihood.
        resp, total_ll = do_estep_clone_label(
            log_pi,
            bb_args,
            nb_args,
            spot_purities=spot_purities,
            chunk_size=chunk_size,
        )

        if t > 0 and abs(total_ll - prev_ll) / (abs(prev_ll) + 1e-10) < tol:
            logging.info(f"  EM converged at iter {t}")
            break
        prev_ll = total_ll

        # M-step
        labels = resp.argmax(axis=1)
        pi = np.bincount(labels, minlength=n_clones).astype(np.float64)
        pi = np.clip(pi / n_cells, 1e-10, None)
        pi /= pi.sum()
        log_pi = np.log(pi)

        # M-step: per-canonical-state dispersion MLEs (tau, invphi). Reuses the
        # E-step kwargs bundles; either may be None (BB-only or NB-only EM).
        update_nb_bb_dispersion(
            labels,
            bb_args,
            nb_args,
            min_tau=min_tau,
            max_tau=max_tau,
            min_invphi=min_invphi,
            max_invphi=max_invphi,
            update_tau=update_tau,
            update_invphi=update_invphi,
        )

        if t % 10 == 0:
            logging.info(
                f"  EM iter {t}: LL={total_ll:.1f}, pi=[{', '.join(f'{p:.3f}' for p in pi)}]"
            )

    return resp.argmax(axis=1), pi, rdr_baf_params
