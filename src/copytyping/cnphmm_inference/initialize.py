"""Reference-cell seeding for the CNP-HMM.

A per-clone Beta-Binomial (BAF) / Negative-Binomial (RDR) block-coordinate EM
over *fixed* bulk CN profiles that (1) picks the per-dataset reference-cell set,
(2) estimates the per-segment read-depth baseline, and (3) fits per-state
dispersions — the values used to initialize the HMM. The clone-level emissions,
the clone EM, the mask helper, and the ``initialize_copytyping`` wrapper are
folded into this one module (moved verbatim from the former ``joint_inference``
package; the HMM's own per-state emissions / M-step live in ``emissions.py`` /
``optimize.py``).
"""

import logging
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.special import betaln, gammaln, logsumexp

# memory budget (in array elements) for the invphi grid sub-blocks
_GRID_BUDGET = 2_000_000


# ===================== clone-level emissions (BB / NB) =======================
def bb_logpmf_nz(
    B_nz: np.ndarray,
    A_nz: np.ndarray,
    comb_nz: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    return comb_nz + betaln(B_nz + alpha, A_nz + beta) - betaln(alpha, beta)


def nb_logpmf_zero(invphi: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return invphi * (np.log(invphi) - np.log(invphi + mu))


def nb_logpmf_nz_adjustment(
    X_nz: np.ndarray,
    logfact_nz: np.ndarray,
    invphi: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    return (
        gammaln(X_nz + invphi)
        - gammaln(invphi)
        - logfact_nz
        + X_nz * (np.log(mu) - np.log(invphi + mu))
    )


def do_estep_clone_label(
    log_pi: np.ndarray,
    bb_args: dict[str, Any] | None = None,
    nb_args: dict[str, Any] | None = None,
    spot_purities: np.ndarray | None = None,
    chunk_size: int = 1000,
) -> tuple[np.ndarray, float]:
    """E-step: sums BB and/or NB conditional log-PMFs, adds ``log_pi``,
    softmaxes over clones. Returns ``(resp, total_ll)`` with ``resp`` of shape
    (N, M) and ``total_ll = sum_n logsumexp_m``.

    ``bb_args`` / ``nb_args`` are kwargs bundles for the BB / NB log-PMFs (at
    least one required); built once outside the EM loop, they hold a reference
    to ``rdr_baf_params`` so M-step mutations stay visible.
    """
    ll = None
    if bb_args is not None:
        ll = cond_betabin_logpmf(
            **bb_args, spot_purities=spot_purities, chunk_size=chunk_size
        )
    if nb_args is not None:
        nb = cond_negbin_logpmf(
            **nb_args, spot_purities=spot_purities, chunk_size=chunk_size
        )
        ll = nb if ll is None else ll + nb
    if ll is None:
        raise ValueError(
            "do_estep_clone_label: at least one of bb_args, nb_args required"
        )
    ll = ll + log_pi[None, :]
    log_norm = logsumexp(ll, axis=1, keepdims=True)
    resp = np.exp(ll - log_norm)
    return resp, float(log_norm.sum())


def cond_betabin_logpmf(
    B: sparse.csr_matrix,
    C: sparse.csr_matrix,
    cna_profile: np.ndarray,
    phase: np.ndarray,
    rdr_baf_states: np.ndarray,
    rdr_baf_params: np.ndarray,
    nz_seg: np.ndarray,
    nz_cell: np.ndarray,
    B_nz: np.ndarray,
    A_nz: np.ndarray,
    comb_nz: np.ndarray,
    base_props: np.ndarray | None = None,
    clone_norm: np.ndarray | None = None,
    spot_purities: np.ndarray | None = None,
    eps: float = 1e-12,
    chunk_size: int = 1000,
) -> np.ndarray:
    """BetaBinomial log-likelihood, returns shape (N, M).

    ``BetaBin(b | c=0, .) = 1``, so zero-total-allele bins contribute 0 — we
    evaluate only at the C-nonzero entries and scatter-add via ``bincount``.
    ``cna_profile`` indices state-encoded ``(A, B)`` (orientation in the state
    itself, no mirror flag). ``phase`` is the per-seg phasing chain shared by
    all clones — ``phase[g]=0`` flips observed B/A at bin g, equivalent to
    using ``1 - state.baf`` for the BB BAF.

    Single-cell (``spot_purities is None``):
        b | c, l=m ~ BetaBin(c, tau*p_m, tau*(1-p_m))
    Spot:
        p_hat = (theta*rdr_m*p_m + (1-theta)*0.5) / (theta*rdr_m + (1-theta))
        b | c, l=m ~ BetaBin(c, tau*p_hat, tau*(1-p_hat))
    """
    N = B.shape[1]
    M = cna_profile.shape[1]
    p_gm = rdr_baf_states[cna_profile, 1]  # (G, M) canonical state BAF
    tau_gm = rdr_baf_params[cna_profile, 1]  # (G, M)
    if phase is not None:
        flip_g = phase == 0
        p_gm = np.where(flip_g[:, None], 1.0 - p_gm, p_gm)  # phase-corrected BAF

    ll_nm = np.zeros((N, M), dtype=np.float64)

    if spot_purities is None:
        alpha_gm = tau_gm * p_gm
        beta_gm = tau_gm * (1.0 - p_gm)
        for m in range(M):
            contrib = bb_logpmf_nz(
                B_nz, A_nz, comb_nz, alpha_gm[nz_seg, m], beta_gm[nz_seg, m]
            )
            ll_nm[:, m] = np.bincount(nz_cell, weights=contrib, minlength=N)
        return ll_nm

    # spot model: purity-reweighted BAF p_hat, evaluated per nonzero entry
    mu_gm = rdr_baf_states[cna_profile, 0]  # (G, M) rdr
    S_m = (
        clone_norm
        if clone_norm is not None
        else (base_props[:, None] * mu_gm).sum(axis=0)
    )
    rdr_norm_gm = mu_gm / np.clip(S_m[None, :], eps, None)  # (G, M)
    theta_nz = spot_purities[nz_cell]  # (nnz,)
    for m in range(M):
        rdr_norm = rdr_norm_gm[nz_seg, m]
        numer = rdr_norm * theta_nz * p_gm[nz_seg, m] + (1.0 - theta_nz) * 0.5
        denom = rdr_norm * theta_nz + (1.0 - theta_nz)
        p_hat = np.clip(numer / np.clip(denom, eps, None), eps, 1.0 - eps)
        alpha = tau_gm[nz_seg, m] * p_hat
        beta = tau_gm[nz_seg, m] * (1.0 - p_hat)
        contrib = bb_logpmf_nz(B_nz, A_nz, comb_nz, alpha, beta)
        ll_nm[:, m] = np.bincount(nz_cell, weights=contrib, minlength=N)
    return ll_nm


def cond_negbin_logpmf(
    X: sparse.csr_matrix,
    T: np.ndarray,
    cna_profile: np.ndarray,
    rdr_baf_states: np.ndarray,
    rdr_baf_params: np.ndarray,
    base_props: np.ndarray,
    clone_norm: np.ndarray | None,
    nz_seg: np.ndarray,
    nz_cell: np.ndarray,
    X_nz: np.ndarray,
    logfact_nz: np.ndarray,
    spot_purities: np.ndarray | None = None,
    eps: float = 1e-12,
    chunk_size: int = 1000,
) -> np.ndarray:
    """NegBinomial log-likelihood, returns shape (N, M).

    Factors per-bin sum into a dense normalization term (every bin,
    accumulated in cell-chunks) plus a sparse read-count term (each summand
    vanishes when x=0):
        dense:  sum_g invphi * (log invphi - log(invphi + mu))
        sparse: sum_{x>0} [gammaln(x+invphi) - gammaln(invphi) - gammaln(x+1)
                           + x*(log mu - log(invphi+mu))]

    Single-cell (``spot_purities is None``), eq 1.11-1.12:
        mu = T_n * base_props_g * rdr_m_g / S_m
    Spot, eq 1.16:
        mu = T_n * base_props_g * (theta * rdr_m_g / S_m + (1-theta))

    ``clone_norm`` is the genome-wide S_m. ``nz_seg / nz_cell / X_nz /
    logfact_nz`` are the precomputed X-nonzeros.
    """
    N = X.shape[1]
    M = cna_profile.shape[1]
    mu_gm = rdr_baf_states[cna_profile, 0]  # (G, M) rdr
    invphi_gm = rdr_baf_params[cna_profile, 0]  # (G, M)
    S_m = (
        clone_norm
        if clone_norm is not None
        else (base_props[:, None] * mu_gm).sum(axis=0)
    )
    rdr_norm_gm = mu_gm / np.clip(S_m[None, :], eps, None)  # (G, M)
    pi_gm = base_props[:, None] * rdr_norm_gm  # (G, M) single-cell read fraction

    spot = spot_purities is not None

    # dense baseline term: sum_g log P(x=0 | mu, invphi), accumulated in cell chunks
    ll_nm = np.empty((N, M), dtype=np.float64)
    ip_g1m = invphi_gm[:, None, :]  # (G, 1, M)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        T_ch = T[start:end]  # (cs,)
        if spot:
            theta_ch = spot_purities[start:end]  # (cs,)
            mu_gnm = (
                T_ch[None, :, None]
                * base_props[:, None, None]
                * (
                    theta_ch[None, :, None] * rdr_norm_gm[:, None, :]
                    + (1.0 - theta_ch[None, :, None])
                )
            )  # (G, cs, M)
        else:
            mu_gnm = pi_gm[:, None, :] * T_ch[None, :, None]  # (G, cs, M)
        mu_gnm = np.clip(mu_gnm, eps, None)
        ll_nm[start:end] = nb_logpmf_zero(ip_g1m, mu_gnm).sum(axis=0)

    # sparse adjustment: log P(x | mu, invphi) - log P(x=0 | mu, invphi), at x > 0
    theta_nz = spot_purities[nz_cell] if spot else None
    for m in range(M):
        invphi = invphi_gm[nz_seg, m]  # (nnz,)
        if spot:
            mu = (
                T[nz_cell]
                * base_props[nz_seg]
                * (theta_nz * rdr_norm_gm[nz_seg, m] + (1.0 - theta_nz))
            )
        else:
            mu = pi_gm[nz_seg, m] * T[nz_cell]
        mu = np.clip(mu, eps, None)
        contrib = nb_logpmf_nz_adjustment(X_nz, logfact_nz, invphi, mu)
        ll_nm[:, m] += np.bincount(nz_cell, weights=contrib, minlength=N)
    return ll_nm


# ===================== clone EM (fixed-CNP block ascent) =====================
def build_bb_args(
    B_seg: sparse.csr_matrix,
    C_seg: sparse.csr_matrix,
    cna_profile_seg: np.ndarray,
    phase: np.ndarray,
    rdr_baf_states: np.ndarray,
    model_params: dict[str, Any],
    clone_norm: np.ndarray | None,
    bb_mask: np.ndarray,
) -> dict[str, Any]:
    """BetaBinomial sufficient-statistic bundle for the BB E-step emissions
    and the anchored-objective helpers.

    Carries the mask-restricted allele matrices ``(B, C)``, the COO indices
    of ``C``'s nonzeros (``nz_seg`` / ``nz_cell``) plus the per-nonzero
    count split ``(B_nz, A_nz)`` and ``comb_nz``. References stored by
    reference (no copy): masked ``cna_profile``, masked ``phase`` (per-seg
    phasing chain shared by clones; ``phase[g]=0`` flips BAF at bin g),
    masked ``base_props``, genome-wide ``clone_norm``, ``rdr_baf_states``,
    ``rdr_baf_params``.
    """
    rdr_baf_params = model_params["rdr_baf_params"]
    base_props = model_params["base_props"]

    B_al = B_seg[bb_mask]
    C_al = C_seg[bb_mask]
    state_al = cna_profile_seg[bb_mask]
    phase_al = phase[bb_mask]
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
    return dict(
        B=B_al,
        C=C_al,
        cna_profile=state_al,
        phase=phase_al,
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


def build_nb_args(
    X_seg: sparse.csr_matrix,
    T_seg: np.ndarray,
    cna_profile_seg: np.ndarray,
    rdr_baf_states: np.ndarray,
    model_params: dict[str, Any],
    clone_norm: np.ndarray | None,
    nb_mask: np.ndarray,
) -> dict[str, Any]:
    """NegBinomial sufficient-statistic bundle for the NB E-step emissions
    and the anchored-objective helpers.

    Carries the mask-restricted depth matrix ``X`` and per-cell library
    sizes ``T``, the COO indices of ``X``'s nonzeros (``nz_seg`` /
    ``nz_cell``) and per-nonzero counts ``X_nz`` — the data-only sufficient
    stats for the NB log-PMF — plus the data-only ``logfact_nz`` =
    ``gammaln(x+1)``. The rest are references the emission code needs:
    masked ``cna_profile`` / ``base_props``, genome-wide ``clone_norm``,
    ``rdr_baf_states``, ``rdr_baf_params``.

    Every shared array — including ``rdr_baf_params`` — is held by
    reference, so M-step in-place mutations flow through without rebuild.
    """
    rdr_baf_params = model_params["rdr_baf_params"]
    base_props = model_params["base_props"]

    X_dp = X_seg[nb_mask]
    state_dp = cna_profile_seg[nb_mask]
    base_dp = base_props[nb_mask]
    X_coo = X_dp.tocoo()
    depth_bin, depth_cell = X_coo.row, X_coo.col
    depth_count = X_coo.data.astype(np.float64)
    depth_logfact = gammaln(depth_count + 1)
    return dict(
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


def update_nb_bb_dispersion(
    labels: np.ndarray,
    em_kwargs: dict[str, Any],
    bb_args: dict[str, Any] | None = None,
    nb_args: dict[str, Any] | None = None,
) -> None:
    """Per-canonical-state 1-D bounded MLE for BB ``tau`` (col 1 of
    ``rdr_baf_params``) and NB ``invphi`` (col 0); mutates in place.

    Reads ``min_tau`` / ``max_tau`` / ``min_invphi`` / ``max_invphi`` /
    ``update_tau`` / ``update_invphi`` / ``eps`` from ``em_kwargs`` and
    ignores any other keys it carries.

    Skeleton: assign each nonzero to its clone's canonical CN state, then
    minimize the negative Q per state. Reuses the E-step kwargs bundles;
    either may be None (BB-only or NB-only EM), leaving that param untouched.
    """
    min_tau = em_kwargs["min_tau"]
    max_tau = em_kwargs["max_tau"]
    min_invphi = em_kwargs["min_invphi"]
    max_invphi = em_kwargs["max_invphi"]
    update_tau = em_kwargs["update_tau"]
    update_invphi = em_kwargs["update_invphi"]
    eps = em_kwargs["eps"]

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
        state_idx = bb_args["cna_profile"]
        phase = bb_args["phase"]
        rdr_baf_states, rdr_baf_params = (
            bb_args["rdr_baf_states"],
            bb_args["rdr_baf_params"],
        )

        assigned_clone = labels[nz_cell]
        assigned_state = state_idx[nz_seg, assigned_clone]
        assigned_flip = phase[nz_seg] == 0
        baf_s = rdr_baf_states[:, 1]
        assigned_baf = np.where(
            assigned_flip, 1.0 - baf_s[assigned_state], baf_s[assigned_state]
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
    B_seg: sparse.csr_matrix,
    C_seg: sparse.csr_matrix,
    X_seg: sparse.csr_matrix,
    T_seg: np.ndarray,
    cna_profile_seg: np.ndarray,
    phase: np.ndarray,
    rdr_baf_states: np.ndarray,
    model_params: dict[str, Any],
    clone_norm: np.ndarray | None,
    bb_mask: np.ndarray | None,
    nb_mask: np.ndarray | None,
    em_kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    """EM over bulk-derived clones.

    Reads ``rdr_baf_params`` / ``base_props`` / ``spot_purities`` from
    ``model_params``; mutates ``rdr_baf_params`` in place via the M-step and
    writes the final mixing weights to ``model_params["pi"]``.

    ``phase`` is the per-seg phasing chain (shared by all clones); folded
    into the BB BAF inside emission code (``phase[g]=0`` → ``baf_eff = 1 -
    state.baf`` at bin g).

    Returns ``(labels, resp, total_ll)``: MAP per-cell clone assignments,
    posterior responsibilities, and the joint marginal log-likelihood
    ``sum_n logsumexp_m`` evaluated at the converged params.
    """
    spot_purities = model_params["spot_purities"]

    niters = em_kwargs["niters"]
    tol = em_kwargs["tol"]
    chunk_size = em_kwargs["chunk_size"]

    n_cells = B_seg.shape[1]
    n_clones = cna_profile_seg.shape[1]
    if bb_mask is None and nb_mask is None:
        raise ValueError(
            "block_coordinate_ascent_fixed_cnp: at least one of bb_mask, nb_mask must be given"
        )

    bb_args = (
        build_bb_args(
            B_seg,
            C_seg,
            cna_profile_seg,
            phase,
            rdr_baf_states,
            model_params,
            clone_norm,
            bb_mask,
        )
        if bb_mask is not None
        else None
    )
    nb_args = (
        build_nb_args(
            X_seg,
            T_seg,
            cna_profile_seg,
            rdr_baf_states,
            model_params,
            clone_norm,
            nb_mask,
        )
        if nb_mask is not None
        else None
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
        update_nb_bb_dispersion(labels, em_kwargs, bb_args, nb_args)

        if t % 10 == 0:
            logging.info(
                f"  EM iter {t}: LL={total_ll:.1f}, pi=[{', '.join(f'{p:.3f}' for p in pi)}]"
            )

    model_params["pi"] = pi
    return resp.argmax(axis=1), resp, float(total_ll)


# ===================== reference cells / baseline / init =====================
def estimate_reference_cells(
    B_seg: sparse.csr_matrix,
    C_seg: sparse.csr_matrix,
    X_seg: sparse.csr_matrix,
    T_seg: np.ndarray,
    cna_profile_seg: np.ndarray,
    phase: np.ndarray,
    rdr_baf_states: np.ndarray,
    clone_ids: list[str],
    model_params: dict[str, Any],
    em_kwargs: dict[str, Any],
) -> tuple[np.ndarray, int, np.ndarray]:
    """BB-only EM to pick the per-dataset reference-cell set.

    Returns ``(labels_init, ref_clone, ref_cells)``.
    """
    n_cells = X_seg.shape[1]
    n_clones = cna_profile_seg.shape[1]
    has_normal = clone_ids[0] == "normal"
    masks = model_params["masks"]

    if has_normal:
        tumor_col = 1 if n_clones > 1 else 0
        profile_used = cna_profile_seg[:, [0, tumor_col]]
        bb_mask = masks["CLONAL_IMBALANCED"]
        mask_label = "CLONAL_IMBALANCED"
    else:
        profile_used = cna_profile_seg
        bb_mask = masks["IMBALANCED"]
        mask_label = "IMBALANCED"

    logging.info(
        f"estimate_reference_cells: BB-only EM on {int(bb_mask.sum())} "
        f"{mask_label} segs over {profile_used.shape[1]} clones"
    )
    labels_init, _resp, _ll = block_coordinate_ascent_fixed_cnp(
        B_seg,
        C_seg,
        X_seg,
        T_seg,
        profile_used,
        phase,
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


def get_masks_from_cna_profile(
    cna_int_states: np.ndarray,
    cna_profile: np.ndarray,
) -> dict[str, np.ndarray]:
    """Partition bins by CN-state pattern. ``(A, B)`` per (g, m) from
    ``cna_int_states[cna_profile[g, m]]`` directly (no mirror). Returns masks:

      * ``IMBALANCED``        — any clone has A ≠ B.
      * ``ANEUPLOID``         — any clone has total ≠ 2.
      * ``CLONAL_IMBALANCED`` — imbalanced and all tumor clones agree.
      * ``SUBCLONAL``         — tumor clones disagree (non-diploid).
    """
    a_cn = cna_int_states[cna_profile, 0]  # (G, M)
    b_cn = cna_int_states[cna_profile, 1]  # (G, M)

    imbalanced = np.any(a_cn != b_cn, axis=1)
    aneuploid = np.any((a_cn + b_cn) != 2, axis=1)

    # diploid -> clonal -> subclonal (disjoint, partition all bins)
    diploid = np.all((a_cn == 1) & (b_cn == 1), axis=1)
    tumor_same = np.all(a_cn[:, 1:] == a_cn[:, 1:2], axis=1) & np.all(
        b_cn[:, 1:] == b_cn[:, 1:2], axis=1
    )
    clonal = tumor_same & ~diploid
    subclonal = ~clonal & ~diploid

    return {
        "IMBALANCED": imbalanced,
        "ANEUPLOID": aneuploid,
        "CLONAL_IMBALANCED": imbalanced & clonal,
        "SUBCLONAL": subclonal,
    }


# ================================== plotting ==================================


# ================================ top wrapper =================================


def initialize_copytyping(
    X_seg: sparse.csr_matrix,
    B_seg: sparse.csr_matrix,
    C_seg: sparse.csr_matrix,
    T_seg: np.ndarray,
    cna_int_states: np.ndarray,
    cna_profile_seg: np.ndarray,
    phase: np.ndarray,
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
        "masks": get_masks_from_cna_profile(cna_int_states, cna_profile_seg),
        "base_props": None,
        "spot_purities": None,
    }

    labels_init, ref_clone, ref_cells = estimate_reference_cells(
        B_seg,
        C_seg,
        X_seg,
        T_seg,
        cna_profile_seg,
        phase,
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
            rdr_baf_states,
            model_params["base_props"],
            get_clone_norm(model_params["base_props"], cna_profile_seg, rdr_baf_states),
        )

    # ``labels_init`` and ``ref_clone`` are not model parameters — they're
    # inference outputs of the BB-only seeding pass. They travel back to the
    # caller via the explicit triple return below.
    return labels_init, ref_clone, model_params
