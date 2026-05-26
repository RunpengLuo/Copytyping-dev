"""Sparse BetaBinomial / NegBinomial log-PMF for the EM E-step. Single-cell
(``spot_purities is None``) and spot model (per-spot tumor purity) share the
same function — both evaluated only at precomputed nonzero entries.
"""

import numpy as np
from scipy.special import betaln, gammaln, logsumexp


def do_estep_clone_label(
    log_pi, bb_args=None, nb_args=None, spot_purities=None, chunk_size=1000
):
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
    B,
    C,
    cna_profile,
    cna_mirrored,
    rdr_baf_states,
    rdr_baf_params,
    nz_seg,
    nz_cell,
    B_nz,
    A_nz,
    comb_nz,
    base_props=None,
    clone_norm=None,
    spot_purities=None,
    eps=1e-12,
    chunk_size=1000,
):
    """BetaBinomial log-likelihood, returns shape (N, M).

    ``BetaBin(b | c=0, .) = 1``, so zero-total-allele bins contribute 0 — we
    evaluate only at the C-nonzero entries and scatter-add via ``bincount``
    (no dense (G, N, M) tensor). The mirror flag folds into the effective BAF.

    Single-cell (``spot_purities is None``), eq 1.13:
        b | c, l=m ~ BetaBin(c, tau*p_m, tau*(1-p_m))
    Spot, eq 1.17-1.18:
        p_hat = (theta*rdr_m*p_m + (1-theta)*0.5) / (theta*rdr_m + (1-theta))
        b | c, l=m ~ BetaBin(c, tau*p_hat, tau*(1-p_hat))

    ``nz_seg / nz_cell / B_nz / A_nz / comb_nz`` are the precomputed C-nonzeros
    with log multinomial coefficients; ``clone_norm`` (genome-wide S_m) is
    needed for the spot model only.
    """
    N = B.shape[1]
    M = cna_profile.shape[1]
    p_gm = rdr_baf_states[cna_profile, 1]  # (G, M) canonical BAF
    tau_gm = rdr_baf_params[cna_profile, 1]  # (G, M)
    if cna_mirrored is not None:
        p_gm = np.where(cna_mirrored == 1, 1.0 - p_gm, p_gm)  # effective BAF

    ll_nm = np.zeros((N, M), dtype=np.float64)

    if spot_purities is None:
        alpha_gm = tau_gm * p_gm
        beta_gm = tau_gm * (1.0 - p_gm)
        neg_betaln_gm = -betaln(alpha_gm, beta_gm)  # (G, M)
        for m in range(M):
            contrib = (
                comb_nz
                + betaln(B_nz + alpha_gm[nz_seg, m], A_nz + beta_gm[nz_seg, m])
                + neg_betaln_gm[nz_seg, m]
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
        a = tau_gm[nz_seg, m] * p_hat
        b = tau_gm[nz_seg, m] * (1.0 - p_hat)
        contrib = comb_nz + betaln(B_nz + a, A_nz + b) - betaln(a, b)
        ll_nm[:, m] = np.bincount(nz_cell, weights=contrib, minlength=N)
    return ll_nm


def cond_negbin_logpmf(
    X,
    T,
    cna_profile,
    rdr_baf_states,
    rdr_baf_params,
    base_props,
    clone_norm,
    nz_seg,
    nz_cell,
    X_nz,
    logfact_nz,
    spot_purities=None,
    eps=1e-12,
    chunk_size=1000,
):
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
    log_ip = np.log(invphi_gm)  # (G, M)

    spot = spot_purities is not None

    # dense normalization term, accumulated over cell chunks (bounded memory)
    ll_nm = np.empty((N, M), dtype=np.float64)
    norm_const_m = (invphi_gm * log_ip).sum(axis=0)  # (M,)
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
        ll_nm[start:end] = norm_const_m - (ip_g1m * np.log(ip_g1m + mu_gnm)).sum(axis=0)

    # sparse read-count term over the read-depth nonzeros
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
        contrib = (
            gammaln(X_nz + invphi)
            - gammaln(invphi)
            - logfact_nz
            + X_nz * (np.log(mu) - np.log(invphi + mu))
        )
        ll_nm[:, m] += np.bincount(nz_cell, weights=contrib, minlength=N)
    return ll_nm
