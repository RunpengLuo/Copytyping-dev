"""Conditional Beta-Binomial (BAF) + Negative-Binomial (RDR) log-PMFs and the
1-D MLE fits for their dispersions. Pure numpy/scipy; no copytyping dependencies."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import betaln, gammaln


##################################################
# dispersion broadcasting
##################################################


def _broadcast_dispersion(val: np.ndarray, G: int, N: int):
    """Reshape tau / inv_phi for (G, N, K) likelihood broadcast.

    Accepts scalar, (G,) per-bin, (N,) per-spot/cell, or (G, K) per-(bin, clone)
    (per-CNA-state mode). Returns (1,1,1), (G,1,1), (1,N,1), or (G,1,K).
    """
    arr = np.atleast_1d(val)
    if arr.size == 1:
        return arr.reshape(1, 1, 1)
    if arr.ndim == 2 and arr.shape[0] == G:
        return arr[:, None, :]
    if arr.shape == (G,):
        return arr[:, None, None]
    if arr.shape == (N,):
        return arr[None, :, None]
    raise ValueError(f"dispersion shape {arr.shape} not broadcastable for G={G}, N={N}")


##################################################
# conditional log-PMFs
##################################################


def cond_betabin_logpmf(
    count_B: np.ndarray,
    count_N: np.ndarray,
    tau: np.ndarray,
    p: np.ndarray,
):
    """Conditional BetaBinomial log-PMF: bb_ll_{g,n,k} = logP(count_B_{g,n}|l_n=k;param).

    Args:
        count_B: b-allele counts (G, N)
        count_N: total-allele counts (G, N)
        tau: dispersion (G,)
        p: BAF (G, K)

    Returns:
        (G, N, K) log-likelihood array.
    """
    (G, N) = count_B.shape
    count_B_gnk = count_B[:, :, None]
    count_N_gnk = count_N[:, :, None]
    count_A_gnk = count_N_gnk - count_B_gnk
    tau_gnk = _broadcast_dispersion(tau, G, N)
    p_gnk = p[:, None, :]

    a = tau_gnk * p_gnk
    b = tau_gnk * (1.0 - p_gnk)

    ll = (
        gammaln(count_N_gnk + 1)
        - gammaln(count_B_gnk + 1)
        - gammaln(count_A_gnk + 1)
        + betaln(count_B_gnk + a, count_A_gnk + b)
        - betaln(a, b)
    )
    return ll


def cond_negbin_logpmf(
    count_X: np.ndarray,
    count_T: np.ndarray,
    pi_gk: np.ndarray,
    inv_phi: np.ndarray,
    eps: float = 1e-12,
):
    """Conditional NegBinomial log-PMF: nb_ll_{g,n,k} = logP(count_X_{g,n}|l_n=k;param).

    Args:
        count_X: (G, N) observed read counts
        count_T: (N,) library size
        pi_gk: (G, K)
        inv_phi: (G,)

    Returns:
        (G, N, K) log-likelihood array.
    """
    (G, N) = count_X.shape
    mu_gnk = pi_gk[:, None, :] * count_T[None, :, None]
    mu_gnk = np.clip(mu_gnk, eps, None)
    count_X_gnk = count_X[:, :, None]

    inv_phi = _broadcast_dispersion(inv_phi, G, N)

    # Single expression so result broadcasts to (G, N, K) regardless of inv_phi shape.
    ll = (
        gammaln(count_X_gnk + inv_phi)
        - gammaln(inv_phi)
        - gammaln(count_X_gnk + 1.0)
        + inv_phi * np.log(inv_phi / (inv_phi + mu_gnk))
        + count_X_gnk * np.log(mu_gnk / (inv_phi + mu_gnk))
    )
    return ll


def cond_betabin_logpmf_theta(
    count_B: np.ndarray,
    count_N: np.ndarray,
    tau: np.ndarray,
    p: np.ndarray,
    rdrs_gk: np.ndarray,
    theta: np.ndarray,
    eps: float = 1e-12,
):
    """Spot-level conditional BetaBinomial log-PMF with tumor purity.

    p_hat = (theta * mu * p + 0.5*(1-theta)) / (theta*mu + 1 - theta)

    Returns:
        (G, N, K) log-likelihood array.
    """
    (G, N) = count_B.shape
    count_B_gnk = count_B[:, :, None]
    count_N_gnk = count_N[:, :, None]
    count_A_gnk = count_N_gnk - count_B_gnk
    tau_gnk = _broadcast_dispersion(tau, G, N)
    p_gnk = p[:, None, :]

    rdrs_gnk = rdrs_gk[:, None, :]
    theta_gnk = theta[None, :, None]

    denom = rdrs_gnk * theta_gnk + (1.0 - theta_gnk)
    num = p_gnk * rdrs_gnk * theta_gnk + 0.5 * (1.0 - theta_gnk)

    p_hat = num / np.clip(denom, eps, None)
    p_hat = np.clip(p_hat, eps, 1.0 - eps)

    a = tau_gnk * p_hat
    b = tau_gnk * (1.0 - p_hat)

    ll = (
        gammaln(count_N_gnk + 1)
        - gammaln(count_B_gnk + 1)
        - gammaln(count_A_gnk + 1)
        + betaln(count_B_gnk + a, count_A_gnk + b)
        - betaln(a, b)
    )
    return ll


def cond_negbin_logpmf_theta(
    count_X: np.ndarray,
    count_T: np.ndarray,
    lam_g: np.ndarray,
    inv_phi: np.ndarray,
    rdrs_gk: np.ndarray,
    theta: np.ndarray,
    eps: float = 1e-12,
):
    """Spot-level conditional NegBinomial log-PMF with tumor purity.

    Returns:
        (G, N, K) log-likelihood array.
    """
    (G, N) = count_X.shape

    count_X_gnk = count_X[:, :, None]
    count_T_gnk = count_T[None, :, None]
    lam_gnk = lam_g[:, None, None]
    rdrs_gnk = rdrs_gk[:, None, :]
    theta_gnk = theta[None, :, None]
    inv_phi = _broadcast_dispersion(inv_phi, G, N)

    mu_gnk = count_T_gnk * lam_gnk * (theta_gnk * rdrs_gnk + (1.0 - theta_gnk))
    mu_gnk = np.clip(mu_gnk, eps, None)

    # Single expression so result broadcasts to (G, N, K) regardless of inv_phi shape.
    ll = (
        gammaln(count_X_gnk + inv_phi)
        - gammaln(inv_phi)
        - gammaln(count_X_gnk + 1.0)
        + inv_phi * np.log(inv_phi / (inv_phi + mu_gnk))
        + count_X_gnk * np.log(mu_gnk / (inv_phi + mu_gnk))
    )
    return ll


##################################################
# dispersion MLE
##################################################


def mle_invphi(
    count_X_gnk: np.ndarray,
    mu_gnk: np.ndarray,
    weights: np.ndarray,
    invphi_bounds: tuple[float, float] = (1.0, 1e6),
    eps: float = 1e-12,
):
    """MLE of NB inv_phi within bounds."""
    mu_gnk = np.clip(mu_gnk, eps, None)

    def neg_Q_invphi(invphi):
        if invphi <= 0.0:
            return np.inf
        log_pmf = (
            gammaln(count_X_gnk + invphi)
            - gammaln(invphi)
            - gammaln(count_X_gnk + 1.0)
            + invphi * np.log(invphi / (invphi + mu_gnk))
            + count_X_gnk * np.log(mu_gnk / (invphi + mu_gnk))
        )
        return -np.sum(weights * log_pmf)

    res = minimize_scalar(
        neg_Q_invphi,
        bounds=invphi_bounds,
        method="bounded",
        options={"xatol": 1e-8},
    )
    return float(np.clip(res.x, invphi_bounds[0], invphi_bounds[1]))


def mle_tau(
    count_B_gnk: np.ndarray,
    count_N_gnk: np.ndarray,
    p_gnk: np.ndarray,
    weights: np.ndarray,
    tau_bounds: tuple[float, float] = (1.0, 1e6),
):
    """MLE of BB tau within bounds (optimized in log-space)."""
    count_A_gnk = count_N_gnk - count_B_gnk
    const = (
        gammaln(count_N_gnk + 1.0)
        - gammaln(count_B_gnk + 1.0)
        - gammaln(count_A_gnk + 1.0)
    )
    logtau_bounds = (np.log(tau_bounds[0]), np.log(tau_bounds[1]))

    def neg_Q_logtau(logtau):
        tau = np.exp(logtau)
        alpha = tau * p_gnk
        beta = tau * (1.0 - p_gnk)
        log_pmf = (
            const
            + betaln(count_B_gnk + alpha, count_A_gnk + beta)
            - betaln(alpha, beta)
        )
        return -np.sum(weights * log_pmf)

    res = minimize_scalar(
        neg_Q_logtau,
        bounds=logtau_bounds,
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(np.exp(np.clip(res.x, logtau_bounds[0], logtau_bounds[1])))


##################################################
# per-CNA-state dispersion MLE
##################################################


def _state_groups(cn_A: np.ndarray, cn_B: np.ndarray):
    """Yield (label, flat_idx) per distinct (A|B) state over a (G, K) CN grid.

    flat_idx indexes the flattened (G*K,) array; label is the "A|B" string.
    """
    code = cn_A.astype(np.int64) * 100000 + cn_B.astype(np.int64)
    flat = code.reshape(-1)
    for s in np.unique(flat):
        yield f"{s // 100000}|{s % 100000}", np.flatnonzero(flat == s)


def expand_state_map(
    cn_A: np.ndarray, cn_B: np.ndarray, state_map: dict[str, float], default: float
) -> np.ndarray:
    """Build a (G, K) per-(bin, clone) dispersion array from a {"A|B": value} map.

    Each (g, k) gets ``state_map[cn_A[g,k]|cn_B[g,k]]`` (``default`` if absent).
    Lets dispersion be estimated over one set of bins and broadcast over another.
    """
    G, K = cn_A.shape
    arr = np.full((G, K), default, dtype=float)
    flat = arr.reshape(-1)
    for label, gk in _state_groups(cn_A, cn_B):
        if label in state_map:
            flat[gk] = state_map[label]
    return arr


def mle_tau_per_state(
    count_B: np.ndarray,
    count_N: np.ndarray,
    BAF: np.ndarray,
    gamma: np.ndarray,
    cn_A: np.ndarray,
    cn_B: np.ndarray,
    tau_bounds: tuple[float, float] = (1.0, 1e6),
    eps: float = 1e-12,
) -> dict[str, float]:
    """Per-(A|B)-CNA-state BB tau MLE, pooling all cells within each state.

    Estimated over ALL bins passed in (not just the imbalanced mask) so a state
    such as (1|1) uses its full genome-wide signal. Use ``expand_state_map`` to
    broadcast the returned map onto the bins where the likelihood needs it.

    Args:
        count_B, count_N: (G, N) B-allele / total-allele counts (all clusters).
        BAF: (G, K) per-clone expected BAF.
        gamma: (N, K) posterior responsibilities.
        cn_A, cn_B: (G, K) per-clone copy numbers (the allele state).
        tau_bounds: (lo, hi) bounds for tau.

    Returns:
        tau_states: {"A|B": tau}, one entry per distinct state.
    """
    G, K = BAF.shape
    count_A = count_N - count_B  # (G, N)
    const = (
        gammaln(count_N + 1.0) - gammaln(count_B + 1.0) - gammaln(count_A + 1.0)
    )  # (G, N)
    logb = (np.log(tau_bounds[0]), np.log(tau_bounds[1]))

    tau_states: dict[str, float] = {}
    for label, gk in _state_groups(cn_A, cn_B):
        g_idx, k_idx = gk // K, gk % K
        w = gamma[:, k_idx].T  # (P, N)
        if w.sum() < eps:  # unassigned state -> leave at Binomial limit
            tau_states[label] = float(tau_bounds[1])
            continue
        p = BAF[g_idx, k_idx][:, None]  # (P, 1)
        count_B_p = count_B[g_idx]  # (P, N)
        count_A_p = count_A[g_idx]
        const_p = const[g_idx]

        def neg_Q(
            logtau, p=p, count_B_p=count_B_p, count_A_p=count_A_p, const_p=const_p, w=w
        ):
            tau = np.exp(logtau)
            a, b = tau * p, tau * (1.0 - p)
            return -np.sum(
                w * (const_p + betaln(count_B_p + a, count_A_p + b) - betaln(a, b))
            )

        res = minimize_scalar(
            neg_Q, bounds=logb, method="bounded", options={"xatol": 1e-6}
        )
        tau_states[label] = float(np.exp(np.clip(res.x, logb[0], logb[1])))
    return tau_states


def mle_invphi_per_state(
    count_X: np.ndarray,
    props_gk: np.ndarray,
    count_T: np.ndarray,
    gamma: np.ndarray,
    cn_A: np.ndarray,
    cn_B: np.ndarray,
    invphi_bounds: tuple[float, float] = (1.0, 1e6),
    eps: float = 1e-12,
) -> dict[str, float]:
    """Per-(A|B)-CNA-state NB inv_phi MLE, pooling all cells within each state.

    Estimated over all bins passed in (pass the lambda>0 bins, not just the
    aneuploid mask, so a diploid state uses its full signal). Use
    ``expand_state_map`` to broadcast onto the likelihood bins.

    Args:
        count_X: (G, N) observed read counts.
        props_gk: (G, K) per-clone expected read-depth proportion (lambda * RDR).
        count_T: (N,) per-cell library size.
        gamma: (N, K) posterior responsibilities.
        cn_A, cn_B: (G, K) per-clone copy numbers (the CNA state).
        invphi_bounds: (lo, hi) bounds for inv_phi.

    Returns:
        invphi_states: {"A|B": inv_phi}, one entry per distinct state.
    """
    G, K = props_gk.shape

    invphi_states: dict[str, float] = {}
    for label, gk in _state_groups(cn_A, cn_B):
        g_idx, k_idx = gk // K, gk % K
        w = gamma[:, k_idx].T  # (P, N)
        if w.sum() < eps:  # unassigned state -> leave at Poisson limit
            invphi_states[label] = float(invphi_bounds[1])
            continue
        mu = np.clip(
            props_gk[g_idx, k_idx][:, None] * count_T[None, :], eps, None
        )  # (P, N)
        count_X_p = count_X[g_idx]  # (P, N)

        def neg_Q(invphi, mu=mu, count_X_p=count_X_p, w=w):
            if invphi <= 0.0:
                return np.inf
            log_pmf = (
                gammaln(count_X_p + invphi)
                - gammaln(invphi)
                - gammaln(count_X_p + 1.0)
                + invphi * np.log(invphi / (invphi + mu))
                + count_X_p * np.log(mu / (invphi + mu))
            )
            return -np.sum(w * log_pmf)

        res = minimize_scalar(
            neg_Q, bounds=invphi_bounds, method="bounded", options={"xatol": 1e-8}
        )
        invphi_states[label] = float(np.clip(res.x, invphi_bounds[0], invphi_bounds[1]))
    return invphi_states
