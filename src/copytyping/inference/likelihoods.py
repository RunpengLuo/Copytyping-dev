"""Conditional Beta-Binomial (BAF) + Negative-Binomial (RDR) log-PMFs and the
1-D MLE fits for their dispersions. Pure numpy/scipy; no copytyping dependencies."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import betaln, gammaln


##################################################
# dispersion broadcasting
##################################################


def _broadcast_dispersion(val: np.ndarray, G: int, N: int) -> np.ndarray:
    """Reshape tau / inv_phi for (G, N, K) likelihood broadcast.

    Accepts scalar, (G,) per-bin, or (N,) per-spot/cell. Returns (G,1,1) or (1,N,1).
    """
    arr = np.atleast_1d(val)
    if arr.size == 1:
        return arr.reshape(1, 1, 1)
    if arr.shape == (G,):
        return arr[:, None, None]
    if arr.shape == (N,):
        return arr[None, :, None]
    raise ValueError(f"dispersion shape {arr.shape} not in {{(1,), ({G},), ({N},)}}")


##################################################
# conditional log-PMFs
##################################################


def cond_betabin_logpmf(
    Y: np.ndarray,
    D: np.ndarray,
    tau: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    """Conditional BetaBinomial log-PMF: bb_ll_{g,n,k} = logP(Y_{g,n}|l_n=k;param).

    Args:
        Y: b-allele counts (G, N)
        D: total-allele counts (G, N)
        tau: dispersion (G,)
        p: BAF (G, K)

    Returns:
        (G, N, K) log-likelihood array.
    """
    (G, N) = Y.shape
    Y_gnk = Y[:, :, None]
    D_gnk = D[:, :, None]
    X_gnk = D_gnk - Y_gnk
    tau_gnk = _broadcast_dispersion(tau, G, N)
    p_gnk = p[:, None, :]

    a = tau_gnk * p_gnk
    b = tau_gnk * (1.0 - p_gnk)

    ll = (
        gammaln(D_gnk + 1)
        - gammaln(Y_gnk + 1)
        - gammaln(X_gnk + 1)
        + betaln(Y_gnk + a, X_gnk + b)
        - betaln(a, b)
    )
    return ll


def cond_negbin_logpmf(
    X: np.ndarray,
    T: np.ndarray,
    pi_gk: np.ndarray,
    inv_phi: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Conditional NegBinomial log-PMF: nb_ll_{g,n,k} = logP(X_{g,n}|l_n=k;param).

    Args:
        X: (G, N) observed counts
        T: (N,) library size
        pi_gk: (G, K)
        inv_phi: (G,)

    Returns:
        (G, N, K) log-likelihood array.
    """
    (G, N) = X.shape
    mu_gnk = pi_gk[:, None, :] * T[None, :, None]
    mu_gnk = np.clip(mu_gnk, eps, None)
    X_gnk = X[:, :, None]

    inv_phi = _broadcast_dispersion(inv_phi, G, N)

    # Single expression so result broadcasts to (G, N, K) regardless of inv_phi shape.
    ll = (
        gammaln(X_gnk + inv_phi)
        - gammaln(inv_phi)
        - gammaln(X_gnk + 1.0)
        + inv_phi * np.log(inv_phi / (inv_phi + mu_gnk))
        + X_gnk * np.log(mu_gnk / (inv_phi + mu_gnk))
    )
    return ll


def cond_betabin_logpmf_theta(
    Y: np.ndarray,
    D: np.ndarray,
    tau: np.ndarray,
    p: np.ndarray,
    rdrs_gk: np.ndarray,
    theta: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Spot-level conditional BetaBinomial log-PMF with tumor purity.

    p_hat = (theta * mu * p + 0.5*(1-theta)) / (theta*mu + 1 - theta)

    Returns:
        (G, N, K) log-likelihood array.
    """
    (G, N) = Y.shape
    Y_gnk = Y[:, :, None]
    D_gnk = D[:, :, None]
    X_gnk = D_gnk - Y_gnk
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
        gammaln(D_gnk + 1)
        - gammaln(Y_gnk + 1)
        - gammaln(X_gnk + 1)
        + betaln(Y_gnk + a, X_gnk + b)
        - betaln(a, b)
    )
    return ll


def cond_negbin_logpmf_theta(
    X: np.ndarray,
    T: np.ndarray,
    lam_g: np.ndarray,
    inv_phi: np.ndarray,
    rdrs_gk: np.ndarray,
    theta: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Spot-level conditional NegBinomial log-PMF with tumor purity.

    Returns:
        (G, N, K) log-likelihood array.
    """
    (G, N) = X.shape

    X_gnk = X[:, :, None]
    T_gnk = T[None, :, None]
    lam_gnk = lam_g[:, None, None]
    rdrs_gnk = rdrs_gk[:, None, :]
    theta_gnk = theta[None, :, None]
    inv_phi = _broadcast_dispersion(inv_phi, G, N)

    mu_gnk = T_gnk * lam_gnk * (theta_gnk * rdrs_gnk + (1.0 - theta_gnk))
    mu_gnk = np.clip(mu_gnk, eps, None)

    # Single expression so result broadcasts to (G, N, K) regardless of inv_phi shape.
    ll = (
        gammaln(X_gnk + inv_phi)
        - gammaln(inv_phi)
        - gammaln(X_gnk + 1.0)
        + inv_phi * np.log(inv_phi / (inv_phi + mu_gnk))
        + X_gnk * np.log(mu_gnk / (inv_phi + mu_gnk))
    )
    return ll


##################################################
# dispersion MLE
##################################################


def mle_invphi(
    X_gnk: np.ndarray,
    mu_gnk: np.ndarray,
    weights: np.ndarray,
    invphi_bounds: tuple[float, float] = (1.0, 1e6),
    eps: float = 1e-12,
) -> float:
    """MLE of NB inv_phi within bounds."""
    mu_gnk = np.clip(mu_gnk, eps, None)

    def neg_Q_invphi(invphi):
        if invphi <= 0.0:
            return np.inf
        log_pmf = (
            gammaln(X_gnk + invphi)
            - gammaln(invphi)
            - gammaln(X_gnk + 1.0)
            + invphi * np.log(invphi / (invphi + mu_gnk))
            + X_gnk * np.log(mu_gnk / (invphi + mu_gnk))
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
    Y_gnk: np.ndarray,
    D_gnk: np.ndarray,
    p_gnk: np.ndarray,
    weights: np.ndarray,
    tau_bounds: tuple[float, float] = (1.0, 1e6),
) -> float:
    """MLE of BB tau within bounds (optimized in log-space)."""
    X_gnk = D_gnk - Y_gnk
    const = gammaln(D_gnk + 1.0) - gammaln(Y_gnk + 1.0) - gammaln(X_gnk + 1.0)
    logtau_bounds = (np.log(tau_bounds[0]), np.log(tau_bounds[1]))

    def neg_Q_logtau(logtau):
        tau = np.exp(logtau)
        alpha = tau * p_gnk
        beta = tau * (1.0 - p_gnk)
        log_pmf = const + betaln(Y_gnk + alpha, X_gnk + beta) - betaln(alpha, beta)
        return -np.sum(weights * log_pmf)

    res = minimize_scalar(
        neg_Q_logtau,
        bounds=logtau_bounds,
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(np.exp(np.clip(res.x, logtau_bounds[0], logtau_bounds[1])))
