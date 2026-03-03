import numpy as np
from scipy.special import softmax, expit, betaln, digamma, gammaln, logsumexp
from scipy.stats import binom, beta, norm
from scipy.optimize import minimize_scalar


##################################################
# Likelihood functions
def cond_betabin_logpmf(
    Y: np.ndarray,
    D: np.ndarray,
    tau: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    """
        compute loglik conditioned on labels per bin per cell per clone
        bb_ll_{g,n,k} = logP(Y_{g,n}|l_n=k;param)

    Args:
        Y (np.ndarray): b-allele counts (G, N)
        D (np.ndarray): total-allele counts (G, N)
        tau (np.ndarray): dispersion (G,)
        p (np.ndarray): BAF (G,K)

    Returns:
        np.ndarray: (G,N,K)
    """
    (G, N) = Y.shape
    K = p.shape[1]

    # (G, N, K)
    Y_gnk = Y[:, :, None]
    D_gnk = D[:, :, None]
    X_gnk = D_gnk - Y_gnk
    tau_gnk = np.broadcast_to(np.atleast_1d(tau)[:, None, None], (G, 1, 1))
    p_gnk = p[:, None, :]

    a = tau_gnk * p_gnk
    b = tau_gnk * (1.0 - p_gnk)

    ll = (gammaln(D_gnk + 1) - gammaln(Y_gnk + 1) - gammaln(X_gnk + 1)
          + betaln(Y_gnk + a, X_gnk + b) - betaln(a, b))
    return ll


def cond_negbin_logpmf(
    X: np.ndarray,
    T: np.ndarray,
    pi_gk: np.ndarray,
    inv_phi: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """compute loglik conditioned on labels per bin per cell per clone
        nb_ll_{g,n,k} = logP(X_{g,n}|l_n=k;param)

    Args:
        X (np.ndarray): (G,N), observed counts
        T (np.ndarray): (N,), library size
        pi_gk (np.ndarray): (G, K),
        inv_phi (np.ndarray): (G,)
        eps (float): floor for mu to avoid log(0) NaNs
    Returns:
        np.ndarray: (G,N,K)
    """
    (G, N) = X.shape
    K = pi_gk.shape[1]
    mu_gnk = pi_gk[:, None, :] * T[None, :, None]  # (G,N,K)
    mu_gnk = np.clip(mu_gnk, eps, None)
    X_gnk = X[:, :, None]  # (G, N, K)

    inv_phi = np.broadcast_to(np.atleast_1d(inv_phi)[:, None, None], (G, N, K))

    ll = gammaln(X_gnk + inv_phi) - gammaln(inv_phi) - gammaln(X_gnk + 1.0)
    ll += inv_phi * np.log(inv_phi / (inv_phi + mu_gnk))
    ll += X_gnk * np.log(mu_gnk / (inv_phi + mu_gnk))
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
    """compute loglik conditioned on labels per bin per cell per clone, spot-level data

    Y_{g,n} | l_n=m ~ BetaBinom(D_{g,n}, a_{g,n,m}, b_{g,n,m})
    with
      p_hat_{g,n,m} = (theta_n * mu_{g,m} * p_{g,m} + 0.5*(1-theta_n)) / (theta_n*mu_{g,m} + 1 - theta_n)
      a_{g,n,m} = tau_g * p_hat_{g,n,m}
      b_{g,n,m} = tau_g * (1 - p_hat_{g,n,m})

    Returns:
      ll: (G, N, K) log P(Y_{g,n} | l_n=m; params)

    Args:
        Y (np.ndarray): b-allele counts (G, N)
        D (np.ndarray): total-allele counts (G, N)
        tau (np.ndarray): dispersion (G,)
        p (np.ndarray): BAF (G,K)
        rdrs_gk (np.ndarray): RDR (G,K)
        theta (np.ndarray): tumor proportions (N,)

    Returns:
        np.ndarray: (G,N,K) loglik matrix
    """
    (G, N) = Y.shape
    K = p.shape[1]

    # (G, N, K)
    Y_gnk = Y[:, :, None]
    D_gnk = D[:, :, None]
    X_gnk = D_gnk - Y_gnk
    tau_gnk = np.broadcast_to(np.atleast_1d(tau)[:, None, None], (G, 1, 1))
    p_gnk = p[:, None, :]

    rdrs_gnk = rdrs_gk[:, None, :]
    theta_gnk = theta[None, :, None]

    denom = rdrs_gnk * theta_gnk + (1.0 - theta_gnk)
    num = p_gnk * rdrs_gnk * theta_gnk + 0.5 * (1.0 - theta_gnk)

    p_hat = num / np.clip(denom, eps, None)
    p_hat = np.clip(p_hat, eps, 1.0 - eps)

    a = tau_gnk * p_hat
    b = tau_gnk * (1.0 - p_hat)

    ll = (gammaln(D_gnk + 1) - gammaln(Y_gnk + 1) - gammaln(X_gnk + 1)
          + betaln(Y_gnk + a, X_gnk + b) - betaln(a, b))
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
    """spot-level data
        nb_ll_{g,n,k} = logP(X_{g,n}|l_n=k;param)

    Args:
        X (np.ndarray): (G,N)
        T (np.ndarray): (N,)
        lam_g (np.ndarray): (G,)
        inv_phi (np.ndarray): (G,)
        rdrs_gk (np.ndarray): (G,K)
        theta (np.ndarray): (N,)
        eps (float, optional): Defaults to 1e-12.
    Returns:
        np.ndarray: (G,N,K) loglik matrix
    """
    (G, N) = X.shape
    K = rdrs_gk.shape[1]

    X_gnk = X[:, :, None]  # (G, N, K)
    T_gnk = T[None, :, None]
    lam_gnk = lam_g[:, None, None]
    rdrs_gnk = rdrs_gk[:, None, :]
    theta_gnk = theta[None, :, None]
    inv_phi = np.broadcast_to(np.atleast_1d(inv_phi)[:, None, None], (G, N, K))

    mu_gnk = T_gnk * lam_gnk * (theta_gnk * rdrs_gnk + (1.0 - theta_gnk))
    mu_gnk = np.clip(mu_gnk, eps, None)

    ll = gammaln(X_gnk + inv_phi) - gammaln(inv_phi) - gammaln(X_gnk + 1.0)
    ll += inv_phi * np.log(inv_phi / (inv_phi + mu_gnk))
    ll += X_gnk * np.log(mu_gnk / (inv_phi + mu_gnk))
    return ll


##################################################
# fit functions
def mle_invphi(X_gnk, mu_gnk, weights, invphi_bounds, eps=1e-12):
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
    invphi_hat = float(np.clip(res.x, invphi_bounds[0], invphi_bounds[1]))
    return invphi_hat


def mle_tau(Y_gnk, D_gnk, p_gnk, weights, logtau_bounds):
    X_gnk = D_gnk - Y_gnk
    const = gammaln(D_gnk + 1.0) - gammaln(Y_gnk + 1.0) - gammaln(X_gnk + 1.0)

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
