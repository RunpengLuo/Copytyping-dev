import numpy as np
from scipy.special import softmax, expit, betaln, digamma, gammaln, logsumexp
from scipy.stats import binom, beta, norm

##################################################
# Likelihood functions
def _cond_betabin_logpmf(
    X: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    tau: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    """
        compute loglik conditioned on labels per bin per cell per clone
        bb_ll_{g,n,k} = logP(Y_{g,n}|l_n=k;param)

    Args:
        X (np.ndarray): a-allele counts (G, N)
        Y (np.ndarray): b-allele counts (G, N)
        D (np.ndarray): total-allele counts (G, N)
        tau (np.ndarray): dispersion (G,)
        p (np.ndarray): BAF (G,K)

    Returns:
        np.ndarray: (G,N,K)
    """
    (G, N) = X.shape
    K = p.shape[1]

    # (G, N, K)
    _X = X[:, :, None]
    _Y = Y[:, :, None]
    _D = D[:, :, None]
    _tau = np.broadcast_to(np.atleast_1d(tau)[:, None, None], (G, 1, 1))
    _p = p[:, None, :]

    a = _tau * _p
    b = _tau * (1.0 - _p)

    log_binom = gammaln(_D + 1) - gammaln(_Y + 1) - gammaln(_X + 1)
    ll = log_binom + betaln(_Y + a, _X + b) - betaln(a, b)
    return ll


def _cond_negbin_logpmf(
    T: np.ndarray,
    Tn: np.ndarray,
    props_gk: np.ndarray,
    inv_phi: np.ndarray,
) -> np.ndarray:
    """compute loglik conditioned on labels per bin per cell per clone
        bb_ll_{g,n,k} = logP(T_{g,n}|l_n=k;param)

    Args:
        T (np.ndarray): (G,N)
        Tn (np.ndarray): (N,)
        props (np.ndarray): (G, K)
        inv_phi (np.ndarray): (G,)
    Returns:
        np.ndarray: (G,N,K)
    """
    (G, N) = T.shape
    K = props_gk.shape[1]
    mu_counts = props_gk[:, None, :] * Tn[None, :, None]  # (G,N,K)
    _T = T[:, :, None]  # (G, N, K)

    _inv_phi = np.broadcast_to(np.atleast_1d(inv_phi)[:, None, None], (G, N, K))

    log_binom = gammaln(_T + _inv_phi) - gammaln(_inv_phi) - gammaln(_T + 1)
    ll = log_binom + _inv_phi * np.log(_inv_phi / (_inv_phi + mu_counts))
    ll = ll + _T * np.log(mu_counts / (_inv_phi + mu_counts))
    return ll
