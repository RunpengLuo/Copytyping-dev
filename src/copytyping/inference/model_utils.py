import logging

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import betaln, gammaln
from scipy.stats import zscore

from copytyping.sx_data.sx_data import SX_Data


def _parse_bounds(s):
    """Parse comma-separated bounds string like '1.0,1e6' into (float, float)."""
    lo, hi = s.split(",")
    return (float(lo), float(hi))


def prepare_params(args, cnv_blocks, platform, data_types):
    """Build init_params and fix_params dicts from CLI args and CNV profile."""
    bulk_props = np.array(list(map(float, cnv_blocks["PROPS"].iloc[0].split(";"))))
    if args.get("init_pi_from_bulk", False):
        pi_init = bulk_props
    else:
        pi_init = np.ones(len(bulk_props)) / len(bulk_props)
    tau_bounds = _parse_bounds(args["tau_bounds"])
    invphi_bounds = _parse_bounds(args["invphi_bounds"])
    init_params = {
        "pi": pi_init,
        "pi_alpha": args["pi_alpha"],
        "tau_bounds": tau_bounds,
        "invphi_bounds": invphi_bounds,
        "ref_label": args["ref_label"],
        "niters": args["niters"],
    }
    fix_params = {"pi": not args.get("update_pi", False)}
    for data_type in data_types:
        fix_params[f"{data_type}-theta"] = True  # theta is fixed after init
        fix_params[f"{data_type}-tau"] = not args.get("update_tau", False)
        fix_params[f"{data_type}-inv_phi"] = not args.get("update_invphi", False)
    return init_params, fix_params


def compute_baseline_proportions(
    X: np.ndarray, T: np.ndarray, normal_labels: np.ndarray
) -> np.ndarray:
    X_normal = X[:, normal_labels]
    T_normal = T[normal_labels]
    base_props = np.sum(X_normal, axis=1) / np.sum(T_normal)
    return base_props


def clone_rdr_gk(lambda_g: np.ndarray, C: np.ndarray):
    """compute mu_{g,k}=C[g,k] / sum_{g}{lam_g * C[g,k]}

    Args:
        lambda_g (np.ndarray): (G,)
        C (np.ndarray): (G,K)
    """
    denom = (lambda_g[:, None] * C).sum(axis=0)  # (K, )
    mu_gk = C / denom  # (G, K)
    return mu_gk


# linear scaling assumption
def clone_pi_gk(lambda_g: np.ndarray, C: np.ndarray):
    """compute pi_gk=lam_g * rdr_gk
    Args:
        lambda_g (np.ndarray): (G,)
        C (np.ndarray): (G,K)
    """
    props_gk = lambda_g[:, None] * C
    props_gk = props_gk / np.sum(props_gk, axis=0, keepdims=True)
    return props_gk


def empirical_baf_gn(Y: np.ndarray, D: np.ndarray, norm=False):
    baf_matrix = np.divide(
        Y, D, out=np.full_like(D, fill_value=np.nan, dtype=np.float32), where=D > 0
    )
    if norm:
        baf_matrix[~np.isnan(baf_matrix)] -= 0.5
        baf_matrix = zscore(baf_matrix, axis=0, nan_policy="omit")
    return baf_matrix


def empirical_rdr_gn(
    X: np.ndarray, T: np.ndarray, base_props: np.ndarray, log2=False, norm=False
):
    """
    X: (G, N) G bin by spot/cell N count matrix
    T: (N,) total expression counts
    T*lambda_g*[(1-rho_n) + rho_n*rdr_gk]
    """
    rdr_denom = base_props[:, None] @ T[None, :]  # (G, N)
    rdr_matrix = np.divide(
        X,
        rdr_denom,
        out=np.full_like(rdr_denom, fill_value=np.nan, dtype=np.float32),
        where=rdr_denom > 0,
    )

    if log2:
        log2_mask = (~np.isnan(rdr_matrix)) & (rdr_matrix > 0)
        rdr_matrix[log2_mask] = np.log2(rdr_matrix[log2_mask])
    if norm:
        rdr_matrix = zscore(rdr_matrix, axis=0, nan_policy="omit")
    return rdr_matrix


def estimate_tumor_proportion(
    sx_data: SX_Data,
    base_props: np.ndarray,
    tau: float,
    inv_phi: float,
    fit_mode: str = "hybrid",
):
    """Per-spot purity init using BB+NB on clonal-only segments (no gamma weighting).

    Mirrors the spot model E-step likelihood but evaluated on segments shared by
    all tumor clones, where BAF_k and RDR_k are identical across k. The 1D MLE
    over theta is therefore independent of clone assignment.

    Args:
        sx_data: SX_Data with X, Y, D, T, A, B, C, BAF, MASK.
        base_props: (G,) baseline lambda from normal cells.
        tau: BB concentration scalar.
        inv_phi: NB inv-phi scalar.
        fit_mode: "allele_only", "total_only", or "hybrid".

    Returns:
        theta_arr: (N,) per-spot purity estimates in [1e-4, 1-1e-4].
    """
    rdrs_tumor = clone_rdr_gk(base_props, sx_data.C)[:, 1:]  # (G, K_tumor)
    non_sub = ~sx_data.MASK["SUBCLONAL"]
    am = sx_data.MASK["IMBALANCED"] & non_sub & (base_props > 0)
    tm = sx_data.MASK["ANEUPLOID"] & non_sub & (base_props > 0)
    n_am = int(am.sum())
    n_tm = int(tm.sum())
    logging.info(f"init theta: {n_am} clonal imbalanced, {n_tm} clonal aneuploid bins")

    theta_arr = np.full(sx_data.N, 0.5, dtype=np.float32)
    use_a = fit_mode in {"allele_only", "hybrid"} and n_am > 0
    use_t = fit_mode in {"total_only", "hybrid"} and n_tm > 0
    if not (use_a or use_t):
        logging.warning("no clonal informative bins; theta=0.5")
        return theta_arr

    if use_a:
        Y_am = sx_data.Y[am]
        D_am = sx_data.D[am]
        BAF_am = sx_data.BAF[am, 1:]
        rdrs_am = rdrs_tumor[am]
    if use_t:
        X_tm = sx_data.X[tm]
        lambda_tm = base_props[tm]
        rdrs_tm = rdrs_tumor[tm]

    for n in range(sx_data.N):

        def neg_Q(tv, _n=n):
            tv_arr = np.array([tv], dtype=float)
            Q = 0.0
            if use_a:
                ll = cond_betabin_logpmf_theta(
                    Y_am[:, _n : _n + 1],
                    D_am[:, _n : _n + 1],
                    tau,
                    BAF_am,
                    rdrs_am,
                    tv_arr,
                )
                # clonal segments → BAF_k identical across k → take k=0
                Q += float(ll[:, 0, 0].sum())
            if use_t:
                ll = cond_negbin_logpmf_theta(
                    X_tm[:, _n : _n + 1],
                    np.array([sx_data.T[_n]], dtype=float),
                    lambda_tm,
                    inv_phi,
                    rdrs_tm,
                    tv_arr,
                )
                Q += float(ll[:, 0, 0].sum())
            return -Q

        res = minimize_scalar(neg_Q, bounds=(1e-4, 1.0 - 1e-4), method="bounded")
        theta_arr[n] = np.clip(res.x, 1e-4, 1.0 - 1e-4)

    return theta_arr


##################################################
# Likelihood functions


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
    tau_gnk = np.broadcast_to(np.atleast_1d(tau)[:, None, None], (G, 1, 1))
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
    K = pi_gk.shape[1]
    mu_gnk = pi_gk[:, None, :] * T[None, :, None]
    mu_gnk = np.clip(mu_gnk, eps, None)
    X_gnk = X[:, :, None]

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
    """Spot-level conditional BetaBinomial log-PMF with tumor purity.

    p_hat = (theta * mu * p + 0.5*(1-theta)) / (theta*mu + 1 - theta)

    Returns:
        (G, N, K) log-likelihood array.
    """
    (G, N) = Y.shape
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
    K = rdrs_gk.shape[1]

    X_gnk = X[:, :, None]
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
# MLE fit functions


def mle_invphi(X_gnk, mu_gnk, weights, invphi_bounds=(1.0, 1e6), eps=1e-12):
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


def mle_tau(Y_gnk, D_gnk, p_gnk, weights, tau_bounds=(1.0, 1e6)):
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
