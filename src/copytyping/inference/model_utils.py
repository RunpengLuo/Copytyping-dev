import logging

import numpy as np
import pandas as pd

from scipy.stats import (
    zscore,
    binom,
)

from scipy.optimize import minimize_scalar
from copytyping.sx_data.sx_data import SX_Data


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


def _fit_theta_spots(Y_bins, D_bins, p_k, mu_k, N, theta_arr, spot_mask):
    """Fit theta for a subset of spots using binomial MLE on given bins."""
    eps = 1e-10
    n_fitted = 0
    for n in np.where(spot_mask)[0]:
        d_n = D_bins[:, n]
        y_n = Y_bins[:, n]
        valid = (d_n > 0) & (mu_k > 0) & np.isfinite(mu_k)
        if valid.sum() == 0:
            continue
        d_v = d_n[valid].astype(np.float64)
        y_v = y_n[valid].astype(np.float64)
        mu_v = mu_k[valid]
        p_v = p_k[valid]

        def neg_ll(theta):
            denom = theta * mu_v + (1 - theta)
            p_hat = (theta * mu_v * p_v + 0.5 * (1 - theta)) / np.maximum(denom, eps)
            p_hat = np.clip(p_hat, eps, 1 - eps)
            return -np.sum(binom.logpmf(y_v, d_v, p_hat))

        res = minimize_scalar(neg_ll, bounds=(1e-4, 1 - 1e-4), method="bounded")
        theta_arr[n] = np.clip(res.x, 1e-4, 1 - 1e-4)
        n_fitted += 1
    return n_fitted


def estimate_tumor_proportion(sx_data: SX_Data, base_props: np.ndarray):
    """Estimate per-spot tumor purity with fallback strategy.

    For each spot, tries (in order):
    1. Clonal LOH segments (strongest BAF signal)
    2. Clonal imbalanced segments (A!=B, same across all tumor clones)
    3. Fallback to theta=0.5

    Args:
        sx_data: SX_Data instance providing A, B, C, BAF, Y, D arrays.
        base_props: Array of shape (G,) with baseline proportions from normal
            spots, used to compute clone-specific RDR (mu_{g,k}).

    Returns:
        Array of shape (N,) with per-spot tumor purity estimates in [1e-4, 1-1e-4].
    """
    A_tumor = sx_data.A[:, 1:]
    B_tumor = sx_data.B[:, 1:]
    rdrs_gk = clone_rdr_gk(base_props, sx_data.C)

    # clonal LOH mask
    loh_mask = sx_data.MASK["CLONAL_LOH"]
    # clonal imbalanced mask (A!=B, same across all tumor clones)
    is_imbalanced = A_tumor[:, 0] != B_tumor[:, 0]
    is_clonal = np.ones(sx_data.G, dtype=bool)
    for k in range(1, A_tumor.shape[1]):
        is_clonal &= (A_tumor[:, k] == A_tumor[:, 0]) & (B_tumor[:, k] == B_tumor[:, 0])
    clonal_imb_mask = is_imbalanced & is_clonal

    n_loh = loh_mask.sum()
    n_imb = clonal_imb_mask.sum()
    logging.info(f"init theta: {n_loh} clonal LOH bins, {n_imb} clonal imbalanced bins")

    theta_arr = np.full(sx_data.N, 0.5, dtype=np.float32)
    remaining = np.ones(sx_data.N, dtype=bool)

    # tier 1: clonal LOH
    if n_loh > 0:
        p_loh = sx_data.BAF[loh_mask, 1]
        mu_loh = rdrs_gk[loh_mask, 1]
        Y_loh = sx_data.Y[loh_mask]
        D_loh = sx_data.D[loh_mask]
        # a spot has LOH coverage if any D > 0 in LOH bins
        has_loh = np.any(D_loh > 0, axis=0)
        n_fit = _fit_theta_spots(
            Y_loh, D_loh, p_loh, mu_loh, sx_data.N, theta_arr, remaining & has_loh
        )
        remaining[has_loh] = False
        logging.info(f"  LOH: fitted {n_fit} spots, {remaining.sum()} remaining")

    # tier 2: clonal imbalanced (for spots without LOH coverage)
    if remaining.any() and n_imb > 0:
        p_imb = sx_data.BAF[clonal_imb_mask, 1]
        mu_imb = rdrs_gk[clonal_imb_mask, 1]
        Y_imb = sx_data.Y[clonal_imb_mask]
        D_imb = sx_data.D[clonal_imb_mask]
        has_imb = np.any(D_imb > 0, axis=0)
        n_fit = _fit_theta_spots(
            Y_imb, D_imb, p_imb, mu_imb, sx_data.N, theta_arr, remaining & has_imb
        )
        remaining[has_imb] = False
        logging.info(f"  imbalanced: fitted {n_fit} spots, {remaining.sum()} remaining")

    # tier 3: fallback (already 0.5)
    if remaining.any():
        logging.info(f"  fallback: {remaining.sum()} spots left at theta=0.5")

    return theta_arr
