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


def estimate_tumor_proportion(
    sx_data: SX_Data,
    base_props: np.ndarray,
    segment_selection: str = "clonal_imbalanced",
):
    """Estimate per-spot tumor purity using selected segments.

    For each spot, fits theta via scalar MLE using the binomial model:

        p_hat = (theta * mu_{g,k} * p_{g,k} + 0.5*(1-theta))
                / (theta * mu_{g,k} + (1-theta))
        Y_{g,n} ~ Binom(D_{g,n}, p_hat)

    Spots with no valid coverage at the selected bins are left at theta=0.5.

    Args:
        sx_data: SX_Data instance providing A, B, C, BAF, Y, D arrays.
        base_props: Array of shape (G,) with baseline proportions from normal
            spots, used to compute clone-specific RDR (mu_{g,k}).
        segment_selection: Which segments to use for theta estimation.
            - "clonal_imbalanced": A≠B, same (A,B) across all tumor clones (default)
            - "clonal_loh": LOH segments from MASK["CLONAL_LOH"]

    Returns:
        Array of shape (N,) with per-spot tumor purity estimates in [1e-4, 1-1e-4].
    """
    A_tumor = sx_data.A[:, 1:]  # (G, K_tumor)
    B_tumor = sx_data.B[:, 1:]

    if segment_selection == "clonal_loh":
        clonal_imb = sx_data.MASK["CLONAL_LOH"]
    else:
        # Imbalanced in the first tumor clone
        is_imbalanced = A_tumor[:, 0] != B_tumor[:, 0]
        # Clonal: all tumor clones carry the same (A, B) as clone 1.
        is_clonal = np.ones(sx_data.G, dtype=bool)
        for k in range(1, A_tumor.shape[1]):
            is_clonal &= (A_tumor[:, k] == A_tumor[:, 0]) & (
                B_tumor[:, k] == B_tumor[:, 0]
            )
        clonal_imb = is_imbalanced & is_clonal

    n_clonal_imb = np.sum(clonal_imb)
    logging.info(f"init theta ({segment_selection}): {n_clonal_imb} bins")

    if n_clonal_imb == 0:
        logging.warning("no clonal imbalanced bins, setting theta=0.5")
        return np.full(sx_data.N, 0.5, dtype=np.float32)

    # BAF and RDR from clone 1. Valid because:
    # - clonal_imbalanced: all tumor clones share same (A,B) by construction
    # - clonal_loh: all tumor clones have B=0 (or A=0), so BAF is identical
    rdrs_gk = clone_rdr_gk(base_props, sx_data.C)
    p_k = sx_data.BAF[clonal_imb, 1]  # (G_imb,)
    mu_k = rdrs_gk[clonal_imb, 1]  # (G_imb,)

    # Y is B-allele count matching B copy number from CNP — no orientation needed
    Y_bins = sx_data.Y[clonal_imb]  # (G_imb, N)
    D_bins = sx_data.D[clonal_imb]  # (G_imb, N)

    eps = 1e-10
    theta_arr = np.full(sx_data.N, 0.5, dtype=np.float32)

    for n in range(sx_data.N):
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

    return theta_arr
