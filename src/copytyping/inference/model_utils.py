import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import zscore

from copytyping.inference.likelihoods import (
    cond_betabin_logpmf_theta,
    cond_negbin_logpmf_theta,
)

if TYPE_CHECKING:
    from copytyping.inference.count_data import Count_Data


##################################################
# args → model config
##################################################


def model_kwargs_from_args(args: dict):
    """Extract the model config kwargs (the args the EM models consume) from the
    flat ``args`` dict, so callers can ``Model(..., **model_kwargs_from_args(args))``."""
    return {
        "no_normal": args["no_normal"],
        "pi_alpha": args["pi_alpha"],
        "tau_bounds": (args["min_tau"], args["max_tau"]),
        "invphi_bounds": (args["min_invphi"], args["max_invphi"]),
        "niters": args["niters"],
        "update_pi": args["update_pi"],
        "update_tau": args["update_tau"],
        "update_invphi": args["update_invphi"],
    }


def save_model_params(
    model_params: dict,
    final_ll: float,
    assay_types: list[str],
    path: str,
):
    """Save the EM log-likelihood + per-assay lambda/theta/tau/inv_phi to ``path``."""
    param_dict = {"log_likelihood": np.array([final_ll])}
    for assay_type in assay_types:
        for key in ["lambda", "theta", "tau", "inv_phi"]:
            pk = f"{assay_type}-{key}"
            if pk in model_params:
                param_dict[pk.replace("-", "_")] = np.atleast_1d(model_params[pk])
    np.savez(path, **param_dict)


##################################################
# RDR baseline
##################################################


def compute_baseline_proportions(
    X: np.ndarray,
    T: np.ndarray,
    ref_labels: np.ndarray,
    ref_cn: np.ndarray | None = None,
    eps: float = 1e-12,
):
    """Per-bin read-depth baseline from the reference-cell pseudobulk.

    ref_cn=None assumes a diploid reference (normal cells): lambda_g =
    sum_ref X_g / sum_ref T. When ref_cn (per-bin total CN of the reference
    clone) is given, divide out that clone's copy ratio so a non-diploid major
    clone yields the diploid baseline: lambda_g = (sum_ref X_g / (ref_cn_g/2)),
    normalized to sum 1. The two are identical when ref_cn == 2 everywhere.
    """
    X_ref = np.sum(X[:, ref_labels], axis=1)
    if ref_cn is None:
        return X_ref / np.sum(T[ref_labels])
    base = X_ref / np.clip(ref_cn / 2.0, eps, None)
    total = base.sum()
    return base / total if total > 0 else np.ones_like(base) / len(base)


def compute_rdr_baseline(
    count_data: "Count_Data",
    ref_cells: np.ndarray | None,
    ref_clone: int = 0,
    no_normal: bool = False,
):
    """RDR baseline (G,) from the reference-cell pseudobulk; None if no reference cells.

    CNP-corrected under ``no_normal`` (divides out the reference clone's copy ratio).
    Lets plotting consume the copytyping baseline without knowing how it's computed.
    """
    if ref_cells is None or int(ref_cells.sum()) == 0:
        return None
    ref_cn = count_data.cn_C[:, ref_clone] if no_normal else None
    T = np.asarray(count_data.count_X.sum(axis=0)).ravel()
    return compute_baseline_proportions(count_data.count_X, T, ref_cells, ref_cn=ref_cn)


##################################################
# clone-level RDR / pi
##################################################


def clone_rdr_gk(lambda_g: np.ndarray, C: np.ndarray):
    """compute mu_{g,k}=C[g,k] / sum_{g}{lam_g * C[g,k]}

    Args:
        lambda_g (np.ndarray): (G,)
        C (np.ndarray): (G,K)
    """
    denom = (lambda_g[:, None] * C).sum(axis=0)  # (K, )
    mu_gk = C / denom  # (G, K)
    return mu_gk


def clone_pi_gk(lambda_g: np.ndarray, C: np.ndarray):
    """compute pi_gk=lam_g * rdr_gk (linear scaling assumption)

    Args:
        lambda_g (np.ndarray): (G,)
        C (np.ndarray): (G,K)
    """
    props_gk = lambda_g[:, None] * C
    props_gk = props_gk / np.sum(props_gk, axis=0, keepdims=True)
    return props_gk


##################################################
# empirical BAF / RDR
##################################################


def empirical_baf_gn(Y: np.ndarray, D: np.ndarray, norm: bool = False):
    baf_matrix = np.divide(
        Y, D, out=np.full_like(D, fill_value=np.nan, dtype=np.float32), where=D > 0
    )
    if norm:
        baf_matrix[~np.isnan(baf_matrix)] -= 0.5
        baf_matrix = zscore(baf_matrix, axis=0, nan_policy="omit")
    return baf_matrix


def empirical_rdr_gn(
    X: np.ndarray,
    T: np.ndarray,
    base_props: np.ndarray,
    log2: bool = False,
    norm: bool = False,
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


##################################################
# tumor-purity init
##################################################


def estimate_tumor_proportion(
    count_data: "Count_Data",
    T: np.ndarray,
    base_props: np.ndarray,
    tau: float,
    inv_phi: float,
    fit_mode: str = "allele_total",
):
    """Per-spot purity init using BB+NB on clonal-only segments (no gamma weighting).

    Mirrors the spot model E-step likelihood but evaluated on segments shared by
    all tumor clones, where BAF_k and RDR_k are identical across k. The 1D MLE
    over theta is therefore independent of clone assignment.

    Args:
        count_data: Count_Data with count_X, count_B, count_C, cn_C, cn_BAF, allele_mask, total_mask.
        T: (N,) per-spot library size.
        base_props: (G,) baseline lambda from normal cells.
        tau: BB concentration scalar.
        inv_phi: NB inv-phi scalar.
        fit_mode: "allele", "total", or "allele_total".

    Returns:
        theta_arr: (N,) per-spot purity estimates in [1e-4, 1-1e-4].
    """
    N = count_data.num_cell
    rdrs_tumor = clone_rdr_gk(base_props, count_data.cn_C)[:, 1:]  # (G, K_tumor)
    non_sub = ~count_data.total_mask["SUBCLONAL"]
    am = count_data.allele_mask["IMBALANCED"] & non_sub & (base_props > 0)
    tm = count_data.total_mask["ANEUPLOID"] & non_sub & (base_props > 0)
    n_am = int(am.sum())
    n_tm = int(tm.sum())
    logging.info(f"init theta: {n_am} clonal imbalanced, {n_tm} clonal aneuploid bins")

    theta_arr = np.full(N, 0.5, dtype=np.float32)
    use_a = fit_mode in {"allele", "allele_total"} and n_am > 0
    use_t = fit_mode in {"total", "allele_total"} and n_tm > 0
    if not (use_a or use_t):
        logging.warning("no clonal informative bins; theta=0.5")
        return theta_arr

    if use_a:
        Y_am = count_data.count_B[am]
        D_am = count_data.count_C[am]
        BAF_am = count_data.cn_BAF[am, 1:]
        rdrs_am = rdrs_tumor[am]
    if use_t:
        X_tm = count_data.count_X[tm]
        lambda_tm = base_props[tm]
        rdrs_tm = rdrs_tumor[tm]

    for n in range(N):

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
                    np.array([T[_n]], dtype=float),
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
