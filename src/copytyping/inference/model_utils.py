import numpy as np
import pandas as pd

from scipy.stats import binomtest, chi2, norm, combine_pvalues, goodness_of_fit, zscore
from statsmodels.stats.multitest import multipletests
from scipy.optimize import minimize, minimize_scalar
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF

from copytyping.sx_data.sx_data import SX_Data


##################################################
def compute_baseline_proportions(
    T: np.ndarray, Tn: np.ndarray, normal_labels: np.ndarray
) -> np.ndarray:
    T_normal = T[:, normal_labels]
    Tn_normal = Tn[normal_labels]
    base_props = np.sum(T_normal, axis=1) / np.sum(Tn_normal)
    return base_props


# def estimate_fold_change(
#     T: np.ndarray, Tn: np.ndarray, base_props: np.ndarray
# ) -> np.ndarray:
#     return


# def compute_allele_bin_proportions(
#     bin_info: pd.DataFrame,
#     num_bins: int,
#     full_props: np.ndarray,
#     norm=True,
#     bin_colname="SNP_BIN_ID",
# ):
#     """
#     full_props: (#feats, )
#     allele base props is defined over bins, and should only aggregate HET features' baseprops
#     """
#     bin_props = np.zeros(num_bins, dtype=np.float32)
#     for bin_id, feats in bin_info.groupby(bin_colname, sort=False):
#         bin_props[bin_id] = np.sum(full_props[feats.index.to_numpy()])
#     if norm:
#         bin_props = bin_props / np.sum(bin_props)
#     return bin_props


def compute_rdr(lambda_g: np.ndarray, C: np.ndarray):
    """compute mu_{g,k}=C[g,k] / sum_{g}{lam_g * C[g,k]}

    Args:
        lambda_g (np.ndarray): (G,)
        C (np.ndarray): (G,K)
    """
    denom = (lambda_g[:, None] * C).sum(axis=0)  # (K, )
    mu_gk = C / denom  # (G, K)
    return mu_gk


# linear scaling assumption
def compute_pi_gk(lambda_g: np.ndarray, C: np.ndarray, norm=True):
    """compute mu_{g,k}=lam_g * C[g,k] / sum_{g}{lam_g * C[g,k]}

    Args:
        lambda_g (np.ndarray): (G,)
        C (np.ndarray): (G,K)
    """
    props_gk = lambda_g[:, None] * (C / 2)
    if norm:
        props_gk = props_gk / np.sum(props_gk, axis=0, keepdims=True)
    return props_gk

def empirical_p_gn(Y: np.ndarray, D: np.ndarray, norm=False):
    baf_matrix = np.divide(
        Y, D, out=np.full_like(D, fill_value=np.nan, dtype=np.float32), where=D > 0
    )
    if norm:
        baf_matrix[~np.isnan(baf_matrix)] -= 0.5
        baf_matrix = zscore(baf_matrix, axis=0, nan_policy="omit")
        baf_matrix = np.where(np.isnan(baf_matrix), 0.0, baf_matrix)
    return baf_matrix

def empirical_rdr_gn(X: np.ndarray, T: np.ndarray, base_props: np.ndarray, log2=False, norm=False):
    """
    Tn*lambda_g*[(1-rho_n) + rho_n*rdr_gk]
    """
    rdr_denom = base_props[:, None] @ T[None, :]  # (G, N)
    rdr_matrix = np.divide(
        X,
        rdr_denom,
        out=np.full_like(rdr_denom, fill_value=np.nan, dtype=np.float32),
        where=rdr_denom > 0,
    )
    rdr_matrix[rdr_matrix == 0] = np.nan

    if log2:
        rdr_matrix[~np.isnan(rdr_matrix)] = np.log2(rdr_matrix[~np.isnan(rdr_matrix)])
    if norm:
        rdr_matrix = zscore(rdr_matrix, axis=0, nan_policy="omit")
        rdr_matrix = np.where(np.isnan(rdr_matrix), 0.0, rdr_matrix)
    return rdr_matrix

# TODO
def estimate_spot_proportion_loh(
    sx_data: SX_Data,
    base_props: np.ndarray,
):
    MA = sx_data.apply_allele_mask_shallow(mask_id="CLONAL_LOH")
    # assert
    if len(MA["BAF"]) == 0:
        print(f"no clonal LOH states")
    print(MA["BAF"])
    print

    pass

# def estimate_tumor_proportion_mom(
#     sx_data: SX_Data,
#     base_props: np.ndarray,
# ):
#     """
#     Estimate per-spot tumor proportion using allelic data using MoM
#     """
#     print("esimate spot proportion via MoM")
#     C = sx_data.allele_C
#     rdr_denom = (base_props[:, None] * C).sum(axis=0)  # (K, )
#     rdr_gk = C / rdr_denom  # (G, K)
#     print("rdr_denom: ", rdr_denom)

#     rdr_gk = compute_rdr(base_props, sx_data.allele_C)
#     emp_p_gn = empirical_p_gn(sx_data.Y, sx_data.D)
    
#     baf_mat = 

