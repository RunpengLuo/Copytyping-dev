import numpy as np
import pandas as pd

from scipy.special import softmax, expit, betaln, digamma, gammaln, logsumexp
from scipy.stats import binom, beta, norm
from scipy.stats import binomtest, chi2, norm, combine_pvalues, goodness_of_fit
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


def estimate_fold_change(
    T: np.ndarray, Tn: np.ndarray, base_props: np.ndarray
) -> np.ndarray:
    return


def compute_allele_bin_proportions(
    feat_info: pd.DataFrame,
    num_bins: int,
    full_props: np.ndarray,
    norm=True,
    bin_colname="SNP_BIN_ID",
):
    """
    full_props: (#feats, )
    allele base props is defined over bins, and should only aggregate HET features' baseprops
    """
    bin_props = np.zeros(num_bins, dtype=np.float32)
    for bin_id, feats in feat_info.groupby(bin_colname, sort=False):
        bin_props[bin_id] = np.sum(full_props[feats.index.to_numpy()])
    if norm:
        bin_props = bin_props / np.sum(bin_props)
    return bin_props


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
def compute_props(lambda_g: np.ndarray, C: np.ndarray, norm=True):
    """compute mu_{g,k}=lam_g * C[g,k] / sum_{g}{lam_g * C[g,k]}

    Args:
        lambda_g (np.ndarray): (G,)
        C (np.ndarray): (G,K)
    """
    props_gk = lambda_g[:, None] * (C / 2)
    if norm:
        props_gk = props_gk / np.sum(props_gk, axis=0, keepdims=True)
    return props_gk


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
