import numpy as np
import pandas as pd

from scipy.stats import (
    binomtest,
    chi2,
    norm,
    combine_pvalues,
    goodness_of_fit,
    zscore,
    binom,
    betabinom,
)
from statsmodels.stats.multitest import multipletests
from scipy.optimize import minimize, minimize_scalar
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.base.model import GenericLikelihoodModel
from copytyping.sx_data.sx_data import SX_Data


##################################################
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
def clone_pi_gk(lambda_g: np.ndarray, C: np.ndarray, norm=True):
    """compute mu_{g,k}=lam_g * C[g,k] / sum_{g}{lam_g * C[g,k]}

    Args:
        lambda_g (np.ndarray): (G,)
        C (np.ndarray): (G,K)
    """
    props_gk = lambda_g[:, None] * (C / 2)
    if norm:
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


##################################################
class BAF_Binom(GenericLikelihoodModel):
    """
    Binomial model endog ~ Bin(exposure, p), where p = exog @ params[:-1].
    MLE: max_{params} \sum_{s} log P(endog_s | exog_s; params)

    Attributes
    ----------
    endog : array, (n_samples,)
        Y values.

    exog : array, (n_samples, n_features)
        Design matrix.

    exposure : array, (n_samples,)
        Total number of trials. In BAF case, this is the total number of SNP-covering UMIs.
    """

    def __init__(self, endog, exog, exposure, offset, scaling, **kwargs):
        super(BAF_Binom, self).__init__(endog, exog, **kwargs)
        self.exposure = exposure
        self.offset = offset
        self.scaling = scaling

    def nloglikeobs(self, params, eps=1e-10):
        linear_term = self.exog @ params
        p = self.scaling / (1 + np.exp(-linear_term + self.offset))
        p = np.clip(p, eps, 1.0 - eps)
        llf = binom.logpmf(self.endog, self.exposure, p)
        return -llf

    def fit(self, start_params=np.array([0.0]), maxiter=10_000, maxfun=5_000, **kwargs):
        return super(BAF_Binom, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs
        )

def estimate_tumor_proportion(sx_data: SX_Data, base_props: np.ndarray):
    """Estimate tumor proportion by LOH states
    base_props: (G, ) normalized baseline proportion.
    """

    def estimate_purity(
        B_bin: np.ndarray,
        D_bin: np.ndarray,
        rdr_bin: np.ndarray,
    ):
        """
        0.5 ( 1. - rho ) / (rho * RDR + 1. - rho) = B_count / Total_count for each LOH state.
        Fits the binomial model: p_{g,n} = 0.5 / (1 + rdr_{g,n} * exp(-beta_n))
        and returns rho = 1/(1+exp(beta)).
        """
        m = (
            np.isfinite(B_bin)
            & np.isfinite(D_bin)
            & np.isfinite(rdr_bin)
            & (D_bin > 0)
            & (rdr_bin > 0)
            & (B_bin >= 0)
            & (B_bin <= D_bin)
        )
        l = np.count_nonzero(m)
        if l == 0:
            return np.nan

        model = BAF_Binom(
            endog=B_bin[m],
            exog=np.ones((l, 1)),
            exposure=D_bin[m],
            offset=np.log(rdr_bin[m]),
            scaling=0.5,
        )
        res = model.fit(start_params=np.array([0.0]), disp=False)
        beta = float(np.atleast_1d(res.params)[0])
        rho = 1.0 / (1.0 + np.exp(beta))
        return float(np.clip(rho, 0.0, 1.0)), model.loglike(res.params)

    # spot by clone
    props_nk = np.full((sx_data.N, sx_data.K), np.nan, dtype=np.float32)
    lls_nk = np.full((sx_data.N, sx_data.K), np.nan, dtype=np.float32)

    rdrs_gk = clone_rdr_gk(base_props, sx_data.C)
    for n in range(sx_data.N):
        for k in range(sx_data.K):
            cna, cnb = sx_data.A[:, k], sx_data.B[:, k]
            loh_mask = np.minimum(cna, cnb) == 0
            if not np.any(loh_mask):
                continue
            loh_idxs = np.where(loh_mask)[0]
            Y_bin = sx_data.Y[loh_idxs, n]
            D_bin = sx_data.D[loh_idxs, n]
            B_bin = np.where(cnb[loh_idxs] == 0, Y_bin, D_bin - Y_bin)
            rdr_bin = rdrs_gk[loh_idxs, k]
            props_nk[n, k], lls_nk[n, k] = estimate_purity(B_bin, D_bin, rdr_bin)

    lls_nk[~np.isfinite(lls_nk)] = -np.inf
    k_star = np.argmax(lls_nk, axis=1)   # (N,)
    props = props_nk[np.arange(sx_data.N), k_star]
    return props
