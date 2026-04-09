import logging

import numpy as np

from scipy.special import logsumexp

from copytyping.inference.base_model import Base_Model
from copytyping.inference.likelihood_funcs import (
    cond_betabin_logpmf,
    cond_negbin_logpmf,
    mle_invphi,
    mle_tau,
)
from copytyping.inference.model_utils import clone_pi_gk


class Cell_Model(Base_Model):
    """Single-cell EM model. Assumes tumor purity=1 for each cell.
    Posteriors over all K clones (including normal).
    """

    def __init__(
        self,
        barcodes,
        platform,
        data_types,
        data_sources,
        work_dir=None,
        prefix="copytyping",
        verbose=1,
        modality_masks=None,
    ):
        super().__init__(
            barcodes,
            platform,
            data_types,
            data_sources,
            work_dir,
            prefix,
            verbose,
            modality_masks=modality_masks,
        )

    def _init_params(self, fit_mode, init_fix_params, init_params):
        params = self._init_base_params(fit_mode, init_params)

        if fit_mode in {"total_only", "hybrid"} and any(
            params.get(f"{dt}-lambda", None) is None for dt in self.data_types
        ):
            is_normal = self._identify_normal_cells(
                init_fix_params,
                init_params,
            )
            self._init_lambda(params, is_normal)

        # Init dispersions from neutral cluster (same as spot model)
        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]
            neutral_cids = [
                c
                for c in range(sx_data.G)
                if all(
                    sx_data.A[c, k] == 1 and sx_data.B[c, k] == 1
                    for k in range(sx_data.K)
                )
            ]
            if len(neutral_cids) > 0:
                if fit_mode in {"allele_only", "hybrid"}:
                    if len(neutral_cids) == 1:
                        Y_neut = sx_data.Y[neutral_cids[0] : neutral_cids[0] + 1]
                        D_neut = sx_data.D[neutral_cids[0] : neutral_cids[0] + 1]
                    else:
                        Y_neut = sx_data.Y[neutral_cids].sum(axis=0, keepdims=True)
                        D_neut = sx_data.D[neutral_cids].sum(axis=0, keepdims=True)
                    Y_fit = Y_neut[:, :, None].astype(np.float64)
                    D_fit = D_neut[:, :, None].astype(np.float64)
                    global_tau = mle_tau(
                        Y_fit,
                        D_fit,
                        np.full_like(Y_fit, 0.5),
                        np.ones_like(Y_fit),
                        self._logtau_bounds,
                    )
                    params[f"{data_type}-tau"][:] = global_tau
                    logging.info(f"global tau={global_tau:.2f}")

                if (
                    fit_mode in {"total_only", "hybrid"}
                    and f"{data_type}-lambda" in params
                ):
                    lambda_g = params[f"{data_type}-lambda"]
                    lam_neut = (
                        lambda_g[neutral_cids].sum()
                        if len(neutral_cids) > 1
                        else lambda_g[neutral_cids[0]]
                    )
                    if len(neutral_cids) == 1:
                        X_neut = sx_data.X[neutral_cids[0] : neutral_cids[0] + 1]
                    else:
                        X_neut = sx_data.X[neutral_cids].sum(axis=0, keepdims=True)
                    X_fit = X_neut[:, :, None].astype(np.float64)
                    mu_fit = (sx_data.T[None, :, None] * lam_neut).astype(np.float64)
                    global_invphi = mle_invphi(
                        X_fit,
                        mu_fit,
                        np.ones_like(X_fit),
                        self._invphi_bounds,
                    )
                    params[f"{data_type}-inv_phi"][:] = global_invphi
                    logging.info(
                        f"global inv_phi={global_invphi:.4f} (phi={1 / global_invphi:.2f})"
                    )

        fix_params = {key: False for key in params.keys()}
        if init_fix_params is not None:
            for key in init_fix_params:
                fix_params[key] = init_fix_params[key]

        return params, fix_params

    def compute_log_likelihood(self, fit_mode: str, params: dict):
        global_lls = np.zeros((self.N, self.K), dtype=np.float32)

        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]
            mask_n = self.modality_masks[data_type]

            if fit_mode in {"allele_only", "hybrid"}:
                MA, _ = sx_data.apply_mask_shallow(mask_id="IMBALANCED")
                allele_ll = cond_betabin_logpmf(
                    MA["Y"], MA["D"], params[f"{data_type}-tau"], MA["BAF"]
                )
                contrib = allele_ll.sum(axis=0)
                contrib[~mask_n, :] = 0.0
                global_lls += contrib

            if fit_mode in {"total_only", "hybrid"}:
                lambda_g = params[f"{data_type}-lambda"]
                nb_mask = sx_data.MASK["ANEUPLOID"] & (lambda_g > 0)
                props_gk = clone_pi_gk(lambda_g, sx_data.C)[nb_mask, :]
                inv_phis = params[f"{data_type}-inv_phi"][
                    lambda_g[sx_data.MASK["ANEUPLOID"]] > 0
                ]
                total_ll = cond_negbin_logpmf(
                    sx_data.X[nb_mask],
                    sx_data.T,
                    props_gk,
                    inv_phis,
                )
                contrib = total_ll.sum(axis=0)
                contrib[~mask_n, :] = 0.0
                global_lls += contrib

        global_lls += np.log(params["pi"])[None, :]
        log_marg = logsumexp(global_lls, axis=1)
        return np.sum(log_marg), log_marg, global_lls

    def _m_step(self, fit_mode, gamma, params, fix_params, t=0, eps=1e-10):
        self._update_pi(gamma, params, fix_params, self.N, self.K)

        gamma_gnk = gamma[None, :, :]  # (1, N, K)
        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]

            # NB dispersion (shared across all bins)
            if (
                fit_mode in {"total_only", "hybrid"}
                and not fix_params[f"{data_type}-inv_phi"]
            ):
                lambda_g = params[f"{data_type}-lambda"]
                nb_mask = sx_data.MASK["ANEUPLOID"] & (lambda_g > 0)
                props_gk = clone_pi_gk(lambda_g, sx_data.C)[nb_mask]
                mu_gnk = props_gk[:, None, :] * sx_data.T[None, :, None]
                X_gnk = sx_data.X[nb_mask][:, :, None]
                params[f"{data_type}-inv_phi"][:] = mle_invphi(
                    X_gnk, mu_gnk, gamma_gnk, self._invphi_bounds
                )

            # BB dispersion (shared across all bins)
            if (
                fit_mode in {"allele_only", "hybrid"}
                and not fix_params[f"{data_type}-tau"]
            ):
                MA, _ = sx_data.apply_mask_shallow(mask_id="IMBALANCED")
                p_gnk = MA["BAF"][:, None, :]
                Y_gnk = MA["Y"][:, :, None]
                D_gnk = MA["D"][:, :, None]
                params[f"{data_type}-tau"][:] = mle_tau(
                    Y_gnk, D_gnk, p_gnk, gamma_gnk, self._logtau_bounds
                )
