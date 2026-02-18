import os
import sys
import copy
import logging

import numpy as np
import pandas as pd


from copytyping.utils import *
from copytyping.plot.plot_common import plot_loss
from copytyping.sx_data.sx_data import *
from copytyping.inference.model_utils import *
from copytyping.inference.likelihood_funcs import *
from copytyping.inference.base_model import *

from scipy.special import logsumexp


##################################################
class Cell_Model(Base_Model):
    """Single-cell EM model, no spatial information, tumor purity=1 for each cell."""

    def __init__(
        self,
        barcodes: pd.DataFrame,
        assay_type: str,
        data_types: list,
        data_sources: dict,
        work_dir=None,
        prefix="copytyping",
        verbose=1,
    ) -> None:
        super().__init__(
            barcodes, assay_type, data_types, data_sources, work_dir, prefix, verbose
        )
        return

    ##################################################
    def _init_params(
        self,
        fit_mode: str,
        init_fix_params: dict,
        init_params: dict,
        share_params: dict,
        ref_label="cell_type",
        allele_post_thres=0.90,
        allele_max_iter=10,
    ):
        params = self._init_base_params(fit_mode, init_params)

        # initialize baseline proportions
        if (
            fit_mode in {"total_only", "hybrid"}
            and any(params.get(f"{dt}-lambda", None) is None for dt in self.data_types)
        ):
            is_normal_cell = None
            if ref_label in self.barcodes.columns:
                logging.info(
                    f"use reference label={ref_label} to label normal cells for baseline inits"
                )
                cell_types = self.barcodes[ref_label].unique()
                logging.info(f"Observed cell_types: {cell_types}")
                logging.info(f"Black list: {BLACK_LIST}")
                is_normal_cell = (~self.barcodes[ref_label].isin(BLACK_LIST)).to_numpy()
            else:
                logging.info("infer normal cells using allele-only cell model")
                pure_model = Cell_Model(
                    self.barcodes, self.assay_type, self.data_types, self.data_sources
                )
                allele_params = pure_model.fit(
                    "allele_only",
                    fix_params=init_fix_params,
                    init_params=init_params,
                    share_params=share_params,
                    max_iter=allele_max_iter,
                )
                allele_anns, clone_props = pure_model.predict(
                    "allele_only",
                    allele_params,
                    label="allele_only-label",
                    posterior_thres=allele_post_thres,
                )
                is_normal_cell = (
                    allele_anns["allele_only-label"] == "normal"
                ).to_numpy()
            num_normal_cells = np.sum(is_normal_cell)
            logging.info(
                f"#estimated normal cells={num_normal_cells}/{self.num_barcodes}"
            )
            assert num_normal_cells > 0, (
                "failed to estimate normal cells for baseline proportion estimation"
            )
            for data_type in self.data_types:
                params[f"{data_type}-lambda"] = compute_baseline_proportions(
                    self.data_sources[data_type].X,
                    self.data_sources[data_type].T,
                    is_normal_cell,
                )

        fix_params = {key: False for key in params.keys()}
        if init_fix_params is not None:
            for key in init_fix_params.keys():
                fix_params[key] = init_fix_params[key]

        return params, fix_params

    ##################################################
    def compute_log_likelihood(self, fit_mode: str, params: dict):
        """compute log-likelihoods per cell per clone. (N, K)"""
        global_lls = np.zeros((self.N, self.K), dtype=np.float32)
        # sum over all data types
        for data_type in self.data_types:
            sx_data: SX_Data = self.data_sources[data_type]
            if fit_mode in {"allele_only", "hybrid"}:
                MA, _ = sx_data.apply_mask_shallow(mask_id="IMBALANCED")
                allele_ll_mat = cond_betabin_logpmf(
                    MA["Y"], MA["D"], params[f"{data_type}-tau"], MA["BAF"]
                )
                global_lls += allele_ll_mat.sum(axis=0)

            if fit_mode in {"total_only", "hybrid"}:
                lambda_g = params[f"{data_type}-lambda"]
                nb_mask = (sx_data.MASK["ANEUPLOID"]) & (lambda_g > 0)
                props_gk_cnv = clone_pi_gk(lambda_g, sx_data.C)[nb_mask, :]
                inv_phis = params[f"{data_type}-inv_phi"][
                    lambda_g[sx_data.MASK["ANEUPLOID"]] > 0
                ]
                total_ll_mat = cond_negbin_logpmf(
                    sx_data.X[nb_mask],
                    sx_data.T,
                    props_gk_cnv,
                    inv_phis,
                )
                global_lls += total_ll_mat.sum(axis=0)

        global_lls += np.log(params["pi"])[None, :]  # (N,K)
        log_marg = logsumexp(global_lls, axis=1)  # (N,1)
        ll = np.sum(log_marg)
        return ll, log_marg, global_lls

    def _m_step(
        self,
        fit_mode: str,
        gamma: np.ndarray,
        params: dict,
        fix_params: dict,
        share_params: dict,
        invphi_bounds=(1 / 100, 1 / 10),
        logtau_bounds=(np.log(50), np.log(200)),
        t=0,
        eps=1e-10,
    ):
        if not fix_params["pi"]:
            # update mixing density for clone assignments
            params["pi"] = np.sum(gamma, axis=0) / self.N

        gamma_gnk = gamma[None, :, :]  # (1, N, K)
        for data_type in self.data_types:
            sx_data: SX_Data = self.data_sources[data_type]

            # update NB over-dispersion inv_phi
            if (
                fit_mode in {"total_only", "hybrid"}
                and not fix_params[f"{data_type}-inv_phi"]
            ):
                lambda_g = params[f"{data_type}-lambda"]
                nb_mask = (sx_data.MASK["ANEUPLOID"]) & (lambda_g > 0)
                props_gk_cnv = clone_pi_gk(lambda_g, sx_data.C)[nb_mask]
                mu_gnk = (
                    props_gk_cnv[:, None, :] * sx_data.T[None, :, None]
                )  # (G, 1, K)
                X_gnk = sx_data.X[nb_mask][:, :, None]  # (G, N, 1)

                if share_params.get(f"{data_type}-inv_phi", False):
                    params[f"{data_type}-inv_phi"][:] = mle_invphi(
                        X_gnk, mu_gnk, gamma_gnk, invphi_bounds
                    )
                else:
                    nb_valid_in_aneuploid = np.where(lambda_g[sx_data.MASK["ANEUPLOID"]] > 0)[0]
                    for local_idx, aneuploid_idx in enumerate(nb_valid_in_aneuploid):
                        params[f"{data_type}-inv_phi"][aneuploid_idx] = mle_invphi(
                            X_gnk[local_idx : local_idx + 1],
                            mu_gnk[local_idx : local_idx + 1],
                            gamma_gnk,
                            invphi_bounds,
                        )

            # update BB over-dispersion tau
            if (
                fit_mode in {"allele_only", "hybrid"}
                and not fix_params[f"{data_type}-tau"]
            ):
                bb_mask = sx_data.MASK["IMBALANCED"]
                MA, _ = sx_data.apply_mask_shallow(mask_id="IMBALANCED")
                p_gnk = MA["BAF"][:, None, :]  # (G, 1, K)
                Y_gnk = MA["Y"][:, :, None]  # (G, N, 1)
                D_gnk = MA["D"][:, :, None]  # (G, N, 1)
                if share_params.get(f"{data_type}-tau", False):
                    params[f"{data_type}-tau"][:] = mle_tau(
                        Y_gnk, D_gnk, p_gnk, gamma_gnk, logtau_bounds
                    )
                else:
                    for local_idx in range(Y_gnk.shape[0]):
                        params[f"{data_type}-tau"][local_idx] = mle_tau(
                            Y_gnk[local_idx : local_idx + 1],
                            D_gnk[local_idx : local_idx + 1],
                            p_gnk[local_idx : local_idx + 1],
                            gamma_gnk,
                            logtau_bounds,
                        )
        return

    def fit(
        self,
        fit_mode="hybrid",
        fix_params={},
        init_params={},
        share_params={},
        max_iter=100,
        tol=1e-4,
        eps=1e-10,
    ):
        assert fit_mode in allowed_fit_mode
        logging.info(f"Start cell model inference, fit_mode={fit_mode}")

        params, fix_params = self._init_params(fit_mode, fix_params, init_params, share_params)
        if self.verbose:
            self.print_params(params, fit_mode)

        ll_trace = []
        param_trace = []
        prev_ll = -np.inf
        for t in range(1, max_iter):
            if self.verbose:
                param_trace.append(copy.deepcopy(params))
            gamma = self._e_step(fit_mode, params, t)
            self._m_step(
                fit_mode, gamma, params, fix_params, share_params, t=t, eps=eps
            )

            ll, _, _ = self.compute_log_likelihood(fit_mode, params)
            ll_trace.append(ll)
            if self.verbose:
                logging.info(f"iter={t:03d} log-likelihood = {ll:.6f}")
                self.print_params(params, fit_mode)

            if t > 1:
                rel_change = np.abs(ll - prev_ll) / (np.abs(prev_ll) + eps)
                if rel_change < tol:
                    logging.info(
                        f"Converged at iteration {t} (delta = {rel_change:.2e})"
                    )
                    break
            prev_ll = ll

        if self.verbose:
            out_loss_file = os.path.join(self.work_dir, f"{self.prefix}.log_likelihoods.png")
            plot_loss(ll_trace, out_loss_file, val_type="log-likelihood")
            # TODO store parameter traces?
        return params
