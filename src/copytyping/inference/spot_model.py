import os
import sys
import copy

import numpy as np
import pandas as pd

from copytyping.utils import *
from copytyping.plot.plot_common import plot_loss
from copytyping.sx_data.sx_data import *
from copytyping.inference.model_utils import *
from copytyping.inference.cell_model import Cell_Model
from copytyping.inference.likelihood_funcs import *
from copytyping.inference.base_model import *

from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


##################################################
class Spot_Model(Base_Model):
    """Spot EM model, estimate tumor purity for each spot."""

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
        ref_label="path_label",
        allele_post_thres=0.90,
        allele_max_iter=10,
    ):
        params = self._init_base_params(fit_mode, init_params)
        data_type = self.data_types[0]
        sx_data: SX_Data = self.data_sources[data_type]

        # pre-label normal cells to init baselines and tumor purities
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
            is_normal_cell = (allele_anns["allele_only-label"] == "normal").to_numpy()
        num_normal_cells = np.sum(is_normal_cell)
        logging.info(f"#estimated normal cells={num_normal_cells}/{self.num_barcodes}")
        assert num_normal_cells > 0, (
            "failed to estimate normal cells for baseline proportion estimation"
        )

        # initialize baseline proportions
        if params.get(f"{data_type}-lambda", None) is None:
            params[f"{data_type}-lambda"] = compute_baseline_proportions(
                sx_data.X,
                sx_data.T,
                is_normal_cell,
            )

        # initialize tumor purity
        if params.get(f"{data_type}-theta", None) is None:
            params[f"{data_type}-theta"] = estimate_tumor_proportion(
                sx_data, params[f"{data_type}-lambda"]
            )

        fix_params = {key: False for key in params.keys()}
        if not init_fix_params is None:
            for key in init_fix_params.keys():
                fix_params[key] = init_fix_params[key]

        return params, fix_params

    ##################################################
    def compute_log_likelihood(self, fit_mode: str, params: dict):
        """compute log-likelihoods per spot per clone. (N, K)"""
        global_lls = np.zeros((self.N, self.K), dtype=np.float32)

        data_type = self.data_types[0]
        sx_data: SX_Data = self.data_sources[data_type]

        lambda_g = params[f"{data_type}-lambda"]  # (G,)
        rdrs_gk = clone_rdr_gk(lambda_g, sx_data.C)

        bb_mask = (sx_data.MASK["IMBALANCED"]) & (lambda_g > 0)
        nb_mask = (sx_data.MASK["ANEUPLOID"]) & (lambda_g > 0)

        if fit_mode in {"allele_only", "hybrid"}:
            MA = sx_data.apply_mask_shallow(
                mask_id="IMBALANCED", additional_mask=lambda_g > 0
            )
            allele_ll_mat = cond_betabin_logpmf_theta(
                MA["Y"],
                MA["D"],
                params[f"{data_type}-tau"][lambda_g[sx_data.MASK["IMBALANCED"]] > 0],
                MA["BAF"],
                rdrs_gk[bb_mask],
                params[f"{data_type}-theta"],
            )
            allele_lls = allele_ll_mat.sum(axis=0)  # (N,K)
            global_lls += allele_lls

        if fit_mode in {"total_only", "hybrid"}:
            inv_phis = params[f"{data_type}-inv_phi"][
                lambda_g[sx_data.MASK["ANEUPLOID"]] > 0
            ]
            total_ll_mat = cond_negbin_logpmf_theta(
                sx_data.X[nb_mask],
                sx_data.T,
                lambda_g[nb_mask],
                inv_phis,
                rdrs_gk[nb_mask],
                params[f"{data_type}-theta"],
            )
            total_lls = total_ll_mat.sum(axis=0)  # (N,K)
            global_lls += total_lls

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
        purity_bounds=(1e-4, 1.0 - 1e-4),
        t=0,
        eps=1e-10,
    ):
        if not fix_params["pi"]:
            # update mixing density for clone assignments
            params["pi"] = np.sum(gamma, axis=0) / self.N

        gamma_gnk = gamma[None, :, :]  # (1, N, K)
        data_type = self.data_types[0]
        sx_data: SX_Data = self.data_sources[data_type]

        # parameters
        lambda_g = params[f"{data_type}-lambda"]
        rdrs_gk = clone_rdr_gk(lambda_g, sx_data.C)
        # props_gk = clone_pi_gk(lambda_g, sx_data.C)
        p_gk = sx_data.BAF

        nb_mask = (sx_data.MASK["ANEUPLOID"]) & (lambda_g > 0)
        bb_mask = (sx_data.MASK["IMBALANCED"]) & (lambda_g > 0)

        # dispersions
        inv_phi_g = params[f"{data_type}-inv_phi"][nb_mask]
        tau_g = params[f"{data_type}-tau"][bb_mask]

        # inputs
        X_gn = sx_data.X[nb_mask, :]  # (G, N)
        T_n = sx_data.T  # (N,)
        Y_gn = sx_data.Y[bb_mask, :]  # (G, N)
        D_gn = sx_data.D[bb_mask, :]  # (G, N)

        ##################################################
        # update tumor proportion via MLE
        if not fix_params[f"{data_type}-theta"]:
            theta_arr = np.zeros_like(params[f"{data_type}-theta"], dtype=np.float32)
            for n in range(self.N):

                def neg_Q_theta(theta):
                    theta = np.array([theta], dtype=float)
                    ll_nb = cond_negbin_logpmf_theta(
                        X=X_gn[:, n : n + 1],
                        T=np.array([T_n[n]], dtype=float),
                        lam_g=lambda_g[nb_mask],
                        inv_phi=inv_phi_g,
                        rdrs_gk=rdrs_gk[nb_mask],
                        theta=theta,
                    )  # (Gtot, 1, K)

                    ll_bb = cond_betabin_logpmf_theta(
                        Y=Y_gn[:, n : n + 1],
                        D=D_gn[:, n : n + 1],
                        tau=tau_g,
                        p=p_gk[bb_mask],
                        rdrs_gk=rdrs_gk[bb_mask],
                        theta=theta,
                    )

                    Q = np.sum(ll_nb[:, 0, :] * gamma[n][None, :])
                    Q += np.sum(ll_bb[:, 0, :] * gamma[n][None, :])
                    return -Q

                res = minimize_scalar(
                    neg_Q_theta,
                    bounds=purity_bounds,
                    method="bounded",
                )
                theta_arr[n] = np.clip(res.x, 1e-4, 1.0 - 1e-4)
            params[f"{data_type}-theta"] = theta_arr

        ##################################################
        # update NB over-dispersion inv_phi
        if (
            fit_mode in {"total_only", "hybrid"}
            and not fix_params[f"{data_type}-inv_phi"]
        ):
            X_gnk = X_gn[:, :, None]
            T_gnk = T_n[None, :, None]
            lam_gnk = lambda_g[nb_mask][:, None, None]
            rdrs_gnk = rdrs_gk[nb_mask][:, None, :]
            theta_gnk = params[f"{data_type}-theta"][None, :, None]

            mu_gnk = T_gnk * lam_gnk * (theta_gnk * rdrs_gnk + (1.0 - theta_gnk))
            # mu_gnk = np.clip(mu_gnk, eps, None)

            if share_params.get(f"{data_type}-inv_phi", False):
                params[f"{data_type}-inv_phi"][:] = mle_invphi(
                    X_gnk, mu_gnk, gamma_gnk, invphi_bounds
                )
            else:
                for idx, row in sx_data.cnv_blocks[nb_mask].iterrows():
                    params[f"{data_type}-inv_phi"][idx] = mle_invphi(
                        X_gnk[idx], mu_gnk[idx], gamma_gnk[idx], invphi_bounds
                    )

        ##################################################
        # update BB over-dispersion tau
        if fit_mode in {"allele_only", "hybrid"} and not fix_params[f"{data_type}-tau"]:
            Y_gnk = Y_gn[:, :, None]
            D_gnk = D_gn[:, :, None]
            rdrs_gnk = rdrs_gk[bb_mask][:, None, :]
            theta_gnk = params[f"{data_type}-theta"][None, :, None]
            p_gnk = p_gk[bb_mask][:, None, :]

            denom = rdrs_gnk * theta_gnk + (1.0 - theta_gnk)
            num = p_gnk * rdrs_gnk * theta_gnk + 0.5 * (1.0 - theta_gnk)
            p_hat = num / np.clip(denom, eps, None)
            p_hat = np.clip(p_hat, eps, 1.0 - eps)

            if share_params.get(f"{data_type}-tau", False):
                tau_hat = mle_tau(Y_gnk, D_gnk, p_hat, gamma_gnk, logtau_bounds)
                params[f"{data_type}-tau"][:] = tau_hat
            else:
                for idx, row in sx_data.cnv_blocks[bb_mask].iterrows():
                    params[f"{data_type}-tau"][idx] = mle_tau(
                        Y_gnk[idx],
                        D_gnk[idx],
                        p_gnk[idx],
                        gamma_gnk[idx],
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
        logging.info(f"Start spot model inference, fit_mode={fit_mode}")

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
                fit_mode,
                gamma,
                params,
                fix_params,
                share_params,
                t=t,
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
