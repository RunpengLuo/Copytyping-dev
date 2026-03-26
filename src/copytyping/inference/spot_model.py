import os
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
        # latent variable ranges over tumor clones only (normal modeled by 1-theta)
        self.tumor_clones = self.clones[1:]
        self.K_tumor = len(self.tumor_clones)
        return

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
                f"use reference label={ref_label} to label normal spots for baseline inits"
            )
            cell_types = self.barcodes[ref_label].unique()
            logging.info(f"Observed cell_types: {cell_types}")
            logging.info(f"Black list: {BLACK_LIST_CELLTYPE}")
            is_normal_cell = (
                ~self.barcodes[ref_label].isin(BLACK_LIST_CELLTYPE)
            ).to_numpy()
        else:
            logging.info("infer normal cells using allele-only BB model")
            pure_model = Cell_Model(
                self.barcodes,
                self.assay_type,
                self.data_types,
                self.data_sources,
                work_dir=self.work_dir,
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
                sx_data,
                params[f"{data_type}-lambda"],
            )

        # estimate BB dispersion from normal spots (known genotype p=0.5)
        if fit_mode in {"allele_only", "hybrid"}:
            tau_key = f"{data_type}-tau"
            imb_mask = sx_data.MASK["IMBALANCED"]
            Y_normal = sx_data.Y[imb_mask][:, is_normal_cell]
            D_normal = sx_data.D[imb_mask][:, is_normal_cell]
            logtau_bounds = (
                np.log(init_params["min_tau"]),
                np.log(init_params["max_tau"]),
            )
            n_imb = imb_mask.sum()
            tau_from_normal = np.full(n_imb, init_params["tau0"], dtype=np.float32)
            for g in range(n_imb):
                y = Y_normal[g : g + 1, :, None].astype(np.float64)
                d = D_normal[g : g + 1, :, None].astype(np.float64)
                if d.sum() == 0:
                    continue
                p = np.full_like(y, 0.5)
                w = np.ones_like(y)
                tau_from_normal[g] = mle_tau(y, d, p, w, logtau_bounds)
            params[tau_key] = tau_from_normal
            logging.info(
                f"BB tau from normal spots: "
                f"median={np.median(tau_from_normal):.1f}, "
                f"mean={np.mean(tau_from_normal):.1f}"
            )

        # estimate NB dispersion from normal spots (known RDR=1)
        if fit_mode in {"total_only", "hybrid"}:
            invphi_key = f"{data_type}-inv_phi"
            ane_mask = sx_data.MASK["ANEUPLOID"]
            lambda_g = params[f"{data_type}-lambda"]
            X_normal = sx_data.X[ane_mask][:, is_normal_cell]
            T_normal = sx_data.T[is_normal_cell]
            lam_ane = lambda_g[ane_mask]
            invphi_bounds = (1 / init_params["max_phi"], 1 / init_params["min_phi"])
            n_ane = ane_mask.sum()
            invphi_from_normal = np.full(
                n_ane, 1 / init_params["phi0"], dtype=np.float32
            )
            for g in range(n_ane):
                x = X_normal[g : g + 1, :, None].astype(np.float64)
                mu = (T_normal[None, :, None] * lam_ane[g]).astype(np.float64)
                if mu.sum() == 0:
                    continue
                w = np.ones_like(x)
                invphi_from_normal[g] = mle_invphi(x, mu, w, invphi_bounds)
            params[invphi_key] = invphi_from_normal
            logging.info(
                f"NB inv_phi from normal spots: "
                f"median={np.median(invphi_from_normal):.1f}, "
                f"mean={np.mean(invphi_from_normal):.1f}"
            )

        fix_params = {key: False for key in params.keys()}
        if init_fix_params is not None:
            for key in init_fix_params.keys():
                fix_params[key] = init_fix_params[key]

        return params, fix_params

    def compute_log_likelihood(self, fit_mode: str, params: dict):
        """compute log-likelihoods per tumor spot per tumor clone. (N_tumor, K_tumor)"""
        N_t = self._N_tumor
        global_lls = np.zeros((N_t, self.K_tumor), dtype=np.float32)

        data_type = self.data_types[0]
        sx_data: SX_Data = self.data_sources[data_type]
        tumor_idx = self._tumor_idx

        lambda_g = params[f"{data_type}-lambda"]  # (G,)
        rdrs_gk = clone_rdr_gk(lambda_g, sx_data.C)[:, 1:]  # (G, K_tumor)

        bb_mask = (sx_data.MASK["IMBALANCED"]) & (lambda_g > 0)
        nb_mask = (sx_data.MASK["ANEUPLOID"]) & (lambda_g > 0)

        theta_tumor = params[f"{data_type}-theta"][tumor_idx]

        if fit_mode in {"allele_only", "hybrid"}:
            MA, _ = sx_data.apply_mask_shallow(
                mask_id="IMBALANCED", additional_mask=lambda_g > 0
            )
            allele_ll_mat = cond_betabin_logpmf_theta(
                MA["Y"][:, tumor_idx],
                MA["D"][:, tumor_idx],
                params[f"{data_type}-tau"][lambda_g[sx_data.MASK["IMBALANCED"]] > 0],
                MA["BAF"][:, 1:],
                rdrs_gk[bb_mask],
                theta_tumor,
            )
            global_lls += allele_ll_mat.sum(axis=0)

        if fit_mode in {"total_only", "hybrid"}:
            inv_phis = params[f"{data_type}-inv_phi"][
                lambda_g[sx_data.MASK["ANEUPLOID"]] > 0
            ]
            total_ll_mat = cond_negbin_logpmf_theta(
                sx_data.X[nb_mask][:, tumor_idx],
                sx_data.T[tumor_idx],
                lambda_g[nb_mask],
                inv_phis,
                rdrs_gk[nb_mask],
                theta_tumor,
            )
            global_lls += total_ll_mat.sum(axis=0)

        global_lls += np.log(params["pi"])[None, :]
        log_marg = logsumexp(global_lls, axis=1)
        ll = np.sum(log_marg)
        return ll, log_marg, global_lls

    def _m_step(
        self,
        fit_mode: str,
        gamma: np.ndarray,
        params: dict,
        fix_params: dict,
        share_params: dict,
        t=0,
        eps=1e-10,
    ):
        """M-step using tumor spots only. gamma shape: (N_tumor, K_tumor)."""
        N_t = self._N_tumor
        tumor_idx = self._tumor_idx

        if not fix_params["pi"]:
            alpha = self._pi_alpha
            N_k = np.sum(gamma, axis=0)  # (K_tumor,)
            pi = (N_k + alpha - 1) / (N_t + self.K_tumor * (alpha - 1))
            pi = np.clip(pi, 0, None)
            if pi.sum() > 0:
                pi = pi / pi.sum()
            else:
                pi = np.ones(self.K_tumor) / self.K_tumor
            params["pi"] = pi

        gamma_gnk = gamma[None, :, :]  # (1, N_tumor, K_tumor)
        data_type = self.data_types[0]
        sx_data: SX_Data = self.data_sources[data_type]

        lambda_g = params[f"{data_type}-lambda"]
        rdrs_gk = clone_rdr_gk(lambda_g, sx_data.C)[:, 1:]
        p_gk = sx_data.BAF[:, 1:]

        nb_mask = (sx_data.MASK["ANEUPLOID"]) & (lambda_g > 0)
        bb_mask = (sx_data.MASK["IMBALANCED"]) & (lambda_g > 0)

        T_tumor = sx_data.T[tumor_idx]
        theta_tumor = params[f"{data_type}-theta"][tumor_idx]

        # update tumor purity via per-spot bounded MLE
        if not fix_params[f"{data_type}-theta"]:
            X_gn_all = sx_data.X
            Y_gn_all = sx_data.Y
            D_gn_all = sx_data.D
            theta_arr = params[f"{data_type}-theta"].copy()
            purity_bounds = (1e-4, 1.0 - 1e-4)
            for n in range(self.N):

                def neg_Q_theta(theta_val):
                    theta_val = np.array([theta_val], dtype=float)
                    Q = 0.0
                    if fit_mode in {"total_only", "hybrid"}:
                        ll_nb = cond_negbin_logpmf_theta(
                            X=X_gn_all[nb_mask][:, n : n + 1],
                            T=np.array([sx_data.T[n]], dtype=float),
                            lam_g=lambda_g[nb_mask],
                            inv_phi=params[f"{data_type}-inv_phi"][
                                lambda_g[sx_data.MASK["ANEUPLOID"]] > 0
                            ],
                            rdrs_gk=rdrs_gk[nb_mask],
                            theta=theta_val,
                        )
                        # use uniform weight over clones for non-tumor spots
                        if n in tumor_idx:
                            tidx = np.searchsorted(tumor_idx, n)
                            w = gamma[tidx][None, :]
                        else:
                            w = np.ones((1, self.K_tumor)) / self.K_tumor
                        Q += np.sum(ll_nb[:, 0, :] * w)
                    if fit_mode in {"allele_only", "hybrid"}:
                        ll_bb = cond_betabin_logpmf_theta(
                            Y=Y_gn_all[bb_mask][:, n : n + 1],
                            D=D_gn_all[bb_mask][:, n : n + 1],
                            tau=params[f"{data_type}-tau"][
                                lambda_g[sx_data.MASK["IMBALANCED"]] > 0
                            ],
                            p=p_gk[bb_mask],
                            rdrs_gk=rdrs_gk[bb_mask],
                            theta=theta_val,
                        )
                        if n in tumor_idx:
                            tidx = np.searchsorted(tumor_idx, n)
                            w = gamma[tidx][None, :]
                        else:
                            w = np.ones((1, self.K_tumor)) / self.K_tumor
                        Q += np.sum(ll_bb[:, 0, :] * w)
                    return -Q

                res = minimize_scalar(
                    neg_Q_theta, bounds=purity_bounds, method="bounded"
                )
                theta_arr[n] = np.clip(res.x, 1e-4, 1.0 - 1e-4)
            params[f"{data_type}-theta"] = theta_arr
            # refresh tumor views
            theta_tumor = params[f"{data_type}-theta"][tumor_idx]

        if (
            fit_mode in {"total_only", "hybrid"}
            and not fix_params[f"{data_type}-inv_phi"]
        ):
            inv_phi_g = params[f"{data_type}-inv_phi"][
                lambda_g[sx_data.MASK["ANEUPLOID"]] > 0
            ]
            X_gn = sx_data.X[nb_mask][:, tumor_idx]  # (G_nb, N_tumor)

            X_gnk = X_gn[:, :, None]
            T_gnk = T_tumor[None, :, None]
            lam_gnk = lambda_g[nb_mask][:, None, None]
            rdrs_gnk = rdrs_gk[nb_mask][:, None, :]
            theta_gnk = theta_tumor[None, :, None]

            mu_gnk = T_gnk * lam_gnk * (theta_gnk * rdrs_gnk + (1.0 - theta_gnk))

            if share_params.get(f"{data_type}-inv_phi", False):
                params[f"{data_type}-inv_phi"][:] = mle_invphi(
                    X_gnk, mu_gnk, gamma_gnk, self._invphi_bounds
                )
            else:
                nb_valid_in_aneuploid = np.where(
                    lambda_g[sx_data.MASK["ANEUPLOID"]] > 0
                )[0]
                for local_idx, aneuploid_idx in enumerate(nb_valid_in_aneuploid):
                    params[f"{data_type}-inv_phi"][aneuploid_idx] = mle_invphi(
                        X_gnk[local_idx : local_idx + 1],
                        mu_gnk[local_idx : local_idx + 1],
                        gamma_gnk,
                        self._invphi_bounds,
                    )

        if fit_mode in {"allele_only", "hybrid"} and not fix_params[f"{data_type}-tau"]:
            Y_gn = sx_data.Y[bb_mask][:, tumor_idx]
            D_gn = sx_data.D[bb_mask][:, tumor_idx]

            Y_gnk = Y_gn[:, :, None]
            D_gnk = D_gn[:, :, None]
            rdrs_gnk = rdrs_gk[bb_mask][:, None, :]
            theta_gnk = theta_tumor[None, :, None]
            p_gnk = p_gk[bb_mask][:, None, :]

            denom = rdrs_gnk * theta_gnk + (1.0 - theta_gnk)
            num = p_gnk * rdrs_gnk * theta_gnk + 0.5 * (1.0 - theta_gnk)
            p_hat = num / np.clip(denom, eps, None)
            p_hat = np.clip(p_hat, eps, 1.0 - eps)

            if share_params.get(f"{data_type}-tau", False):
                tau_hat = mle_tau(Y_gnk, D_gnk, p_hat, gamma_gnk, self._logtau_bounds)
                params[f"{data_type}-tau"][:] = tau_hat
            else:
                bb_valid_in_imbalanced = np.where(
                    lambda_g[sx_data.MASK["IMBALANCED"]] > 0
                )[0]
                for local_idx, imbalanced_idx in enumerate(bb_valid_in_imbalanced):
                    params[f"{data_type}-tau"][imbalanced_idx] = mle_tau(
                        Y_gnk[local_idx : local_idx + 1],
                        D_gnk[local_idx : local_idx + 1],
                        p_hat[local_idx : local_idx + 1],
                        gamma_gnk,
                        self._logtau_bounds,
                    )
        return

    def fit(
        self,
        fit_mode="hybrid",
        fix_params={},
        init_params={},
        share_params={},
        max_iter=100,
        purity_threshold=0.5,
        tol=1e-4,
        eps=1e-10,
    ):
        assert fit_mode in allowed_fit_mode
        logging.info(f"Start spot model inference, fit_mode={fit_mode}")

        params, fix_params = self._init_params(
            fit_mode, fix_params, init_params, share_params
        )

        # store hyperparams for M-step
        self._invphi_bounds = (1 / init_params["max_phi"], 1 / init_params["min_phi"])
        self._logtau_bounds = (
            np.log(init_params["min_tau"]),
            np.log(init_params["max_tau"]),
        )
        self._pi_alpha = init_params.get("pi_alpha", 0.5)

        # all spots participate in EM; theta is fixed and determines purity weighting
        self._tumor_idx = np.arange(self.N)
        self._N_tumor = self.N
        logging.info(f"EM on {self.N} spots (all, theta fixed)")

        if self.verbose:
            self.print_params(params, fit_mode)

        ll_trace = []
        param_trace = []
        prev_ll = -np.inf
        for t in range(1, max_iter):
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
            out_loss_file = os.path.join(
                self.work_dir, f"{self.prefix}.log_likelihoods.png"
            )
            plot_loss(ll_trace, out_loss_file, val_type="log-likelihood")
        self.param_trace = param_trace
        self.save_param_trace(param_trace)
        return params

    def predict(
        self,
        fit_mode: str,
        params: dict,
        label: str,
        posterior_thres: float = 0.5,
        margin_thres: float = 0.1,
        tumorprop_threshold: float = 0.5,
    ):
        logging.info("Decode labels with MAP estimation")
        # posteriors for all spots (N, K_tumor)
        posteriors = self._e_step(fit_mode, params)
        tumor_clones = self.clones[1:]

        anns = self.barcodes.copy(deep=True)
        theta_key = f"{self.data_types[0]}-theta"
        anns["tumor_purity"] = params[theta_key]

        # init all posteriors to 0, fill spots in EM
        for c in tumor_clones:
            anns[c] = 0.0
        anns.iloc[self._tumor_idx, anns.columns.get_indexer(tumor_clones)] = posteriors

        probs = anns[tumor_clones].to_numpy()
        probs_sorted = np.sort(probs, axis=1)
        anns["max_posterior"] = probs_sorted[:, -1]
        if probs_sorted.shape[1] > 1:
            anns["margin_delta"] = probs_sorted[:, -1] - probs_sorted[:, -2]
        else:
            anns["margin_delta"] = 1.0

        # hard classify all spots to tumor clones by MAP (no purity threshold)
        anns[label] = anns[tumor_clones].idxmax(axis=1)

        clone_props = {
            clone: np.mean(anns[label].to_numpy() == clone) for clone in tumor_clones
        }
        logging.info(f"clone fractions (all spots): {clone_props}")
        return anns, clone_props
