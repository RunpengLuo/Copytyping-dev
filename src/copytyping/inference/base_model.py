import os
import copy
import logging

import numpy as np
import pandas as pd

from scipy.special import logsumexp

from copytyping.inference.likelihood_funcs import mle_invphi, mle_tau
from copytyping.inference.model_utils import compute_baseline_proportions
from copytyping.plot.plot_common import plot_loss
from copytyping.sx_data.sx_data import SX_Data
from copytyping.utils import is_normal_label

allowed_fit_mode = {"hybrid", "allele_only", "total_only"}


class Base_Model:
    def __init__(
        self,
        barcodes: pd.DataFrame,
        platform: str,
        data_types: list,
        data_sources: dict[str, SX_Data],
        work_dir=None,
        prefix="copytyping",
        verbose=1,
        modality_masks: dict = None,
    ) -> None:
        self.barcodes = barcodes
        self.data_types = data_types
        self.platform = platform
        self.data_sources = data_sources
        self.clones = self.data_sources[self.data_types[0]].clones
        self.N = self.num_barcodes = len(barcodes)
        self.K = self.num_clones = len(self.clones)
        self.work_dir = work_dir
        self.prefix = prefix
        self.verbose = verbose
        if modality_masks is None:
            modality_masks = {dt: np.ones(self.N, dtype=bool) for dt in data_types}
        self.modality_masks = modality_masks

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _init_base_params(self, fit_mode: str, init_params: dict):
        params = {}

        params["pi"] = init_params.get("pi", None)
        if params.get("pi", None) is None:
            params["pi"] = np.ones(self.K) / self.K

        if fit_mode in {"total_only", "hybrid"}:
            for data_type in self.data_types:
                if params.get(f"{data_type}-inv_phi", None) is None:
                    params[f"{data_type}-inv_phi"] = np.full(
                        self.data_sources[data_type].nrows_aneuploid,
                        fill_value=1 / init_params["phi0"],
                        dtype=np.float32,
                    )

        if fit_mode in {"allele_only", "hybrid"}:
            for data_type in self.data_types:
                if params.get(f"{data_type}-tau", None) is None:
                    params[f"{data_type}-tau"] = np.full(
                        self.data_sources[data_type].nrows_imbalanced,
                        fill_value=init_params["tau0"],
                        dtype=np.float32,
                    )
        return params

    def _identify_normal_cells(
        self,
        init_fix_params,
        init_params,
        ref_label="cell_type",
        allele_post_thres=0.90,
        allele_max_iter=10,
    ):
        """Identify normal cells/spots from reference labels or allele-only sub-EM."""
        from copytyping.inference.cell_model import Cell_Model

        is_normal = None
        if ref_label in self.barcodes.columns:
            logging.info(f"use reference label={ref_label} for normal identification")
            is_normal = self.barcodes[ref_label].apply(is_normal_label).to_numpy()

        if is_normal is None or np.sum(is_normal) == 0:
            logging.info("infer normal cells using allele-only BB model")
            pure_model = Cell_Model(
                self.barcodes,
                self.platform,
                self.data_types,
                self.data_sources,
                work_dir=self.work_dir,
                modality_masks=self.modality_masks,
            )
            # Strip pi from init_params so Cell_Model uses uniform init
            cell_init = {k: v for k, v in init_params.items() if k != "pi"}
            allele_params = pure_model.fit(
                "allele_only",
                fix_params=init_fix_params,
                init_params=cell_init,
                max_iter=allele_max_iter,
            )
            allele_anns, _ = pure_model.predict(
                "allele_only",
                allele_params,
                label="allele_only-label",
                posterior_thres=allele_post_thres,
            )
            is_normal = (allele_anns["allele_only-label"] == "normal").to_numpy()

        n_normal = np.sum(is_normal)
        logging.info(f"#normal cells/spots={n_normal}/{self.num_barcodes}")
        assert n_normal > 0, "no normal cells/spots found for baseline estimation"
        return is_normal

    def _init_lambda(self, params, is_normal):
        """Compute baseline proportions from normal cells for all data types."""
        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]
            params[f"{data_type}-lambda"] = compute_baseline_proportions(
                sx_data.X,
                sx_data.T,
                is_normal,
            )

    def _init_tau_from_normals(self, data_type, params, is_normal):
        """Init BB tau via MLE on normal cells at imbalanced segments."""
        sx_data = self.data_sources[data_type]
        tau_key = f"{data_type}-tau"
        if tau_key not in params or np.sum(is_normal) == 0:
            return
        imb = sx_data.MASK["IMBALANCED"]
        Y_norm = sx_data.Y[imb][:, is_normal][:, :, None].astype(np.float64)
        D_norm = sx_data.D[imb][:, is_normal][:, :, None].astype(np.float64)
        tau_init = mle_tau(
            Y_norm,
            D_norm,
            np.full_like(Y_norm, 0.5),
            np.ones_like(Y_norm),
        )
        params[tau_key][:] = tau_init
        logging.info(
            f"{data_type}: tau init from {np.sum(is_normal)} normal "
            f"cells/spots x {np.sum(imb)} imbalanced segments "
            f"= {tau_init:.2f}"
        )

    def _init_invphi_from_normals(self, data_type, params, is_normal):
        """Init NB inv_phi via MLE on normal cells at aneuploid segments."""
        sx_data = self.data_sources[data_type]
        invphi_key = f"{data_type}-inv_phi"
        lambda_key = f"{data_type}-lambda"
        if invphi_key not in params or lambda_key not in params:
            return
        if np.sum(is_normal) == 0:
            return
        aneu = sx_data.MASK["ANEUPLOID"]
        lambda_g = params[lambda_key]
        X_norm = sx_data.X[aneu][:, is_normal][:, :, None].astype(np.float64)
        mu_norm = (
            sx_data.T[is_normal][None, :, None] * lambda_g[aneu, None, None]
        ).astype(np.float64)
        invphi_init = mle_invphi(
            X_norm,
            mu_norm,
            np.ones_like(X_norm),
        )
        params[invphi_key][:] = invphi_init
        logging.info(
            f"{data_type}: inv_phi init from {np.sum(is_normal)} normal "
            f"cells/spots x {np.sum(aneu)} aneuploid segments "
            f"= {invphi_init:.4f} (phi={1 / invphi_init:.2f})"
        )

    # ------------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------------
    def compute_log_likelihood(self, fit_mode: str, params: dict):
        raise NotImplementedError("not implemented")

    def _e_step(self, fit_mode: str, params: dict, t=0) -> np.ndarray:
        """Compute posterior probabilities. Returns gamma (N, K) or (N_tumor, K_tumor)."""
        ll, log_marg, global_lls = self.compute_log_likelihood(fit_mode, params)
        gamma = np.exp(global_lls - logsumexp(global_lls, axis=1, keepdims=True))
        return gamma

    # ------------------------------------------------------------------
    # M-step helpers
    # ------------------------------------------------------------------
    def _update_pi(self, gamma, params, fix_params, N_eff, K_eff):
        """MAP pi update with Dirichlet prior."""
        if not fix_params["pi"]:
            alpha = self._pi_alpha
            N_k = gamma.sum(axis=0)
            pi = (N_k + alpha - 1) / (N_eff + K_eff * (alpha - 1))
            pi = np.clip(pi, 0, None)
            params["pi"] = pi / pi.sum() if pi.sum() > 0 else np.ones(K_eff) / K_eff

    # ------------------------------------------------------------------
    # Fit (common EM loop)
    # ------------------------------------------------------------------
    def fit(
        self,
        fit_mode="hybrid",
        fix_params=None,
        init_params=None,
        max_iter=100,
        tol=1e-4,
        eps=1e-10,
        **kwargs,
    ):
        if fix_params is None:
            fix_params = {}
        if init_params is None:
            init_params = {}

        assert fit_mode in allowed_fit_mode

        self._pi_alpha = init_params["pi_alpha"]
        self._tau_prior = (init_params["tau_prior_a"], init_params["tau_prior_b"])
        self._invphi_prior = (
            init_params["invphi_prior_a"],
            init_params["invphi_prior_b"],
        )
        self._theta_prior = (
            init_params["theta_prior_a"],
            init_params["theta_prior_b"],
        )

        params, fix_params = self._init_params(fit_mode, fix_params, init_params)

        ll_trace, param_trace = [], []
        prev_ll = -np.inf
        for t in range(1, max_iter):
            param_trace.append(copy.deepcopy(params))
            gamma = self._e_step(fit_mode, params, t)
            self._m_step(fit_mode, gamma, params, fix_params, t=t, eps=eps)

            ll, _, _ = self.compute_log_likelihood(fit_mode, params)
            ll_trace.append(ll)
            if self.verbose:
                logging.info(f"iter={t:03d} log-likelihood = {ll:.6f}")

            if t > 1:
                rel_change = np.abs(ll - prev_ll) / (np.abs(prev_ll) + eps)
                if rel_change < tol:
                    logging.info(
                        f"Converged at iteration {t} (delta = {rel_change:.2e})"
                    )
                    break
            prev_ll = ll

        if self.verbose and self.work_dir:
            plot_loss(
                ll_trace,
                os.path.join(self.work_dir, f"{self.prefix}.log_likelihoods.png"),
                val_type="log-likelihood",
            )

        self.param_trace = param_trace
        self.save_param_trace(param_trace)
        return params

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(
        self,
        fit_mode: str,
        params: dict,
        label: str,
        posterior_thres: float = 0.5,
        margin_thres: float = 0.1,
        tumorprop_threshold: float = 0.5,
    ):
        """Cell-model predict: posteriors over all K clones including normal."""
        logging.info("Decode labels with MAP estimation")
        posteriors = self._e_step(fit_mode, params)
        anns = self.barcodes.copy(deep=True)

        anns.loc[:, self.clones] = posteriors
        anns["tumor"] = 1 - anns["normal"]

        probs = anns[self.clones].to_numpy()
        probs_sorted = np.sort(probs, axis=1)
        anns["max_posterior"] = probs_sorted[:, -1]
        anns["margin_delta"] = probs_sorted[:, -1] - probs_sorted[:, -2]

        anns[label] = anns[self.clones].idxmax(axis=1)

        mask_na = (anns["max_posterior"] < posterior_thres) | (
            anns["margin_delta"] < margin_thres
        )
        anns.loc[mask_na, label] = "NA"

        clone_props = {
            clone: np.mean(anns[label].to_numpy() == clone) for clone in self.clones
        }
        return anns, clone_props

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def save_param_trace(self, param_trace: list):
        if not param_trace or not self.work_dir:
            return
        for key in param_trace[0]:
            vals = [p[key] for p in param_trace]
            if isinstance(vals[0], np.ndarray):
                df = pd.DataFrame(vals)
            else:
                df = pd.DataFrame({"value": vals})
            df.index.name = "iter"
            out_path = os.path.join(self.work_dir, f"{self.prefix}.trace.{key}.tsv")
            df.to_csv(out_path, sep="\t")
