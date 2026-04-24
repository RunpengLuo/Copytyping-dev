import os
import logging

import numpy as np
import pandas as pd

from scipy.special import logsumexp

from copytyping.inference.model_utils import mle_invphi, mle_tau
from copytyping.inference.model_utils import compute_baseline_proportions
from copytyping.plot.plot_common import plot_loss
from copytyping.sx_data.sx_data import SX_Data

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
        allele_mask_id: str = "IMBALANCED",
        total_mask_id: str = "ANEUPLOID",
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
        self.allele_mask_id = allele_mask_id
        self.total_mask_id = total_mask_id
        if modality_masks is None:
            modality_masks = {dt: np.ones(self.N, dtype=bool) for dt in data_types}
        self.modality_masks = modality_masks

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _identify_normal_cells(self, init_fix_params, init_params):
        """Identify normal cells/spots via allele-only sub-EM.

        Uses only clonal imbalanced bins (IMBALANCED & ~SUBCLONAL)
        so that the BB model discriminates normal vs tumor without
        inter-clone confusion.
        """
        from copytyping.inference.cell_model import Cell_Model

        logging.info("infer normal cells using allele-only BB model")
        for dt in self.data_types:
            sx = self.data_sources[dt]
            n_clonal = int(sx.MASK["CLONAL_IMBALANCED"].sum())
            n_all = int(sx.MASK["IMBALANCED"].sum())
            logging.info(f"  [{dt}] clonal imbalanced: {n_clonal}/{n_all}")
        pure_model = Cell_Model(
            self.barcodes,
            self.platform,
            self.data_types,
            self.data_sources,
            work_dir=self.work_dir,
            modality_masks=self.modality_masks,
            allele_mask_id="CLONAL_IMBALANCED",
        )
        cell_init = {k: v for k, v in init_params.items() if k != "pi"}
        allele_params, _ = pure_model.fit(
            "allele_only",
            fix_params=init_fix_params,
            init_params=cell_init,
            max_iter=init_params["niters"],
        )
        allele_anns, _ = pure_model.predict(
            "allele_only",
            allele_params,
            label="allele_only-label",
        )
        init_labels = allele_anns["allele_only-label"].to_numpy()
        is_normal = init_labels == "normal"

        n_normal = int(is_normal.sum())
        n_tumor = int(self.num_barcodes - n_normal)
        logging.info(f"#normal={n_normal}, #tumor={n_tumor} / {self.num_barcodes}")
        assert n_normal > 0, "no normal cells/spots found for baseline estimation"
        init_labeling = {
            "labels": init_labels,
            "max_posterior": allele_anns["max_posterior"].to_numpy(),
        }
        return is_normal, init_labeling

    def _init_lambda(self, params, is_normal):
        """Compute baseline proportions from normal cells for all data types."""
        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]
            params[f"{data_type}-lambda"] = compute_baseline_proportions(
                sx_data.X,
                sx_data.T,
                is_normal,
            )

    def _init_tau_from_normals(self, data_type, is_normal, tau_bounds):
        """Estimate BB tau (scalar) via MLE on normal cells at imbalanced segments."""
        sx_data = self.data_sources[data_type]
        imb = sx_data.MASK["IMBALANCED"]
        Y_norm = sx_data.Y[imb][:, is_normal][:, :, None].astype(np.float64)
        D_norm = sx_data.D[imb][:, is_normal][:, :, None].astype(np.float64)
        tau = mle_tau(
            Y_norm,
            D_norm,
            np.full_like(Y_norm, 0.5),
            np.ones_like(Y_norm),
            tau_bounds=tau_bounds,
        )
        logging.info(
            f"{data_type}: tau={tau:.2f} (bounds={tau_bounds}) from "
            f"{int(np.sum(is_normal))} normal x {int(imb.sum())} imbalanced segments"
        )
        return tau

    def _init_invphi_from_normals(self, data_type, lambda_g, is_normal, invphi_bounds):
        """Estimate NB inv_phi (scalar) via MLE on normal cells at aneuploid segments."""
        sx_data = self.data_sources[data_type]
        aneu = sx_data.MASK["ANEUPLOID"]
        X_norm = sx_data.X[aneu][:, is_normal][:, :, None].astype(np.float64)
        mu_norm = (
            sx_data.T[is_normal][None, :, None] * lambda_g[aneu, None, None]
        ).astype(np.float64)
        invphi = mle_invphi(
            X_norm, mu_norm, np.ones_like(X_norm), invphi_bounds=invphi_bounds
        )
        logging.info(
            f"{data_type}: inv_phi={invphi:.4f} (phi={1 / invphi:.2f}, bounds={invphi_bounds}) from "
            f"{int(np.sum(is_normal))} normal x {int(aneu.sum())} aneuploid segments"
        )
        return invphi

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

        n_allele_bins = sum(
            int(sx.MASK[self.allele_mask_id].sum()) for sx in self.data_sources.values()
        )
        n_total_bins = sum(
            int(sx.MASK[self.total_mask_id].sum()) for sx in self.data_sources.values()
        )
        if fit_mode == "allele_only":
            assert n_allele_bins > 0, f"no {self.allele_mask_id} bins for allele_only"
        elif fit_mode == "total_only":
            assert n_total_bins > 0, f"no {self.total_mask_id} bins for total_only"
        else:
            assert n_allele_bins + n_total_bins > 0, (
                f"no {self.allele_mask_id} or {self.total_mask_id} bins for hybrid"
            )

        self._pi_alpha = init_params["pi_alpha"]
        self._purity_min = init_params["purity_min"]
        self._purity_cutoff = init_params["purity_cutoff"]
        self._cq_cutoff = init_params["cq_cutoff"]
        params, fix_params, init_labeling = self._init_params(
            fit_mode, fix_params, init_params
        )

        # pre-allocate LL matrices — updated in-place by compute_log_likelihood
        for dt in self.data_types:
            G = self.data_sources[dt].G
            params[f"{dt}-ll_allele"] = np.zeros((G, self.N, self.K), dtype=np.float64)
            params[f"{dt}-ll_total"] = np.zeros((G, self.N, self.K), dtype=np.float64)
        params["ll_global"] = np.zeros((self.N, self.K), dtype=np.float64)

        ll_trace, gamma_trace = [], []
        self.labeling_trace = [init_labeling]

        prev_ll = -np.inf
        for t in range(1, max_iter):
            gamma = self._e_step(fit_mode, params, t)
            gamma_trace.append(gamma.copy())
            self._m_step(fit_mode, gamma, params, fix_params, t=t, eps=eps)

            ll, _, _ = self.compute_log_likelihood(fit_mode, params)
            ll_trace.append(ll)
            lt = self._map_estimation(gamma, params, "_", as_df=False)
            theta_key = f"{self.data_types[0]}-theta"
            if theta_key in params:
                theta = params[theta_key]
                lt["tumor_purity"] = np.where(lt["labels"] == "normal", 0.0, theta)
            self.labeling_trace.append(lt)
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

        self.gamma_trace = gamma_trace
        return params, ll_trace[-1] if ll_trace else -np.inf

    def _map_estimation(self, gamma, params, label, as_df=True, eps=1e-30):
        r"""Hierarchical MAP estimation.

        1. Normal vs tumor gate: h_n = 0 if r_{n0} >= r_{nT}
        2. Clone MAP: z_n = argmax_k r_tilde_{nk} for tumor spots
        3. CQ score: CQ_n = -10 log10(1 - max_k r_tilde_{nk} + eps)
        4. CQ filter: CQ < cq_cutoff to unassigned_tumor
        """
        N = len(gamma)
        r_n0 = gamma[:, 0]
        r_nT = 1 - r_n0
        clone_names = np.array(self.clones[1:])  # exclude "normal"

        labels = np.where(r_n0 >= r_nT, "normal", "")

        r_tilde = gamma[:, 1:] / np.maximum(r_nT[:, None], eps)
        map_k = r_tilde.argmax(axis=1)
        tumor = labels != "normal"
        labels[tumor] = clone_names[map_k[tumor]]

        # Step 4: CQ score
        max_r_tilde = r_tilde[np.arange(N), map_k]
        cq = -10 * np.log10(1 - max_r_tilde + eps)

        if self._cq_cutoff > 0:
            low_cq = tumor & (cq < self._cq_cutoff)
            labels[low_cq] = "unassigned_tumor"

        if not as_df:
            result = {
                "labels": labels,
                "max_posterior": gamma[np.arange(N), gamma.argmax(axis=1)],
            }
            return result

        anns = self.barcodes.copy(deep=True)
        anns.loc[:, self.clones] = gamma
        anns["r_normal"] = r_n0
        anns["r_tumor"] = r_nT
        anns["CQ"] = np.where(labels != "normal", cq, np.nan)
        anns[label] = labels
        return anns

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, fit_mode, params, label, **kwargs):
        """Predict clone labels via hierarchical gated inference.

        1. Normal vs tumor gate
        2. Clone MAP within tumor branch
        3. CQ (copytyping quality) score
        4. CQ threshold to unassigned_tumor
        """
        gamma = self._e_step(fit_mode, params)
        anns = self._map_estimation(gamma, params, label)
        clone_props = {c: np.mean(anns[label].to_numpy() == c) for c in self.clones}
        self._log_posterior_stats(anns, label)
        return anns, clone_props

    def _log_posterior_stats(self, anns, group_label):
        """Log per-group posterior statistics."""
        logging.info("posterior statistics:")
        for grp, sub in anns.groupby(group_label, sort=True):
            mp = sub["max_posterior"].to_numpy()
            logging.info(
                f"  {grp:8s} (n={len(sub):4d}): "
                f"max_post min={mp.min():.3f} mean={mp.mean():.3f} "
                f"median={np.median(mp):.3f} max={mp.max():.3f}"
            )
