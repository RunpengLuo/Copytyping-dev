import os
import logging

import numpy as np
import pandas as pd

from scipy.special import logsumexp

from copytyping.inference.model_utils import mle_invphi, mle_tau
from copytyping.inference.model_utils import compute_baseline_proportions
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

        # Per-rep grouping (used for per-rep pi / tau / inv_phi)
        rep_series = barcodes["REP_ID"].astype(str)
        self.rep_ids = list(dict.fromkeys(rep_series.tolist()))
        rep_to_idx = {r: i for i, r in enumerate(self.rep_ids)}
        self.rep_idx = np.array([rep_to_idx[r] for r in rep_series], dtype=np.int64)
        self.R = len(self.rep_ids)

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
        """Estimate BB tau per rep via MLE on normal cells at imbalanced segments.

        Returns (R,) array. First fits a global tau (all normals pooled) as a
        fallback, then per-rep fits. Reps with no normals inherit the global tau.
        """
        sx_data = self.data_sources[data_type]
        imb = sx_data.MASK["IMBALANCED"]
        # Global tau (pooled across all normals) — used as fallback for empty reps
        Yg = sx_data.Y[imb][:, is_normal][:, :, None].astype(np.float64)
        Dg = sx_data.D[imb][:, is_normal][:, :, None].astype(np.float64)
        tau_global = mle_tau(
            Yg, Dg, np.full_like(Yg, 0.5), np.ones_like(Yg), tau_bounds=tau_bounds
        )
        logging.info(
            f"{data_type}: pooled tau={tau_global:.2f} from "
            f"{int(is_normal.sum())} normals × {int(imb.sum())} imb"
        )
        tau_arr = np.full(self.R, tau_global, dtype=np.float64)
        for r in range(self.R):
            mask_r = (self.rep_idx == r) & is_normal
            n_r = int(mask_r.sum())
            if n_r == 0:
                logging.warning(
                    f"  {data_type} rep={self.rep_ids[r]}: no normals, "
                    f"using pooled tau={tau_global:.2f}"
                )
                continue
            Y = sx_data.Y[imb][:, mask_r][:, :, None].astype(np.float64)
            D = sx_data.D[imb][:, mask_r][:, :, None].astype(np.float64)
            tau_arr[r] = mle_tau(
                Y, D, np.full_like(Y, 0.5), np.ones_like(Y), tau_bounds=tau_bounds
            )
            logging.info(
                f"  {data_type} rep={self.rep_ids[r]}: tau={tau_arr[r]:.2f} from {n_r} normals"
            )
        return tau_arr

    def _init_invphi_from_normals(self, data_type, lambda_g, is_normal, invphi_bounds):
        """Estimate NB inv_phi per rep via MLE on normal cells at aneuploid segments.

        Returns (R,) array. First fits a global inv_phi (all normals pooled) as a
        fallback, then per-rep fits. Reps with no normals inherit the global inv_phi.
        """
        sx_data = self.data_sources[data_type]
        aneu = sx_data.MASK["ANEUPLOID"]
        # Global inv_phi (pooled across all normals) — fallback for empty reps
        Xg = sx_data.X[aneu][:, is_normal][:, :, None].astype(np.float64)
        mug = (sx_data.T[is_normal][None, :, None] * lambda_g[aneu, None, None]).astype(
            np.float64
        )
        invphi_global = mle_invphi(
            Xg, mug, np.ones_like(Xg), invphi_bounds=invphi_bounds
        )
        logging.info(
            f"{data_type}: pooled inv_phi={invphi_global:.2f} from "
            f"{int(is_normal.sum())} normals × {int(aneu.sum())} aneu"
        )
        invphi_arr = np.full(self.R, invphi_global, dtype=np.float64)
        for r in range(self.R):
            mask_r = (self.rep_idx == r) & is_normal
            n_r = int(mask_r.sum())
            if n_r == 0:
                logging.warning(
                    f"  {data_type} rep={self.rep_ids[r]}: no normals, "
                    f"using pooled inv_phi={invphi_global:.2f}"
                )
                continue
            X = sx_data.X[aneu][:, mask_r][:, :, None].astype(np.float64)
            mu = (sx_data.T[mask_r][None, :, None] * lambda_g[aneu, None, None]).astype(
                np.float64
            )
            invphi_arr[r] = mle_invphi(
                X, mu, np.ones_like(X), invphi_bounds=invphi_bounds
            )
            logging.info(
                f"  {data_type} rep={self.rep_ids[r]}: inv_phi={invphi_arr[r]:.2f} from {n_r} normals"
            )
        return invphi_arr

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
        """MAP per-rep pi update with Dirichlet prior. pi has shape (R, K_eff)."""
        if fix_params["pi"]:
            return
        alpha = self._pi_alpha
        pi = params["pi"]  # (R, K_eff)
        new_pi = np.zeros_like(pi)
        for r in range(self.R):
            mask = self.rep_idx == r
            n_r = int(mask.sum())
            if n_r == 0:
                new_pi[r] = pi[r]
                continue
            N_k = gamma[mask].sum(axis=0)  # (K_eff,)
            denom = max(n_r + K_eff * (alpha - 1), 1e-10)
            row = np.clip((N_k + alpha - 1) / denom, 0, None)
            s = row.sum()
            new_pi[r] = row / s if s > 0 else np.ones(K_eff) / K_eff
        params["pi"] = new_pi

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
        self._tau_bounds = init_params["tau_bounds"]
        self._invphi_bounds = init_params["invphi_bounds"]
        params, fix_params, init_labeling = self._init_params(
            fit_mode, fix_params, init_params
        )

        # pre-allocate LL matrices — updated in-place by compute_log_likelihood
        K_em = getattr(self, "_K_em", self.K)
        for dt in self.data_types:
            G = self.data_sources[dt].G
            params[f"{dt}-ll_allele"] = np.zeros((G, self.N, K_em), dtype=np.float64)
            params[f"{dt}-ll_total"] = np.zeros((G, self.N, K_em), dtype=np.float64)
        params["ll_global"] = np.zeros((self.N, K_em), dtype=np.float64)

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

        self.gamma_trace = gamma_trace
        return params, ll_trace[-1] if ll_trace else -np.inf

    def _map_estimation(self, gamma, params, label, as_df=True):
        r"""Flat MAP estimation: z_n = argmax_k γ_nk (k=0 is normal)."""
        N = len(gamma)
        clone_names = np.array(self.clones)  # ["normal", "clone1", ...]
        map_k = gamma.argmax(axis=1)
        labels = clone_names[map_k]
        max_post = gamma[np.arange(N), map_k]

        if not as_df:
            return {"labels": labels, "max_posterior": max_post}

        anns = self.barcodes.copy(deep=True)
        anns.loc[:, self.clones] = gamma
        anns["max_posterior"] = max_post
        anns[label] = labels
        return anns

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, fit_mode, params, label, **kwargs):
        """Predict clone labels via hierarchical gated inference.

        1. Normal vs tumor gate
        2. Clone MAP within tumor branch
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
