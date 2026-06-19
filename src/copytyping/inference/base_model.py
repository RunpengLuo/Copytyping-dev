import os
import logging

import numpy as np
import pandas as pd

from scipy.special import logsumexp

from copytyping.inference.model_utils import compute_baseline_proportions
from copytyping.sx_data.sx_data import SX_Data


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
        args: dict | None = None,
    ) -> None:
        self.barcodes = barcodes
        self.data_types = data_types
        self.platform = platform
        self.data_sources = data_sources
        self.clones = self.data_sources[self.data_types[0]].clones
        self.num_barcodes = len(barcodes)
        self.num_clones = len(self.clones)
        self.work_dir = work_dir
        self.prefix = prefix
        self.verbose = verbose
        self.allele_mask_id = allele_mask_id
        self.total_mask_id = total_mask_id
        if modality_masks is None:
            modality_masks = {
                dt: np.ones(self.num_barcodes, dtype=bool) for dt in data_types
            }
        self.modality_masks = modality_masks

        # Per-rep grouping (used for per-rep pi / tau / inv_phi)
        rep_series = barcodes["REP_ID"].astype(str)
        self.rep_ids = list(dict.fromkeys(rep_series.tolist()))
        rep_to_idx = {r: i for i, r in enumerate(self.rep_ids)}
        self.rep_idx = np.array([rep_to_idx[r] for r in rep_series], dtype=np.int64)
        self.num_reps = len(self.rep_ids)

        self.args = args
        # reference cells + clone; used by plotting for the RDR baseline
        self.is_reference = None
        self.ref_clone = 0
        self.model_tols = {"tol": 1e-4, "eps": 1e-10}  # EM convergence tolerances
        self.model_params, self.fix_model_params = {}, {}
        if args is not None:
            self.model_params = {
                "pi": np.ones(self.num_clones) / self.num_clones,
                "pi_alpha": args["pi_alpha"],
                "tau_bounds": (args["min_tau"], args["max_tau"]),
                "invphi_bounds": (args["min_invphi"], args["max_invphi"]),
                "ref_label": args["ref_label"],
                "niters": args["niters"],
            }
            self.fix_model_params = {"pi": not args["update_pi"]}
            for data_type in data_types:
                self.fix_model_params[f"{data_type}-theta"] = True  # fixed after init
                self.fix_model_params[f"{data_type}-tau"] = not args["update_tau"]
                self.fix_model_params[f"{data_type}-inv_phi"] = not args[
                    "update_invphi"
                ]

    def _finalize_fix_params(self, params: dict) -> None:
        """Expand fix flags to cover every model-state key (lambda, etc.), keeping
        the user's update_* choices. Mutates self.fix_model_params in place."""
        user_flags = self.fix_model_params
        fix = {key: False for key in params}
        for key in user_flags:
            if key in fix:
                fix[key] = user_flags[key]
        self.fix_model_params = fix

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _estimate_reference_cells(self):
        """Pick the reference-cell set + reference clone via allele-only sub-EM.

        Default (has-normal): cluster cells on clonal imbalanced bins
        (IMBALANCED & ~SUBCLONAL) and use the cells labeled "normal" (clone 0,
        diploid) as the reference. With ``--no_normal`` there is no diploid
        reference, so cluster on all IMBALANCED bins and use the *major clone*
        (most-assigned) as the reference, with a CNP-corrected baseline. The
        sub-model inherits this model's args. Returns
        ``(is_reference, ref_clone, init_labeling)``.
        """
        from copytyping.inference.cell_model import Cell_Model

        no_normal = self.args["no_normal"]
        mask_id = "IMBALANCED" if no_normal else "CLONAL_IMBALANCED"
        logging.info(f"estimate reference cells via allele-only BB model ({mask_id})")
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
            allele_mask_id=mask_id,
            args=self.args,
        )
        allele_params, _ = pure_model.fit("allele_only")
        allele_anns, _ = pure_model.predict(
            "allele_only",
            allele_params,
            label="allele_only_label",
        )
        init_labels = allele_anns["allele_only_label"].to_numpy()

        if no_normal:
            # major clone among tumor clones (normal ignored under --no_normal)
            tumor_counts = [int((init_labels == c).sum()) for c in self.clones[1:]]
            ref_clone = 1 + int(np.argmax(tumor_counts))
            is_reference = init_labels == self.clones[ref_clone]
            n_ref = int(is_reference.sum())
            logging.info(
                f"no_normal: ref_clone={self.clones[ref_clone]} (idx={ref_clone}), "
                f"{n_ref}/{self.num_barcodes} ref cells"
            )
            assert n_ref > 0, "no reference cells found for baseline estimation"
        else:
            ref_clone = 0  # normal clone (1|1 everywhere)
            is_reference = init_labels == "normal"
            n_normal = int(is_reference.sum())
            n_tumor = int(self.num_barcodes - n_normal)
            logging.info(f"#normal={n_normal}, #tumor={n_tumor} / {self.num_barcodes}")
            assert n_normal > 0, "no normal cells/spots found for baseline estimation"

        init_labeling = {
            "labels": init_labels,
            "max_posterior": allele_anns["max_posterior"].to_numpy(),
        }
        self.is_reference = is_reference
        self.ref_clone = ref_clone
        return is_reference, ref_clone, init_labeling

    def _init_lambda(self, params, is_reference, ref_clone):
        """Baseline read-depth proportions from the reference cells per data type.

        Under ``--no_normal`` the reference clone may be non-diploid, so divide
        out its copy ratio (CNP-corrected baseline); otherwise diploid.
        """
        no_normal = self.args["no_normal"]
        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]
            ref_cn = sx_data.C[:, ref_clone] if no_normal else None
            params[f"{data_type}-lambda"] = compute_baseline_proportions(
                sx_data.X,
                sx_data.T,
                is_reference,
                ref_cn=ref_cn,
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

    @staticmethod
    def _harden(gamma: np.ndarray) -> np.ndarray:
        """Hard-EM: collapse soft posteriors to MAP one-hot assignments (N, K_eff)."""
        z = np.zeros_like(gamma)
        z[np.arange(len(gamma)), gamma.argmax(axis=1)] = 1.0
        return z

    # ------------------------------------------------------------------
    # M-step helpers
    # ------------------------------------------------------------------
    def _update_pi(self, gamma, params, N_eff, K_eff):
        """MAP per-rep pi update with Dirichlet prior. pi has shape (R, K_eff)."""
        if self.fix_model_params["pi"]:
            return
        alpha = self._pi_alpha
        pi = params["pi"]  # (R, K_eff)
        new_pi = np.zeros_like(pi)
        for r in range(self.num_reps):
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
    def fit(self, fit_mode="hybrid", **kwargs):
        tol = self.model_tols["tol"]
        eps = self.model_tols["eps"]
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

        self._pi_alpha = self.model_params["pi_alpha"]
        self._tau_bounds = self.model_params["tau_bounds"]
        self._invphi_bounds = self.model_params["invphi_bounds"]
        max_iter = self.model_params["niters"]
        params, init_labeling = self._init_params(fit_mode)

        # pre-allocate LL matrices — updated in-place by compute_log_likelihood
        num_em_clones = getattr(self, "num_em_clones", self.num_clones)
        for dt in self.data_types:
            G = self.data_sources[dt].G
            params[f"{dt}-ll_allele"] = np.zeros(
                (G, self.num_barcodes, num_em_clones), dtype=np.float64
            )
            params[f"{dt}-ll_total"] = np.zeros(
                (G, self.num_barcodes, num_em_clones), dtype=np.float64
            )
        params["ll_global"] = np.zeros(
            (self.num_barcodes, num_em_clones), dtype=np.float64
        )

        ll_trace, gamma_trace = [], []
        self.labeling_trace = [init_labeling]

        prev_ll = -np.inf
        for t in range(1, max_iter):
            gamma = self._e_step(fit_mode, params, t)
            gamma_trace.append(gamma.copy())
            # Hard EM: MAP one-hot assignments drive the M-step
            z = self._harden(gamma)
            self._m_step(fit_mode, z, params, t=t)

            _, _, global_lls = self.compute_log_likelihood(fit_mode, params)
            # hard objective: sum_i max_k global_lls[i,k]
            ll = float(global_lls.max(axis=1).sum())
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
