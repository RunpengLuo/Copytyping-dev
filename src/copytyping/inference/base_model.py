import logging

import numpy as np
import pandas as pd

from scipy.special import logsumexp

from copytyping.inference.count_data import CountData
from copytyping.inference.model_utils import compute_baseline_proportions


class Base_Model:
    def __init__(
        self,
        count_data: dict[str, CountData],
        platform: str,
        assay_types: list[str],
        work_dir: str | None = None,
        prefix: str = "copytyping",
        allele_mask_id: str = "IMBALANCED",
        total_mask_id: str = "ANEUPLOID",
        *,
        no_normal: bool = False,
        pi_alpha: float = 1.0,
        tau_bounds: tuple[float, float] = (1.0, 1e6),
        invphi_bounds: tuple[float, float] = (1.0, 1e6),
        niters: int = 100,
        update_pi: bool = True,
        update_tau: bool = True,
        update_invphi: bool = True,
    ) -> None:
        self.assay_types = assay_types
        self.platform = platform
        # count_data arrives dense from segment_count_data / smooth_spatial_neighbors
        self.count_data = {a: count_data[a] for a in assay_types}
        self.barcodes = self.count_data[assay_types[0]].barcodes

        # per-cell library size per modality
        self.T = {a: self.count_data[a].count_X.sum(axis=0) for a in assay_types}
        self.clones = self.count_data[assay_types[0]].clones
        self.num_barcodes = len(self.barcodes)
        self.num_clones = len(self.clones)
        self.work_dir = work_dir
        self.prefix = prefix
        self.allele_mask_id = allele_mask_id
        self.total_mask_id = total_mask_id

        # model configuration (population-wide; no per-rep grouping)
        self.no_normal = no_normal
        self.pi_alpha = pi_alpha
        self.tau_bounds = tau_bounds
        self.invphi_bounds = invphi_bounds
        self.niters = niters

        # reference cells + clone; used by plotting for the RDR baseline
        self.is_reference = None
        self.ref_clone = 0
        self.model_tols = {"tol": 1e-4, "eps": 1e-10}  # EM convergence tolerances

        # global clone mixture; tau/inv_phi/lambda added at fit init
        self.model_params = {"pi": np.ones(self.num_clones) / self.num_clones}
        self.fix_model_params = {"pi": not update_pi}
        for assay_type in assay_types:
            self.fix_model_params[f"{assay_type}-theta"] = True  # fixed after init
            self.fix_model_params[f"{assay_type}-tau"] = not update_tau
            self.fix_model_params[f"{assay_type}-inv_phi"] = not update_invphi

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _estimate_reference_cells(self) -> tuple[np.ndarray, int, dict]:
        """Pick the reference-cell set + reference clone via allele-only sub-EM.

        Default (has-normal): cluster cells on clonal imbalanced bins
        (IMBALANCED & ~SUBCLONAL) and use the cells labeled "normal" (clone 0,
        diploid) as the reference. With ``--no_normal`` there is no diploid
        reference, so cluster on all IMBALANCED bins and use the *major clone*
        (most-assigned) as the reference, with a CNP-corrected baseline. The
        sub-model inherits this model's config. Returns
        ``(is_reference, ref_clone, init_labeling)``.
        """
        from copytyping.inference.cell_model import Cell_Model

        no_normal = self.no_normal
        mask_id = "IMBALANCED" if no_normal else "CLONAL_IMBALANCED"
        logging.info(f"estimate reference cells via allele-only BB model ({mask_id})")
        for assay in self.assay_types:
            count_data = self.count_data[assay]
            n_clonal = int(count_data.allele_mask["CLONAL_IMBALANCED"].sum())
            n_all = int(count_data.allele_mask["IMBALANCED"].sum())
            logging.info(f"  [{assay}] clonal imbalanced: {n_clonal}/{n_all}")
        pure_model = Cell_Model(
            count_data=self.count_data,
            platform=self.platform,
            assay_types=self.assay_types,
            work_dir=self.work_dir,
            allele_mask_id=mask_id,
            no_normal=self.no_normal,
            pi_alpha=self.pi_alpha,
            tau_bounds=self.tau_bounds,
            invphi_bounds=self.invphi_bounds,
            niters=self.niters,
            update_pi=not self.fix_model_params["pi"],
            update_tau=not self.fix_model_params[f"{self.assay_types[0]}-tau"],
            update_invphi=not self.fix_model_params[f"{self.assay_types[0]}-inv_phi"],
        )
        pure_model.fit("allele")
        allele_anns, _ = pure_model.predict("allele", label="allele_label")
        init_labels = allele_anns["allele_label"].to_numpy()

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

    def _init_lambda(self, is_reference: np.ndarray, ref_clone: int) -> None:
        """Baseline read-depth proportions from the reference cells per data type.

        Under ``--no_normal`` the reference clone may be non-diploid, so divide
        out its copy ratio (CNP-corrected baseline); otherwise diploid.
        """
        no_normal = self.no_normal
        for assay_type in self.assay_types:
            count_data = self.count_data[assay_type]
            ref_cn = count_data.cn_C[:, ref_clone] if no_normal else None
            self.model_params[f"{assay_type}-lambda"] = compute_baseline_proportions(
                count_data.count_X,
                self.T[assay_type],
                is_reference,
                ref_cn=ref_cn,
            )

    # ------------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------------
    def compute_log_likelihood(self, fit_mode: str) -> tuple:
        raise NotImplementedError("not implemented")

    def _e_step(self, fit_mode: str, t: int = 0) -> np.ndarray:
        """Compute posterior probabilities. Returns gamma (N, K) or (N_tumor, K_tumor)."""
        ll, log_marg, global_lls = self.compute_log_likelihood(fit_mode)
        gamma = np.exp(global_lls - logsumexp(global_lls, axis=1, keepdims=True))
        return gamma

    @staticmethod
    def _one_hot(gamma: np.ndarray) -> np.ndarray:
        """Hard-EM: collapse soft posteriors to MAP one-hot assignments (N, K_eff)."""
        z = np.zeros_like(gamma)
        z[np.arange(len(gamma)), gamma.argmax(axis=1)] = 1.0
        return z

    # ------------------------------------------------------------------
    # M-step helpers
    # ------------------------------------------------------------------
    def _update_pi(self, gamma: np.ndarray, N_eff: float, K_eff: int) -> None:
        """Global MAP pi update with Dirichlet(alpha) prior. pi has shape (K_eff,)."""
        if self.fix_model_params["pi"]:
            return
        alpha = self.pi_alpha
        N_k = gamma.sum(axis=0)  # (K_eff,)
        denom = max(N_eff + K_eff * (alpha - 1), 1e-10)
        row = np.clip((N_k + alpha - 1) / denom, 0, None)
        s = row.sum()
        self.model_params["pi"] = row / s if s > 0 else np.ones(K_eff) / K_eff

    # ------------------------------------------------------------------
    # Fit (common EM loop)
    # ------------------------------------------------------------------
    def fit(self, fit_mode: str = "allele_total", **kwargs) -> tuple[dict, float]:
        tol = self.model_tols["tol"]
        eps = self.model_tols["eps"]
        n_allele_bins = sum(
            int(count_data.allele_mask[self.allele_mask_id].sum())
            for count_data in self.count_data.values()
        )
        n_total_bins = sum(
            int(count_data.total_mask[self.total_mask_id].sum())
            for count_data in self.count_data.values()
        )
        if fit_mode == "allele":
            assert n_allele_bins > 0, f"no {self.allele_mask_id} bins for allele"
        elif fit_mode == "total":
            assert n_total_bins > 0, f"no {self.total_mask_id} bins for total"
        else:
            assert n_allele_bins + n_total_bins > 0, (
                f"no {self.allele_mask_id} or {self.total_mask_id} bins for allele_total"
            )

        max_iter = self.niters
        init_labeling = self._init_params(fit_mode)
        params = self.model_params

        # pre-allocate LL matrices — updated in-place by compute_log_likelihood
        num_em_clones = getattr(self, "num_em_clones", self.num_clones)
        for assay in self.assay_types:
            G = self.count_data[assay].num_segment
            params[f"{assay}-ll_allele"] = np.zeros(
                (G, self.num_barcodes, num_em_clones), dtype=np.float64
            )
            params[f"{assay}-ll_total"] = np.zeros(
                (G, self.num_barcodes, num_em_clones), dtype=np.float64
            )
        params["ll_global"] = np.zeros(
            (self.num_barcodes, num_em_clones), dtype=np.float64
        )

        ll_trace, gamma_trace = [], []
        self.labeling_trace = [init_labeling]

        prev_ll = -np.inf
        for t in range(1, max_iter):
            gamma = self._e_step(fit_mode, t)
            gamma_trace.append(gamma.copy())
            # Hard EM: MAP one-hot assignments drive the M-step
            z = self._one_hot(gamma)
            self._m_step(fit_mode, z, t=t)

            _, _, global_lls = self.compute_log_likelihood(fit_mode)
            # hard objective: sum_i max_k global_lls[i,k]
            ll = float(global_lls.max(axis=1).sum())
            ll_trace.append(ll)
            lt = self._map_estimation(gamma, "_", as_df=False)
            theta_key = f"{self.assay_types[0]}-theta"
            if theta_key in params:
                theta = params[theta_key]
                lt["tumor_purity"] = np.where(lt["labels"] == "normal", 0.0, theta)
            self.labeling_trace.append(lt)
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

    def _map_estimation(
        self, gamma: np.ndarray, label: str, as_df: bool = True
    ) -> pd.DataFrame | dict:
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
    def predict(self, fit_mode: str, label: str, **kwargs) -> tuple[pd.DataFrame, dict]:
        """Predict clone labels via hierarchical gated inference.

        1. Normal vs tumor gate
        2. Clone MAP within tumor branch
        """
        gamma = self._e_step(fit_mode)
        anns = self._map_estimation(gamma, label)
        clone_props = {c: np.mean(anns[label].to_numpy() == c) for c in self.clones}
        self._log_posterior_stats(anns, label)
        return anns, clone_props

    def _log_posterior_stats(self, anns: pd.DataFrame, group_label: str) -> None:
        """Log per-group posterior statistics."""
        logging.info("posterior statistics:")
        for grp, sub in anns.groupby(group_label, sort=True):
            mp = sub["max_posterior"].to_numpy()
            logging.info(
                f"  {grp:8s} (n={len(sub):4d}): "
                f"max_post min={mp.min():.3f} mean={mp.mean():.3f} "
                f"median={np.median(mp):.3f} max={mp.max():.3f}"
            )
