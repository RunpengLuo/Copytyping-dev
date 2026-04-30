import logging

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from copytyping.inference.base_model import Base_Model
from copytyping.inference.model_utils import (
    clone_rdr_gk,
    cond_betabin_logpmf_theta,
    cond_negbin_logpmf_theta,
    estimate_tumor_proportion_bin,
)


class Spot_Model(Base_Model):
    """Gated spot EM model for spatial data.

    Each spot has a gate h_n in {normal, tumor}:
    - h_n=normal: θ=0, CN=diploid
    - h_n=tumor: θ_n in [purity_min, 1], clone z_n in {clone1,...,cloneK}

    Posteriors are over K labels: [normal, clone1, ..., cloneK].
    Normal clone uses θ-mixing with CN=(1,1) which naturally produces θ=0 behavior.
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
        allele_mask_id="IMBALANCED",
        total_mask_id="ANEUPLOID",
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
            allele_mask_id=allele_mask_id,
            total_mask_id=total_mask_id,
        )
        self.K_tumor = self.K - 1

    def _init_params(self, fit_mode, init_fix_params, init_params):
        is_normal, init_labeling = self._identify_normal_cells(
            init_fix_params, init_params
        )
        params = {"pi": init_params.get("pi", np.ones(self.K) / self.K)}
        self._init_lambda(params, is_normal)

        for dt in self.data_types:
            sx = self.data_sources[dt]
            lg = params[f"{dt}-lambda"]
            params[f"{dt}-theta"] = estimate_tumor_proportion_bin(
                sx, lg, u_min=init_params["purity_min"]
            )
            if fit_mode in {"allele_only", "hybrid"}:
                params[f"{dt}-tau"] = self._init_tau_from_normals(
                    dt, is_normal, init_params["tau_bounds"]
                )
            if fit_mode in {"total_only", "hybrid"}:
                params[f"{dt}-inv_phi"] = self._init_invphi_from_normals(
                    dt, lg, is_normal, init_params["invphi_bounds"]
                )
            # Precompute fixed arrays
            params[f"{dt}-rdrs"] = clone_rdr_gk(lg, sx.C)
            params[f"{dt}-allele_mask"] = sx.MASK[self.allele_mask_id] & (lg > 0)
            params[f"{dt}-total_mask"] = sx.MASK[self.total_mask_id] & (lg > 0)

        fix_params = {key: False for key in params}
        if init_fix_params is not None:
            for key in init_fix_params:
                if key in fix_params:
                    fix_params[key] = init_fix_params[key]

        return params, fix_params, init_labeling

    def compute_log_likelihood(self, fit_mode, params):
        """Compute log-likelihood, updating per-cluster matrices in params in-place."""
        global_lls = params["ll_global"]
        global_lls[:] = 0.0

        for dt in self.data_types:
            sx = self.data_sources[dt]
            mask_n = self.modality_masks[dt]
            theta = params[f"{dt}-theta"]
            rdrs = params[f"{dt}-rdrs"]
            allele_mask = params[f"{dt}-allele_mask"]
            total_mask = params[f"{dt}-total_mask"]
            ll_a = params[f"{dt}-ll_allele"]
            ll_t = params[f"{dt}-ll_total"]
            ll_a[:] = 0.0
            ll_t[:] = 0.0

            if fit_mode in {"allele_only", "hybrid"} and allele_mask.any():
                MA, _ = sx.apply_mask_shallow(
                    self.allele_mask_id, additional_mask=params[f"{dt}-lambda"] > 0
                )
                ll_a[allele_mask] = cond_betabin_logpmf_theta(
                    MA["Y"],
                    MA["D"],
                    params[f"{dt}-tau"],
                    MA["BAF"],
                    rdrs[allele_mask],
                    theta,
                )
                ll_a[:, ~mask_n, :] = 0.0
                global_lls += ll_a.sum(axis=0)

            if fit_mode in {"total_only", "hybrid"} and total_mask.any():
                ll_t[total_mask] = cond_negbin_logpmf_theta(
                    sx.X[total_mask],
                    sx.T,
                    params[f"{dt}-lambda"][total_mask],
                    params[f"{dt}-inv_phi"],
                    rdrs[total_mask],
                    theta,
                )
                ll_t[:, ~mask_n, :] = 0.0
                global_lls += ll_t.sum(axis=0)

        # Gated pi: π_0=ρ, π_k=(1-ρ)·ω_k
        pi = params["pi"]
        rho = pi[0]
        pi_clone = pi[1:] / np.maximum(pi[1:].sum(), 1e-30)
        global_lls[:, 0] += np.log(np.maximum(rho, 1e-30))
        global_lls[:, 1:] += np.log(np.maximum(1 - rho, 1e-30)) + np.log(
            np.maximum(pi_clone, 1e-30)
        )

        log_marg = logsumexp(global_lls, axis=1)
        return np.sum(log_marg), log_marg, global_lls

    def _m_step(self, fit_mode, gamma, params, fix_params, t=0, eps=1e-10):
        # Mixture weights: ρ and ω_k
        rho = gamma[:, 0].mean()
        r_tumor = gamma[:, 1:]
        r_nT = r_tumor.sum(axis=1)
        omega = r_tumor.sum(axis=0) / np.maximum(r_nT.sum(), eps)
        params["pi"] = np.concatenate([[rho], (1 - rho) * omega])

        # Conditional tumor posterior for theta update
        r_tilde = r_tumor / np.maximum(r_nT[:, None], eps)

        # Theta update (per-spot, tumor clones only)
        if any(not fix_params[f"{dt}-theta"] for dt in self.data_types):
            purity_bounds = (self._purity_min, 1.0 - 1e-4)
            theta_arr = params[f"{self.data_types[0]}-theta"]

            for n in range(self.N):
                w = r_tilde[n]

                def neg_Q(tv, _n=n, _w=w):
                    tv_arr = np.array([tv], dtype=float)
                    Q = 0.0
                    for dt in self.data_types:
                        if not self.modality_masks[dt][_n]:
                            continue
                        sx = self.data_sources[dt]
                        rdrs_t = params[f"{dt}-rdrs"][:, 1:]
                        am = params[f"{dt}-allele_mask"]
                        if fit_mode in {"allele_only", "hybrid"} and am.any():
                            ll = cond_betabin_logpmf_theta(
                                sx.Y[am, _n : _n + 1],
                                sx.D[am, _n : _n + 1],
                                params[f"{dt}-tau"],
                                sx.BAF[am, 1:],
                                rdrs_t[am],
                                tv_arr,
                            )
                            Q += np.sum(ll[:, 0, :] * _w)
                        tm = params[f"{dt}-total_mask"]
                        if fit_mode in {"total_only", "hybrid"} and tm.any():
                            ll = cond_negbin_logpmf_theta(
                                sx.X[tm, _n : _n + 1],
                                np.array([sx.T[_n]], dtype=float),
                                params[f"{dt}-lambda"][tm],
                                params[f"{dt}-inv_phi"],
                                rdrs_t[tm],
                                tv_arr,
                            )
                            Q += np.sum(ll[:, 0, :] * _w)
                    return -Q

                theta_arr[n] = minimize_scalar(
                    neg_Q, bounds=purity_bounds, method="bounded"
                ).x

    def predict(self, fit_mode, params, label, **kwargs):
        """Predict clone labels with purity filter at multiple cutoffs.

        1. Normal vs tumor gate
        2. Purity filter: tumor spots with θ <= cutoff to normal (per cutoff)
        3. Clone MAP within tumor branch
        4. CQ score
        """
        gamma = self._e_step(fit_mode, params)
        theta = params[f"{self.data_types[0]}-theta"]

        anns = self._map_estimation(gamma, params, label)
        anns["tumor_purity"] = theta
        base_labels = anns[label].values.copy()
        base_cq = anns["CQ"].values.copy()

        pcut_labels = []
        for pcut in self._purity_cutoffs:
            col = f"{label}_pcut{pcut}"
            labels_pcut = base_labels.copy()
            low_pur = (labels_pcut != "normal") & (theta <= pcut)
            n_low = int(low_pur.sum())
            if n_low > 0:
                logging.info(
                    f"purity filter (pcut={pcut}): {n_low} spots relabeled to normal"
                )
                labels_pcut[low_pur] = "normal"
            anns[col] = labels_pcut
            pcut_labels.append(col)

        # Use first cutoff as the main label
        anns[label] = anns[pcut_labels[0]].values
        anns["CQ"] = np.where(anns[label] == "normal", 0, base_cq)

        clone_props = {c: np.mean(anns[label].to_numpy() == c) for c in self.clones}
        self._log_posterior_stats(anns, label)
        return anns, clone_props
