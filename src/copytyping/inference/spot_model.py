import logging

import numpy as np
from scipy.special import logsumexp

from copytyping.inference.base_model import Base_Model
from copytyping.inference.model_utils import (
    clone_rdr_gk,
    cond_betabin_logpmf_theta,
    cond_negbin_logpmf_theta,
    estimate_tumor_proportion,
)


class Spot_Model(Base_Model):
    """Purity-only spot model for spatial data (no normal clone in EM).

    Latent variables per spot:
    - z_n in {clone1, ..., cloneK} (tumor clones only)
    - θ_n in [0, 1] (tumor purity, 0 = pure normal)

    Normal/tumor assignment is decided post-EM by thresholding θ.
    This avoids the identifiability issue of having both a normal clone
    and θ→0 explain the same observation.
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
        self.K_tumor = self.K - 1  # number of tumor clones (exclude normal)
        self._K_em = self.K_tumor  # EM operates on tumor clones only
        self.tumor_clones = self.clones[1:]  # ["clone1", "clone2", ...]

    def _init_params(self, fit_mode, init_fix_params, init_params):
        is_normal, init_labeling = self._identify_normal_cells(
            init_fix_params, init_params
        )
        # pi per rep over tumor clones only: shape (R, K_tumor)
        bulk_pi = init_params.get("pi", np.ones(self.K) / self.K)
        tumor_pi = bulk_pi[1:]
        tumor_pi = tumor_pi / tumor_pi.sum()
        pi_per_rep = np.tile(tumor_pi, (self.R, 1))  # (R, K_tumor)
        params = {"pi": pi_per_rep}
        self._init_lambda(params, is_normal)

        for dt in self.data_types:
            sx = self.data_sources[dt]
            lg = params[f"{dt}-lambda"]
            tau_per_rep = inv_phi_per_rep = None
            if fit_mode in {"allele_only", "hybrid"}:
                tau_per_rep = self._init_tau_from_normals(
                    dt, is_normal, init_params["tau_bounds"]
                )
                params[f"{dt}-tau"] = tau_per_rep  # (R,)
            if fit_mode in {"total_only", "hybrid"}:
                inv_phi_per_rep = self._init_invphi_from_normals(
                    dt, lg, is_normal, init_params["invphi_bounds"]
                )
                params[f"{dt}-inv_phi"] = inv_phi_per_rep  # (R,)
            # estimate_tumor_proportion uses scalar dispersion;
            # use the pooled (mean) value as a representative.
            tau_init = float(np.mean(tau_per_rep)) if tau_per_rep is not None else None
            invphi_init = (
                float(np.mean(inv_phi_per_rep)) if inv_phi_per_rep is not None else None
            )
            params[f"{dt}-theta"] = estimate_tumor_proportion(
                sx, lg, tau_init, invphi_init, fit_mode=fit_mode
            )
            # Precompute: rdrs only for tumor clones (exclude normal column)
            rdrs_full = clone_rdr_gk(lg, sx.C)
            params[f"{dt}-rdrs"] = rdrs_full[:, 1:]  # (G, K_tumor)
            params[f"{dt}-allele_mask"] = sx.MASK[self.allele_mask_id] & (lg > 0)
            params[f"{dt}-total_mask"] = sx.MASK[self.total_mask_id] & (lg > 0)

        fix_params = {key: False for key in params}
        if init_fix_params is not None:
            for key in init_fix_params:
                if key in fix_params:
                    fix_params[key] = init_fix_params[key]

        return params, fix_params, init_labeling

    def compute_log_likelihood(self, fit_mode, params):
        """Compute log-likelihood over tumor clones only (K_tumor components)."""
        global_lls = params["ll_global"]
        global_lls[:] = 0.0

        for dt in self.data_types:
            sx = self.data_sources[dt]
            mask_n = self.modality_masks[dt]
            theta = params[f"{dt}-theta"]
            rdrs = params[f"{dt}-rdrs"]  # (G, K_tumor)
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
                tau_per_spot = params[f"{dt}-tau"][self.rep_idx]  # (N,)
                ll_a[allele_mask] = cond_betabin_logpmf_theta(
                    MA["Y"],
                    MA["D"],
                    tau_per_spot,
                    MA["BAF"][:, 1:],  # tumor clone BAFs only
                    rdrs[allele_mask],
                    theta,
                )
                ll_a[:, ~mask_n, :] = 0.0
                global_lls += ll_a.sum(axis=0)

            if fit_mode in {"total_only", "hybrid"} and total_mask.any():
                invphi_per_spot = params[f"{dt}-inv_phi"][self.rep_idx]  # (N,)
                ll_t[total_mask] = cond_negbin_logpmf_theta(
                    sx.X[total_mask],
                    sx.T,
                    params[f"{dt}-lambda"][total_mask],
                    invphi_per_spot,
                    rdrs[total_mask],
                    theta,
                )
                ll_t[:, ~mask_n, :] = 0.0
                global_lls += ll_t.sum(axis=0)

        # Per-rep pi prior: log pi[rep_idx[n], k] added per spot
        pi = params["pi"]  # (R, K_tumor)
        global_lls += np.log(np.maximum(pi[self.rep_idx], 1e-30))

        log_marg = logsumexp(global_lls, axis=1)
        return np.sum(log_marg), log_marg, global_lls

    def _m_step(self, fit_mode, gamma, params, fix_params, t=0, eps=1e-10):
        # Mixture weights: standard simplex update.
        # theta is fixed after init (see estimate_tumor_proportion).
        self._update_pi(gamma, params, fix_params, self.N, self.K_tumor)

    def _e_step(self, fit_mode, params, t=0):
        """E-step: posterior over K_tumor clones."""
        ll, log_marg, global_lls = self.compute_log_likelihood(fit_mode, params)
        gamma = np.exp(global_lls - logsumexp(global_lls, axis=1, keepdims=True))
        return gamma

    def _map_estimation(self, gamma, params, label, as_df=True):
        """MAP over tumor clones only. Normal assigned by purity threshold."""
        N = len(gamma)
        clone_names = np.array(self.tumor_clones)
        map_k = gamma.argmax(axis=1)
        labels = clone_names[map_k]
        max_post = gamma[np.arange(N), map_k]

        if not as_df:
            return {"labels": labels, "max_posterior": max_post}

        anns = self.barcodes.copy(deep=True)
        anns.loc[:, self.tumor_clones] = gamma
        anns["max_posterior"] = max_post
        anns[label] = labels
        return anns

    def predict(self, fit_mode, params, label, **kwargs):
        """Predict clone labels via MAP. Purity reported but does not affect labels.

        Clone MAP: z_n = argmax_k gamma_nk (over tumor clones).
        """
        gamma = self._e_step(fit_mode, params)
        theta = params[f"{self.data_types[0]}-theta"]

        clone_names = np.array(self.tumor_clones)
        map_k = gamma.argmax(axis=1)
        labels = clone_names[map_k]
        max_post = gamma[np.arange(self.N), map_k]

        anns = self.barcodes.copy(deep=True)
        anns.loc[:, self.tumor_clones] = gamma
        anns["max_posterior"] = max_post
        anns["tumor_purity"] = theta
        anns[label] = labels

        clone_props = {c: np.mean(anns[label].to_numpy() == c) for c in self.clones}
        self._log_posterior_stats(anns, label)
        return anns, clone_props
