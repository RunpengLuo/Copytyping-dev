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
    mle_invphi,
    mle_tau,
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
        is_normal = self._identify_normal_cells(init_fix_params, init_params)
        params = {"pi": init_params.get("pi", np.ones(self.K) / self.K)}
        self._init_lambda(params, is_normal)

        for dt in self.data_types:
            sx = self.data_sources[dt]
            lg = params[f"{dt}-lambda"]
            params[f"{dt}-theta"] = estimate_tumor_proportion_bin(sx, lg)
            if fit_mode in {"allele_only", "hybrid"}:
                n = int(sx.MASK[self.allele_mask_id].sum())
                params[f"{dt}-tau"] = np.full(n, init_params["tau0"], dtype=np.float32)
            if fit_mode in {"total_only", "hybrid"}:
                n = int(sx.MASK[self.total_mask_id].sum())
                params[f"{dt}-inv_phi"] = np.full(
                    n, 1 / init_params["phi0"], dtype=np.float32
                )
            # Precompute fixed arrays
            params[f"{dt}-rdrs"] = clone_rdr_gk(lg, sx.C)
            params[f"{dt}-allele_mask"] = sx.MASK[self.allele_mask_id] & (lg > 0)
            params[f"{dt}-total_mask"] = sx.MASK[self.total_mask_id] & (lg > 0)
            params[f"{dt}-tau_valid_idx"] = lg[sx.MASK[self.allele_mask_id]] > 0
            params[f"{dt}-invphi_valid_idx"] = lg[sx.MASK[self.total_mask_id]] > 0

        fix_params = {key: False for key in params}
        if init_fix_params is not None:
            for key in init_fix_params:
                if key in fix_params:
                    fix_params[key] = init_fix_params[key]

        self._purity_min = init_params.get("purity_min", 0.1)
        return params, fix_params

    def compute_log_likelihood(self, fit_mode, params):
        """Compute log-likelihood for all K clones (normal + tumor).

        Normal clone (k=0) has CN=(1,1), so theta-mixing naturally gives
        BAF=0.5 and RDR=1 regardless of theta. No separate normal branch needed.
        """
        N = self.N
        global_lls = np.zeros((N, self.K), dtype=np.float64)

        for dt in self.data_types:
            sx = self.data_sources[dt]
            mask_n = self.modality_masks[dt]
            theta = params[f"{dt}-theta"]
            rdrs = params[f"{dt}-rdrs"]
            allele_mask = params[f"{dt}-allele_mask"]
            total_mask = params[f"{dt}-total_mask"]

            if fit_mode in {"allele_only", "hybrid"} and allele_mask.any():
                MA, _ = sx.apply_mask_shallow(
                    self.allele_mask_id, additional_mask=params[f"{dt}-lambda"] > 0
                )
                tau_valid = params[f"{dt}-tau"][params[f"{dt}-tau_valid_idx"]]
                ll = cond_betabin_logpmf_theta(
                    MA["Y"],
                    MA["D"],
                    tau_valid,
                    MA["BAF"],
                    rdrs[allele_mask],
                    theta,
                )
                contrib = ll.sum(axis=0)  # (N, K)
                contrib[~mask_n, :] = 0.0
                global_lls += contrib

            if fit_mode in {"total_only", "hybrid"} and total_mask.any():
                invphi_valid = params[f"{dt}-inv_phi"][params[f"{dt}-invphi_valid_idx"]]
                ll = cond_negbin_logpmf_theta(
                    sx.X[total_mask],
                    sx.T,
                    params[f"{dt}-lambda"][total_mask],
                    invphi_valid,
                    rdrs[total_mask],
                    theta,
                )
                contrib = ll.sum(axis=0)
                contrib[~mask_n, :] = 0.0
                global_lls += contrib

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

        for dt in self.data_types:
            sx = self.data_sources[dt]
            lg = params[f"{dt}-lambda"]
            theta_t = params[f"{dt}-theta"]
            rdrs = params[f"{dt}-rdrs"]
            allele_mask = params[f"{dt}-allele_mask"]
            total_mask = params[f"{dt}-total_mask"]

            if (
                fit_mode in {"allele_only", "hybrid"}
                and not fix_params[f"{dt}-tau"]
                and allele_mask.any()
            ):
                rdr = rdrs[allele_mask]
                baf = sx.BAF[allele_mask]
                p_gnk = (
                    theta_t[None, :, None] * rdr[:, None, :] * baf[:, None, :]
                    + 0.5 * (1 - theta_t[None, :, None])
                ) / (
                    theta_t[None, :, None] * rdr[:, None, :]
                    + (1 - theta_t[None, :, None])
                )
                Y = sx.Y[allele_mask][:, :, None].astype(np.float64)
                D = sx.D[allele_mask][:, :, None].astype(np.float64)
                params[f"{dt}-tau"][:] = mle_tau(
                    np.broadcast_to(Y, p_gnk.shape),
                    np.broadcast_to(D, p_gnk.shape),
                    p_gnk,
                    gamma[None, :, :],
                    tau_bounds=self._tau_bounds,
                )

            if (
                fit_mode in {"total_only", "hybrid"}
                and not fix_params[f"{dt}-inv_phi"]
                and total_mask.any()
            ):
                rdr = rdrs[total_mask]
                mu_gnk = (
                    sx.T[None, :, None]
                    * lg[total_mask, None, None]
                    * (
                        theta_t[None, :, None] * rdr[:, None, :]
                        + (1 - theta_t[None, :, None])
                    )
                )
                X = sx.X[total_mask][:, :, None].astype(np.float64)
                params[f"{dt}-inv_phi"][:] = mle_invphi(
                    np.broadcast_to(X, mu_gnk.shape),
                    mu_gnk,
                    gamma[None, :, :],
                    invphi_bounds=self._invphi_bounds,
                )

        # Theta update (per-spot, tumor clones only)
        if any(not fix_params[f"{dt}-theta"] for dt in self.data_types):
            a_theta, b_theta = self._theta_prior
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
                                params[f"{dt}-tau"][params[f"{dt}-tau_valid_idx"]],
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
                                params[f"{dt}-inv_phi"][
                                    params[f"{dt}-invphi_valid_idx"]
                                ],
                                rdrs_t[tm],
                                tv_arr,
                            )
                            Q += np.sum(ll[:, 0, :] * _w)
                    Q += (a_theta - 1) * np.log(tv) + (b_theta - 1) * np.log(1 - tv)
                    return -Q

                theta_arr[n] = minimize_scalar(
                    neg_Q, bounds=purity_bounds, method="bounded"
                ).x

    def predict(
        self, fit_mode, params, label, posterior_thres=0.5, margin_thres=0.1, **kwargs
    ):
        logging.info("Decode labels with MAP estimation")
        posteriors = self._e_step(fit_mode, params)
        anns = self.barcodes.copy(deep=True)

        all_clones = list(self.clones)
        anns.loc[:, all_clones] = posteriors
        anns["tumor"] = 1 - anns["normal"]

        probs = anns[all_clones].to_numpy()
        probs_sorted = np.sort(probs, axis=1)
        anns["max_posterior"] = probs_sorted[:, -1]
        anns["margin_delta"] = probs_sorted[:, -1] - probs_sorted[:, -2]
        anns[label] = anns[all_clones].idxmax(axis=1)

        theta = params[f"{self.data_types[0]}-theta"]
        anns["tumor_purity"] = np.where(anns[label] == "normal", 0.0, theta)

        clone_props = {c: np.mean(anns[label].to_numpy() == c) for c in all_clones}
        self._log_posterior_stats(anns, label)
        return anns, clone_props
