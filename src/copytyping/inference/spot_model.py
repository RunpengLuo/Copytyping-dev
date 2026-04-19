import logging

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from copytyping.inference.base_model import Base_Model
from copytyping.inference.model_utils import (
    clone_rdr_gk,
    cond_betabin_logpmf,
    cond_betabin_logpmf_theta,
    cond_negbin_logpmf,
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

    Posteriors are over K+1 labels: [normal, clone1, ..., cloneK].
    Pi has two levels: gate (normal vs tumor) and clone (within tumor).
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
        hard_em=False,
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
            hard_em=hard_em,
            allele_mask_id=allele_mask_id,
            total_mask_id=total_mask_id,
        )
        self.tumor_clones = self.clones[1:]
        self.K_tumor = len(self.tumor_clones)

    def _init_params(self, fit_mode, init_fix_params, init_params):
        is_normal = self._identify_normal_cells(
            init_fix_params, init_params, ref_label="path_label"
        )
        self._init_is_normal = is_normal

        # pi over all K+1 (gate + clone combined)
        bulk_pi = init_params.get("pi", np.ones(self.K) / self.K)
        params = {"pi": bulk_pi}

        self._init_lambda(params, is_normal)

        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]
            params[f"{data_type}-theta"] = estimate_tumor_proportion_bin(
                sx_data, params[f"{data_type}-lambda"]
            )
            if fit_mode in {"allele_only", "hybrid"}:
                n_allele = int(sx_data.MASK[self.allele_mask_id].sum())
                params[f"{data_type}-tau"] = np.full(
                    n_allele, init_params["tau0"], dtype=np.float32
                )
            if fit_mode in {"total_only", "hybrid"}:
                n_total = int(sx_data.MASK[self.total_mask_id].sum())
                params[f"{data_type}-inv_phi"] = np.full(
                    n_total, 1 / init_params["phi0"], dtype=np.float32
                )

        fix_params = {key: False for key in params}
        if init_fix_params is not None:
            for key in init_fix_params:
                if key in fix_params:
                    fix_params[key] = init_fix_params[key]

        self._tumor_idx = np.arange(self.N)
        self._N_tumor = self.N
        self._purity_min = init_params.get("purity_min", 0.1)

        return params, fix_params

    def _compute_normal_ll(self, fit_mode, params):
        """Log-likelihood for normal gate (θ=0, CN=diploid) for all spots."""
        normal_ll = np.zeros(self.N, dtype=np.float64)

        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]
            mask_n = self.modality_masks[data_type]
            lambda_g = params[f"{data_type}-lambda"]

            if fit_mode in {"allele_only", "hybrid"}:
                MA, _ = sx_data.apply_mask_shallow(
                    self.allele_mask_id, additional_mask=lambda_g > 0
                )
                tau_valid = params[f"{data_type}-tau"][
                    lambda_g[sx_data.MASK[self.allele_mask_id]] > 0
                ]
                p_normal = np.full((MA["BAF"].shape[0], 1), 0.5)
                ll_bb = cond_betabin_logpmf(MA["Y"], MA["D"], tau_valid, p_normal)
                contrib = ll_bb[:, :, 0].sum(axis=0)
                contrib[~mask_n] = 0.0
                normal_ll += contrib

            if fit_mode in {"total_only", "hybrid"}:
                total_mask = sx_data.MASK[self.total_mask_id] & (lambda_g > 0)
                invphi_valid = params[f"{data_type}-inv_phi"][
                    lambda_g[sx_data.MASK[self.total_mask_id]] > 0
                ]
                pi_normal = lambda_g[total_mask, None]
                ll_nb = cond_negbin_logpmf(
                    sx_data.X[total_mask], sx_data.T, pi_normal, invphi_valid
                )
                contrib = ll_nb[:, :, 0].sum(axis=0)
                contrib[~mask_n] = 0.0
                normal_ll += contrib

        return normal_ll

    def compute_log_likelihood(self, fit_mode, params):
        N_t = self._N_tumor
        tumor_lls = np.zeros((N_t, self.K_tumor), dtype=np.float64)
        tumor_idx = self._tumor_idx

        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]
            mask_n = self.modality_masks[data_type]
            tumor_mask_n = mask_n[tumor_idx]

            lambda_g = params[f"{data_type}-lambda"]
            rdrs_gk = clone_rdr_gk(lambda_g, sx_data.C)[:, 1:]
            theta_tumor = params[f"{data_type}-theta"][tumor_idx]

            if fit_mode in {"allele_only", "hybrid"}:
                MA, _ = sx_data.apply_mask_shallow(
                    self.allele_mask_id, additional_mask=lambda_g > 0
                )
                allele_mask = sx_data.MASK[self.allele_mask_id] & (lambda_g > 0)
                tau_valid = params[f"{data_type}-tau"][
                    lambda_g[sx_data.MASK[self.allele_mask_id]] > 0
                ]
                allele_ll = cond_betabin_logpmf_theta(
                    MA["Y"][:, tumor_idx],
                    MA["D"][:, tumor_idx],
                    tau_valid,
                    MA["BAF"][:, 1:],
                    rdrs_gk[allele_mask],
                    theta_tumor,
                )
                contrib = allele_ll.sum(axis=0)
                contrib[~tumor_mask_n, :] = 0.0
                tumor_lls += contrib

            if fit_mode in {"total_only", "hybrid"}:
                total_mask = sx_data.MASK[self.total_mask_id] & (lambda_g > 0)
                invphi_valid = params[f"{data_type}-inv_phi"][
                    lambda_g[sx_data.MASK[self.total_mask_id]] > 0
                ]
                total_ll = cond_negbin_logpmf_theta(
                    sx_data.X[total_mask][:, tumor_idx],
                    sx_data.T[tumor_idx],
                    lambda_g[total_mask],
                    invphi_valid,
                    rdrs_gk[total_mask],
                    theta_tumor,
                )
                contrib = total_ll.sum(axis=0)
                contrib[~tumor_mask_n, :] = 0.0
                tumor_lls += contrib

        # Normal gate ll
        normal_ll = self._compute_normal_ll(fit_mode, params)

        # Gated log-posteriors: [normal, clone1, ..., cloneK]
        pi = params["pi"]
        pi_normal = pi[0]
        pi_tumor = 1.0 - pi_normal
        pi_clone = pi[1:] / np.maximum(pi[1:].sum(), 1e-30)

        global_lls = np.zeros((self.N, self.K), dtype=np.float64)
        global_lls[:, 0] = normal_ll + np.log(np.maximum(pi_normal, 1e-30))
        global_lls[tumor_idx, 1:] = (
            tumor_lls
            + np.log(np.maximum(pi_tumor, 1e-30))
            + np.log(np.maximum(pi_clone, 1e-30))[None, :]
        )

        log_marg = logsumexp(global_lls, axis=1)
        return np.sum(log_marg), log_marg, global_lls

    def _m_step(self, fit_mode, gamma, params, fix_params, t=0, eps=1e-10):
        tumor_idx = self._tumor_idx

        # --- Mixture weights (doc: M-step mixture weights) ---
        # ρ = (1/N) Σ r_{n0}
        rho = gamma[:, 0].mean()
        # ω_k = Σ r_{nk} / Σ r_{nT}
        r_tumor = gamma[:, 1:]  # (N, K_tumor)
        r_nT = r_tumor.sum(axis=1)  # (N,)
        omega = r_tumor.sum(axis=0) / np.maximum(r_nT.sum(), eps)
        # Store as marginal: π_0=ρ, π_k=(1-ρ)ω_k
        params["pi"] = np.concatenate([[rho], (1 - rho) * omega])

        # Conditional tumor posterior: r̃_{nk} = r_{nk} / r_{nT}
        r_tilde = r_tumor / np.maximum(r_nT[:, None], eps)  # (N, K_tumor)

        # --- Dispersion updates (doc: weighted by ALL responsibilities) ---
        # Build (G, N, K_tumor+1) weights for joint normal+tumor dispersion fit
        # For tau: normal branch uses p=0.5, tumor branches use p̃_{gnk}(v_n)
        # For inv_phi: normal branch uses mu=T*λ, tumor uses T*λ*(θ*rdr+(1-θ))
        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]
            lambda_g = params[f"{data_type}-lambda"]
            theta = params[f"{data_type}-theta"]
            theta_t = theta[tumor_idx]
            rdrs_gk = clone_rdr_gk(lambda_g, sx_data.C)[:, 1:]

            # BB tau update — weighted by all K+1 branches
            if (
                fit_mode in {"allele_only", "hybrid"}
                and not fix_params[f"{data_type}-tau"]
                and sx_data.MASK[self.allele_mask_id].sum() > 0
            ):
                allele_mask = sx_data.MASK[self.allele_mask_id] & (lambda_g > 0)
                baf_gk = sx_data.BAF[:, 1:]
                rdr_bb = rdrs_gk[allele_mask]
                baf_bb = baf_gk[allele_mask]
                # Tumor p̃: (G_bb, N, K_tumor)
                p_tumor = (
                    theta_t[None, :, None] * rdr_bb[:, None, :] * baf_bb[:, None, :]
                    + 0.5 * (1 - theta_t[None, :, None])
                ) / (
                    theta_t[None, :, None] * rdr_bb[:, None, :]
                    + (1 - theta_t[None, :, None])
                )
                # Normal p̃ = 0.5: (G_bb, N, 1)
                p_normal = np.full((p_tumor.shape[0], p_tumor.shape[1], 1), 0.5)
                # Concat: (G_bb, N, K+1) with weights (1, N, K+1)
                p_all = np.concatenate([p_normal, p_tumor], axis=2)
                w_all = gamma[None, :, :]  # (1, N, K+1)
                Y_gnk = sx_data.Y[allele_mask][:, tumor_idx, None].astype(np.float64)
                D_gnk = sx_data.D[allele_mask][:, tumor_idx, None].astype(np.float64)
                # Broadcast Y,D to match K+1 columns
                Y_all = np.broadcast_to(Y_gnk, p_all.shape)
                D_all = np.broadcast_to(D_gnk, p_all.shape)
                tau_map = mle_tau(Y_all, D_all, p_all, w_all, prior=self._tau_prior)
                params[f"{data_type}-tau"][:] = tau_map

            # NB inv_phi update — weighted by all K+1 branches
            if (
                fit_mode in {"total_only", "hybrid"}
                and not fix_params[f"{data_type}-inv_phi"]
                and sx_data.MASK[self.total_mask_id].sum() > 0
            ):
                total_mask = sx_data.MASK[self.total_mask_id] & (lambda_g > 0)
                rdr_nb = rdrs_gk[total_mask]
                # Tumor mu: (G_nb, N, K_tumor)
                mu_tumor = (
                    sx_data.T[tumor_idx][None, :, None]
                    * lambda_g[total_mask, None, None]
                    * (
                        theta_t[None, :, None] * rdr_nb[:, None, :]
                        + (1 - theta_t[None, :, None])
                    )
                )
                # Normal mu = T*λ*2 (m_{g0}=2, but RDR=1 so mu=T*λ)
                mu_normal = (
                    sx_data.T[tumor_idx][None, :, None]
                    * lambda_g[total_mask, None, None]
                )  # (G_nb, N, 1)
                mu_all = np.concatenate([mu_normal, mu_tumor], axis=2)
                w_all = gamma[None, :, :]
                X_gnk = sx_data.X[total_mask][:, tumor_idx, None].astype(np.float64)
                X_all = np.broadcast_to(X_gnk, mu_all.shape)
                invphi_map = mle_invphi(X_all, mu_all, w_all, prior=self._invphi_prior)
                params[f"{data_type}-inv_phi"][:] = invphi_map

        # --- Theta update ---
        if any(not fix_params[f"{dt}-theta"] for dt in self.data_types):
            purity_bounds = (self._purity_min, 1.0 - 1e-4)
            a_theta, b_theta = self._theta_prior
            theta_arr = params[f"{self.data_types[0]}-theta"]

            for n in range(self.N):
                w = r_tilde[n]

                def neg_Q_theta(tv, _n=n, _w=w):
                    tv_arr = np.array([tv], dtype=float)
                    Q = 0.0
                    for data_type in self.data_types:
                        if not self.modality_masks[data_type][_n]:
                            continue
                        sx = self.data_sources[data_type]
                        lg = params[f"{data_type}-lambda"]
                        rdrs_gk = clone_rdr_gk(lg, sx.C)[:, 1:]

                        if fit_mode in {"allele_only", "hybrid"}:
                            am = sx.MASK[self.allele_mask_id] & (lg > 0)
                            if am.any():
                                ll = cond_betabin_logpmf_theta(
                                    sx.Y[am, _n : _n + 1],
                                    sx.D[am, _n : _n + 1],
                                    params[f"{data_type}-tau"][
                                        lg[sx.MASK[self.allele_mask_id]] > 0
                                    ],
                                    sx.BAF[am, 1:],
                                    rdrs_gk[am],
                                    tv_arr,
                                )
                                Q += np.sum(ll[:, 0, :] * _w)

                        if fit_mode in {"total_only", "hybrid"}:
                            tm = sx.MASK[self.total_mask_id] & (lg > 0)
                            if tm.any():
                                ll = cond_negbin_logpmf_theta(
                                    sx.X[tm, _n : _n + 1],
                                    np.array([sx.T[_n]], dtype=float),
                                    lg[tm],
                                    params[f"{data_type}-inv_phi"][
                                        lg[sx.MASK[self.total_mask_id]] > 0
                                    ],
                                    rdrs_gk[tm],
                                    tv_arr,
                                )
                                Q += np.sum(ll[:, 0, :] * _w)

                    Q += (a_theta - 1.0) * np.log(tv) + (b_theta - 1.0) * np.log(
                        1.0 - tv
                    )
                    return -Q

                res = minimize_scalar(
                    neg_Q_theta, bounds=purity_bounds, method="bounded"
                )
                theta_arr[n] = np.clip(res.x, purity_bounds[0], purity_bounds[1])

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(
        self,
        fit_mode,
        params,
        label,
        posterior_thres=0.5,
        margin_thres=0.1,
        **kwargs,
    ):
        logging.info("Decode labels with MAP estimation")
        posteriors = self._e_step(fit_mode, params)

        anns = self.barcodes.copy(deep=True)

        theta_list = [
            params[f"{dt}-theta"] for dt in self.data_types if f"{dt}-theta" in params
        ]
        if len(theta_list) == 1:
            anns["tumor_purity"] = theta_list[0]
        else:
            anns["tumor_purity"] = np.mean(theta_list, axis=0)

        all_clones = list(self.clones)
        anns.loc[:, all_clones] = posteriors
        anns["tumor"] = 1 - anns["normal"]

        probs = anns[all_clones].to_numpy()
        probs_sorted = np.sort(probs, axis=1)
        anns["max_posterior"] = probs_sorted[:, -1]
        anns["margin_delta"] = probs_sorted[:, -1] - probs_sorted[:, -2]

        anns[label] = anns[all_clones].idxmax(axis=1)

        # Hard effective purity: 0 if normal, v_n if tumor
        anns["tumor_purity"] = np.where(
            anns[label] == "normal", 0.0, anns["tumor_purity"]
        )

        clone_props = {c: np.mean(anns[label].to_numpy() == c) for c in all_clones}
        self._log_posterior_stats(anns, label)
        return anns, clone_props
