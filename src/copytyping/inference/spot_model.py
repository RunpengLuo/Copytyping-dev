import logging

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from copytyping.inference.base_model import Base_Model
from copytyping.inference.likelihood_funcs import (
    cond_betabin_logpmf_theta,
    cond_negbin_logpmf_theta,
    mle_invphi,
    mle_tau,
)
from copytyping.inference.model_utils import (
    clone_rdr_gk,
    compute_baseline_proportions,
    estimate_tumor_proportion,
)


class Spot_Model(Base_Model):
    """Spot EM model for spatial data.

    Posteriors over tumor clones only (K_tumor = K-1); normal is modeled
    via per-spot purity theta. Dispersions (tau, phi) are estimated once
    from the neutral (1|1) cluster and held fixed; only theta and pi are
    updated during EM.
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
            init_fix_params,
            init_params,
            ref_label="path_label",
        )
        self._init_is_normal = is_normal

        params = {
            "pi": init_params.get("pi", np.ones(self.K_tumor) / self.K_tumor),
        }

        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]
            lambda_g = compute_baseline_proportions(sx_data.X, sx_data.T, is_normal)
            params[f"{data_type}-lambda"] = lambda_g
            params[f"{data_type}-theta"] = estimate_tumor_proportion(sx_data, lambda_g)

            if fit_mode in {"allele_only", "hybrid"}:
                n_allele = int(sx_data.MASK[self.allele_mask_id].sum())
                params[f"{data_type}-tau"] = np.full(
                    n_allele,
                    init_params["tau0"],
                    dtype=np.float32,
                )
            if fit_mode in {"total_only", "hybrid"}:
                n_total = int(sx_data.MASK[self.total_mask_id].sum())
                params[f"{data_type}-inv_phi"] = np.full(
                    n_total,
                    1 / init_params["phi0"],
                    dtype=np.float32,
                )

        fix_params = {key: False for key in params}
        if init_fix_params is not None:
            for key in init_fix_params:
                if key in fix_params:
                    fix_params[key] = init_fix_params[key]

        self._tumor_idx = np.arange(self.N)
        self._N_tumor = self.N

        return params, fix_params

    def compute_log_likelihood(self, fit_mode, params):
        N_t = self._N_tumor
        global_lls = np.zeros((N_t, self.K_tumor), dtype=np.float32)
        tumor_idx = self._tumor_idx

        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]
            mask_n = self.modality_masks[data_type]
            tumor_mask_n = mask_n[tumor_idx]

            lambda_g = params[f"{data_type}-lambda"]
            rdrs_gk = clone_rdr_gk(lambda_g, sx_data.C)[:, 1:]
            theta_tumor = params[f"{data_type}-theta"][tumor_idx]

            allele_mask = sx_data.MASK[self.allele_mask_id] & (lambda_g > 0)
            total_mask = sx_data.MASK[self.total_mask_id] & (lambda_g > 0)

            if fit_mode in {"allele_only", "hybrid"}:
                MA, _ = sx_data.apply_mask_shallow(
                    self.allele_mask_id, additional_mask=lambda_g > 0
                )
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
                global_lls += contrib

            if fit_mode in {"total_only", "hybrid"}:
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
                global_lls += contrib

        global_lls += np.log(np.clip(params["pi"], 1e-300, None))[None, :]
        log_marg = logsumexp(global_lls, axis=1)
        return np.sum(log_marg), log_marg, global_lls

    def _m_step(self, fit_mode, gamma, params, fix_params, t=0, eps=1e-10):
        N_t = self._N_tumor
        tumor_idx = self._tumor_idx

        self._update_pi(gamma, params, fix_params, N_t, self.K_tumor)

        gamma_gnk = gamma[None, :, :]
        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]
            lambda_g = params[f"{data_type}-lambda"]
            theta = params[f"{data_type}-theta"]
            theta_t = theta[tumor_idx]  # (N_tumor,)
            rdrs_gk = clone_rdr_gk(lambda_g, sx_data.C)[:, 1:]

            # BB tau update
            if (
                fit_mode in {"allele_only", "hybrid"}
                and not fix_params[f"{data_type}-tau"]
                and sx_data.MASK[self.allele_mask_id].sum() > 0
            ):
                allele_mask = sx_data.MASK[self.allele_mask_id] & (lambda_g > 0)
                baf_gk = sx_data.BAF[:, 1:]
                # theta-adjusted BAF: p = (theta*rdr*BAF + 0.5*(1-theta)) / (theta*rdr + (1-theta))
                rdr_bb = rdrs_gk[allele_mask]  # (G_bb, K_tumor)
                baf_bb = baf_gk[allele_mask]  # (G_bb, K_tumor)
                p_gnk = (
                    theta_t[None, :, None] * rdr_bb[:, None, :] * baf_bb[:, None, :]
                    + 0.5 * (1 - theta_t[None, :, None])
                ) / (
                    theta_t[None, :, None] * rdr_bb[:, None, :]
                    + (1 - theta_t[None, :, None])
                )
                Y_gnk = sx_data.Y[allele_mask][:, tumor_idx, None].astype(np.float64)
                D_gnk = sx_data.D[allele_mask][:, tumor_idx, None].astype(np.float64)
                tau_map = mle_tau(
                    Y_gnk,
                    D_gnk,
                    p_gnk,
                    gamma_gnk,
                    prior=self._tau_prior,
                )
                params[f"{data_type}-tau"][:] = tau_map
                tau_mle = mle_tau(Y_gnk, D_gnk, p_gnk, gamma_gnk)
                logging.debug(
                    f"{data_type} tau: MLE={tau_mle:.2f} MAP={tau_map:.2f} "
                    f"ratio={tau_map / tau_mle:.2f}"
                )

            # NB inv_phi update
            if (
                fit_mode in {"total_only", "hybrid"}
                and not fix_params[f"{data_type}-inv_phi"]
                and sx_data.MASK[self.total_mask_id].sum() > 0
            ):
                total_mask = sx_data.MASK[self.total_mask_id] & (lambda_g > 0)
                rdr_nb = rdrs_gk[total_mask]  # (G_nb, K_tumor)
                # theta-adjusted mu: mu = T * lambda * (theta * rdr + (1-theta))
                mu_gnk = (
                    sx_data.T[tumor_idx][None, :, None]
                    * lambda_g[total_mask, None, None]
                    * (
                        theta_t[None, :, None] * rdr_nb[:, None, :]
                        + (1 - theta_t[None, :, None])
                    )
                )
                X_gnk = sx_data.X[total_mask][:, tumor_idx, None].astype(np.float64)
                invphi_map = mle_invphi(
                    X_gnk,
                    mu_gnk,
                    gamma_gnk,
                    prior=self._invphi_prior,
                )
                params[f"{data_type}-inv_phi"][:] = invphi_map
                invphi_mle = mle_invphi(X_gnk, mu_gnk, gamma_gnk)
                logging.debug(
                    f"{data_type} inv_phi: MLE={invphi_mle:.4f} "
                    f"MAP={invphi_map:.4f} "
                    f"(phi: MLE={1 / invphi_mle:.2f} MAP={1 / invphi_map:.2f})"
                )

        purity_bounds = (1e-4, 1.0 - 1e-4)
        a_theta, b_theta = self._theta_prior

        allele_masks = {}
        total_masks = {}
        rdrs = {}
        for data_type in self.data_types:
            sx = self.data_sources[data_type]
            lg = params[f"{data_type}-lambda"]
            rdrs[data_type] = clone_rdr_gk(lg, sx.C)[:, 1:]
            total_masks[data_type] = sx.MASK[self.total_mask_id] & (lg > 0)
            allele_masks[data_type] = sx.MASK[self.allele_mask_id] & (lg > 0)

        theta_arr = params[f"{self.data_types[0]}-theta"].copy()

        for n in range(self.N):

            def neg_Q_theta(theta_val, _n=n):
                theta_val = np.array([theta_val], dtype=float)
                Q = 0.0
                if _n in tumor_idx:
                    tidx = np.searchsorted(tumor_idx, _n)
                    w = gamma[tidx][None, :]
                else:
                    w = np.ones((1, self.K_tumor)) / self.K_tumor

                for data_type in self.data_types:
                    if not self.modality_masks[data_type][_n]:
                        continue
                    sx = self.data_sources[data_type]
                    lg = params[f"{data_type}-lambda"]
                    total_mask = total_masks[data_type]
                    allele_mask = allele_masks[data_type]
                    rdrs_gk = rdrs[data_type]

                    if fit_mode in {"total_only", "hybrid"} and total_mask.any():
                        ll_nb = cond_negbin_logpmf_theta(
                            sx.X[total_mask][:, _n : _n + 1],
                            np.array([sx.T[_n]], dtype=float),
                            lg[total_mask],
                            params[f"{data_type}-inv_phi"][
                                lg[sx.MASK[self.total_mask_id]] > 0
                            ],
                            rdrs_gk[total_mask],
                            theta_val,
                        )
                        Q += np.sum(ll_nb[:, 0, :] * w)

                    if fit_mode in {"allele_only", "hybrid"} and allele_mask.any():
                        ll_bb = cond_betabin_logpmf_theta(
                            sx.Y[allele_mask][:, _n : _n + 1],
                            sx.D[allele_mask][:, _n : _n + 1],
                            params[f"{data_type}-tau"][
                                lg[sx.MASK[self.allele_mask_id]] > 0
                            ],
                            sx.BAF[allele_mask, 1:],
                            rdrs_gk[allele_mask],
                            theta_val,
                        )
                        Q += np.sum(ll_bb[:, 0, :] * w)

                # Beta(a, b) log-prior: (a-1)*log(θ) + (b-1)*log(1-θ)
                tv = float(theta_val[0])
                Q += (a_theta - 1.0) * np.log(tv) + (b_theta - 1.0) * np.log(
                    1.0 - tv
                )
                return -Q

            res = minimize_scalar(
                neg_Q_theta, bounds=purity_bounds, method="bounded"
            )
            theta_arr[n] = np.clip(res.x, 1e-4, 1.0 - 1e-4)

        for data_type in self.data_types:
            if not fix_params[f"{data_type}-theta"]:
                params[f"{data_type}-theta"] = theta_arr.copy()

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
    ):
        logging.info("Decode labels with MAP estimation")
        posteriors = self._e_step(fit_mode, params)
        tumor_clones = self.clones[1:]

        anns = self.barcodes.copy(deep=True)

        theta_list = [
            params[f"{dt}-theta"] for dt in self.data_types if f"{dt}-theta" in params
        ]
        if len(theta_list) == 1:
            anns["tumor_purity"] = theta_list[0]
        else:
            anns["tumor_purity"] = np.mean(theta_list, axis=0)

        for c in tumor_clones:
            anns[c] = 0.0
        anns.iloc[self._tumor_idx, anns.columns.get_indexer(tumor_clones)] = posteriors

        probs = anns[tumor_clones].to_numpy()
        probs_sorted = np.sort(probs, axis=1)
        anns["max_posterior"] = probs_sorted[:, -1]
        anns["margin_delta"] = (
            probs_sorted[:, -1] - probs_sorted[:, -2]
            if probs_sorted.shape[1] > 1
            else 1.0
        )

        anns[label] = anns[tumor_clones].idxmax(axis=1)

        clone_props = {c: np.mean(anns[label].to_numpy() == c) for c in tumor_clones}

        return anns, clone_props
