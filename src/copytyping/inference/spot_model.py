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
        )
        self.tumor_clones = self.clones[1:]
        self.K_tumor = len(self.tumor_clones)

    def _init_params(self, fit_mode, init_fix_params, init_params):
        is_normal = self._identify_normal_cells(
            init_fix_params,
            init_params,
            ref_label="path_label",
        )

        logtau_bounds = (
            np.log(init_params["min_tau"]),
            np.log(init_params["max_tau"]),
        )
        invphi_bounds = (
            1 / init_params["max_phi"],
            1 / init_params["min_phi"],
        )

        params = {
            "pi": init_params.get("pi", np.ones(self.K_tumor) / self.K_tumor),
        }

        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]

            # 1. Baseline proportions
            lambda_g = compute_baseline_proportions(sx_data.X, sx_data.T, is_normal)
            params[f"{data_type}-lambda"] = lambda_g
            params[f"{data_type}-theta"] = estimate_tumor_proportion(sx_data, lambda_g)

            # 2. Find neutral cluster(s) — all clones (1,1)
            neutral_cids = [
                c
                for c in range(sx_data.G)
                if all(
                    sx_data.A[c, k] == 1 and sx_data.B[c, k] == 1
                    for k in range(sx_data.K)
                )
            ]
            assert len(neutral_cids) > 0, f"no neutral (1|1) cluster for {data_type}"

            if len(neutral_cids) == 1:
                c0 = neutral_cids[0]
                Y_neut = sx_data.Y[c0 : c0 + 1]
                D_neut = sx_data.D[c0 : c0 + 1]
                X_neut = sx_data.X[c0 : c0 + 1]
                lam_neut = lambda_g[c0]
            else:
                Y_neut = sx_data.Y[neutral_cids].sum(axis=0, keepdims=True)
                D_neut = sx_data.D[neutral_cids].sum(axis=0, keepdims=True)
                X_neut = sx_data.X[neutral_cids].sum(axis=0, keepdims=True)
                lam_neut = lambda_g[neutral_cids].sum()
            logging.info(
                f"{data_type}: neutral={neutral_cids}, "
                f"lambda={lam_neut:.6f}, "
                f"median_D={np.median(D_neut):.0f}, "
                f"median_X={np.median(X_neut):.0f}"
            )

            # 3. Global BB tau from neutral cluster
            if fit_mode in {"allele_only", "hybrid"}:
                Y_fit = Y_neut[:, :, None].astype(np.float64)
                D_fit = D_neut[:, :, None].astype(np.float64)
                global_tau = mle_tau(
                    Y_fit,
                    D_fit,
                    np.full_like(Y_fit, 0.5),
                    np.ones_like(Y_fit),
                    logtau_bounds,
                )
                params[f"{data_type}-tau"] = np.full(
                    sx_data.nrows_imbalanced,
                    global_tau,
                    dtype=np.float32,
                )
                logging.info(f"{data_type}: global tau={global_tau:.2f}")

            # 4. Global NB inv_phi from neutral cluster
            if fit_mode in {"total_only", "hybrid"}:
                X_fit = X_neut[:, :, None].astype(np.float64)
                mu_fit = (sx_data.T[None, :, None] * lam_neut).astype(np.float64)
                global_invphi = mle_invphi(
                    X_fit, mu_fit, np.ones_like(X_fit), invphi_bounds
                )
                params[f"{data_type}-inv_phi"] = np.full(
                    sx_data.nrows_aneuploid,
                    global_invphi,
                    dtype=np.float32,
                )
                logging.info(
                    f"{data_type}: global inv_phi={global_invphi:.4f} "
                    f"(phi={1 / global_invphi:.2f})"
                )

        # Build fix_params
        fix_params = {key: False for key in params}
        if init_fix_params is not None:
            for key in init_fix_params:
                if key in fix_params:
                    fix_params[key] = init_fix_params[key]

        # All spots participate in EM
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

            bb_mask = sx_data.MASK["IMBALANCED"] & (lambda_g > 0)
            nb_mask = sx_data.MASK["ANEUPLOID"] & (lambda_g > 0)

            if fit_mode in {"allele_only", "hybrid"}:
                MA, _ = sx_data.apply_mask_shallow(
                    "IMBALANCED", additional_mask=lambda_g > 0
                )
                tau_valid = params[f"{data_type}-tau"][
                    lambda_g[sx_data.MASK["IMBALANCED"]] > 0
                ]
                allele_ll = cond_betabin_logpmf_theta(
                    MA["Y"][:, tumor_idx],
                    MA["D"][:, tumor_idx],
                    tau_valid,
                    MA["BAF"][:, 1:],
                    rdrs_gk[bb_mask],
                    theta_tumor,
                )
                contrib = allele_ll.sum(axis=0)
                contrib[~tumor_mask_n, :] = 0.0
                global_lls += contrib

            if fit_mode in {"total_only", "hybrid"}:
                invphi_valid = params[f"{data_type}-inv_phi"][
                    lambda_g[sx_data.MASK["ANEUPLOID"]] > 0
                ]
                total_ll = cond_negbin_logpmf_theta(
                    sx_data.X[nb_mask][:, tumor_idx],
                    sx_data.T[tumor_idx],
                    lambda_g[nb_mask],
                    invphi_valid,
                    rdrs_gk[nb_mask],
                    theta_tumor,
                )
                contrib = total_ll.sum(axis=0)
                contrib[~tumor_mask_n, :] = 0.0
                global_lls += contrib

        global_lls += np.log(params["pi"])[None, :]
        log_marg = logsumexp(global_lls, axis=1)
        return np.sum(log_marg), log_marg, global_lls

    def _m_step(self, fit_mode, gamma, params, fix_params, t=0, eps=1e-10):
        N_t = self._N_tumor
        tumor_idx = self._tumor_idx

        self._update_pi(gamma, params, fix_params, N_t, self.K_tumor)

        # Per-spot theta update: sum Q across all data_types
        theta_keys = [
            f"{dt}-theta"
            for dt in self.data_types
            if f"{dt}-theta" in params and not fix_params.get(f"{dt}-theta", True)
        ]
        if theta_keys:
            purity_bounds = (1e-4, 1.0 - 1e-4)

            # Pre-compute per-modality masks and intermediates
            dt_info = []
            for data_type in self.data_types:
                if f"{data_type}-theta" not in params:
                    continue
                sx_data = self.data_sources[data_type]
                mask_n = self.modality_masks[data_type]
                lambda_g = params[f"{data_type}-lambda"]
                rdrs_gk = clone_rdr_gk(lambda_g, sx_data.C)[:, 1:]
                nb_mask = sx_data.MASK["ANEUPLOID"] & (lambda_g > 0)
                bb_mask = sx_data.MASK["IMBALANCED"] & (lambda_g > 0)
                dt_info.append(
                    {
                        "data_type": data_type,
                        "sx_data": sx_data,
                        "mask_n": mask_n,
                        "lambda_g": lambda_g,
                        "rdrs_gk": rdrs_gk,
                        "nb_mask": nb_mask,
                        "bb_mask": bb_mask,
                        "p_gk": sx_data.BAF[:, 1:],
                    }
                )

            theta_arr = np.full(self.N, 0.5, dtype=np.float64)
            # Initialize from first available modality's theta
            for info in dt_info:
                dt = info["data_type"]
                theta_arr = params[f"{dt}-theta"].copy()
                break

            for n in range(self.N):

                def neg_Q_theta(theta_val, _n=n):
                    theta_val = np.array([theta_val], dtype=float)
                    Q = 0.0
                    if _n in tumor_idx:
                        tidx = np.searchsorted(tumor_idx, _n)
                        w = gamma[tidx][None, :]
                    else:
                        w = np.ones((1, self.K_tumor)) / self.K_tumor

                    for info in dt_info:
                        if not info["mask_n"][_n]:
                            continue
                        sx = info["sx_data"]
                        lg = info["lambda_g"]
                        dt = info["data_type"]

                        if fit_mode in {"total_only", "hybrid"}:
                            ll_nb = cond_negbin_logpmf_theta(
                                sx.X[info["nb_mask"]][:, _n : _n + 1],
                                np.array([sx.T[_n]], dtype=float),
                                lg[info["nb_mask"]],
                                params[f"{dt}-inv_phi"][lg[sx.MASK["ANEUPLOID"]] > 0],
                                info["rdrs_gk"][info["nb_mask"]],
                                theta_val,
                            )
                            Q += np.sum(ll_nb[:, 0, :] * w)

                        if fit_mode in {"allele_only", "hybrid"}:
                            ll_bb = cond_betabin_logpmf_theta(
                                sx.Y[info["bb_mask"]][:, _n : _n + 1],
                                sx.D[info["bb_mask"]][:, _n : _n + 1],
                                params[f"{dt}-tau"][lg[sx.MASK["IMBALANCED"]] > 0],
                                info["p_gk"][info["bb_mask"]],
                                info["rdrs_gk"][info["bb_mask"]],
                                theta_val,
                            )
                            Q += np.sum(ll_bb[:, 0, :] * w)

                    return -Q

                res = minimize_scalar(
                    neg_Q_theta, bounds=purity_bounds, method="bounded"
                )
                theta_arr[n] = np.clip(res.x, 1e-4, 1.0 - 1e-4)

            # Set the same theta for all modalities
            for info in dt_info:
                params[f"{info['data_type']}-theta"] = theta_arr.copy()

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
        tumorprop_threshold=0.5,
    ):
        logging.info("Decode labels with MAP estimation")
        posteriors = self._e_step(fit_mode, params)
        tumor_clones = self.clones[1:]

        anns = self.barcodes.copy(deep=True)

        # Combine theta across modalities (average)
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
        logging.info(f"clone fractions (all spots): {clone_props}")

        return anns, clone_props
