import os
import sys
import copy

import numpy as np
import pandas as pd

from copytyping.utils import *
from copytyping.sx_data.sx_data import *
from copytyping.inference.model_utils import *
from copytyping.inference.cell_model import Cell_Model
from copytyping.inference.likelihood_funcs import *

from scipy.optimize import minimize_scalar
from scipy.special import softmax, expit, betaln, digamma, gammaln, logsumexp


##################################################
class Spot_Model:
    """EM model for spot-level data, variable tumor purity for each spot."""

    def __init__(
        self,
        barcodes: pd.DataFrame,
        haplo_blocks: pd.DataFrame,
        data_types: list,  # [U1, ...]
        mod_dirs: dict,
        modality="VISIUM",
        verbose=1,
    ) -> None:
        data_type = data_types[0]
        self.barcodes = barcodes
        self.haplo_blocks = haplo_blocks
        self.data_types = data_types
        self.mod_dirs = mod_dirs
        self.N = self.num_barcodes = len(barcodes)
        self.modality = modality
        self.data_sources = {
            data_type: SX_Data(barcodes, haplo_blocks, mod_dirs[data_type], data_type)
        }
        self.clones = self.data_sources[data_type].clones
        self.K = self.num_clones = len(self.clones)
        self.verbose = verbose
        return

    ##################################################
    def _init_params(
        self,
        init_fix_params=None,
        init_params=None,
        default_tau=50,
        default_phi=30,
        allele_post_thres=0.90,
        ref_label="path_label",
    ):
        params = {}
        if not init_params is None:
            params["pi"] = init_params.get("pi", None)
            default_phi = init_params.get("phi0", default_phi)
            default_tau = init_params.get("tau0", default_tau)

        if params.get("pi", None) is None:
            params["pi"] = np.ones(self.K) / self.K

        sx_data: SX_Data = self.data_sources[self.data_types[0]]
        if params.get(f"lambda", None) is None:
            # run allele only model to infer baseline proportions
            pure_model = Cell_Model(
                self.barcodes,
                self.haplo_blocks,
                self.data_types,
                self.mod_dirs,
                self.modality,
            )
            allele_params = pure_model.fit("allele_only")
            allele_anns, clone_props = pure_model.predict(
                "allele_only",
                allele_params,
                label="spot_label",
                posterior_thres=allele_post_thres,
            )
            is_normal_cell = (allele_anns["spot_label"] == "normal").to_numpy()
            num_normal_cells = np.sum(is_normal_cell)
            print(f"#estimated normal cells={num_normal_cells}/{self.num_barcodes}")
            barcodes = self.barcodes
            if ref_label in barcodes.columns:
                print("refine estimated normal spots by pathology annotation")
                black_list = ["tumor"]
                ref_labels = barcodes[ref_label].unique()
                print("All reference labels: ", ref_labels)
                print(f"Black list: ", black_list)
                path_mask = (~barcodes[ref_label].isin(black_list)).to_numpy()
                is_normal_cell = is_normal_cell & path_mask
                num_normal_cells = np.sum(is_normal_cell)
                print(
                    f"#after path annotation filtering={num_normal_cells}/{self.num_barcodes}"
                )
            assert num_normal_cells > 0, "no normal cells estimated in allele model"
            params["lambda"] = compute_baseline_proportions(
                sx_data.X, sx_data.T, is_normal_cell
            )

        if params.get("theta", None) is None:
            params["theta"] = estimate_tumor_proportion(sx_data, params["lambda"])

        if params.get("inv_phi", None) is None:
            params["inv_phi"] = np.full(
                sx_data.nrows_eff_feat,
                fill_value=1 / default_phi,
                dtype=np.float32,
            )
        if params.get("tau", None) is None:
            params["tau"] = np.full(
                sx_data.nrows_eff_allele,
                fill_value=default_tau,
                dtype=np.float32,
            )

        fix_params = {key: False for key in params.keys()}
        if not init_fix_params is None:
            for key in init_fix_params.keys():
                fix_params[key] = init_fix_params[key]

        return params, fix_params

    ##################################################
    def compute_log_likelihood(self, params: dict):
        global_lls = np.zeros((self.N, self.K), dtype=np.float32)

        data_type = self.data_types[0]
        sx_data: SX_Data = self.data_sources[data_type]
        lambda_g = params[f"{data_type}-lambda"]  # (G,)
        rdrs_gk = clone_rdr_gk(lambda_g, sx_data.C)

        # allele log-probs
        MA = sx_data.apply_mask_shallow(mask_id="IMBALANCED")
        allele_mask = lambda_g[sx_data.MASK["IMBALANCED"]] > 0
        allele_ll_mat = cond_betabin_logpmf_theta(
            MA["Y"][allele_mask],
            MA["D"][allele_mask],
            MA["tau"][allele_mask],
            MA["BAF"][allele_mask],
            rdrs_gk[(sx_data.MASK["IMBALANCED"]) & (lambda_g > 0)],
            params["theta"],
        )
        allele_lls = allele_ll_mat.sum(axis=0)  # (N,K)
        global_lls += allele_lls

        # total log-probs
        total_mask = lambda_g[sx_data.MASK["ANEUPLOID"]] > 0
        MF = sx_data.apply_mask_shallow(mask_id="ANEUPLOID")
        total_ll_mat = cond_negbin_logpmf_theta(
            MF["X"][total_mask],
            sx_data.T,
            lambda_g[total_mask],
            params[f"{data_type}-inv_phi"][total_mask],
            rdrs_gk[(sx_data.MASK["ANEUPLOID"]) & (lambda_g > 0)],
            params["theta"],
        )
        total_lls = total_ll_mat.sum(axis=0)  # (N,K)
        global_lls += total_lls

        global_lls += np.log(params["pi"])[None, :]  # (N,K)
        log_marg = logsumexp(global_lls, axis=1)  # (N,1)
        ll = np.sum(log_marg)
        return ll, log_marg, global_lls

    def _e_step(self, mode: str, params: dict, t=0) -> np.ndarray:
        """E-step, compute latent clone label posteriors (N, K)

        Args:
            params (dict): parameters

        Returns:
            np.ndarray: gammas (N,K)
        """
        ll, log_marg, global_lls = self.compute_log_likelihood(mode, params)

        # normalize
        gamma = np.exp(
            global_lls - logsumexp(global_lls, axis=1, keepdims=True)
        )  # softmax
        return gamma

    def _m_step(
        self,
        mode: str,
        gamma: np.ndarray,
        params: dict,
        fix_params: dict,
        share_invphi=True,
        share_tau=True,
        invphi_bounds=(1 / 100, 1 / 10),
        logtau_bounds=(np.log(50), np.log(200)),
        t=0,
        eps=1e-12,
    ):
        """M-step, update parameters

        Args:
            gammas (np.ndarray): (N,K)
            params (dict): parameter values from previous iteration.
            fix_params (dict): which parameter is fixed.
        """
        if not fix_params["pi"]:
            # update mixing density for clone assignments
            params["pi"] = np.sum(gamma, axis=0) / self.N

        gamma_gnk = gamma[None, :, :]  # (1, N, K)
        data_type = self.data_types[0]
        sx_data: SX_Data = self.data_sources[data_type]
        bin_info = sx_data.bin_info

        # parameters
        lambda_g = params[f"{data_type}-lambda"]
        rdrs_gk = clone_rdr_gk(lambda_g, sx_data.C)
        props_gk = clone_pi_gk(lambda_g, sx_data.C)
        p_gk = sx_data.BAF

        # masks
        nb_mask = (sx_data.MASK["IMBALANCED"]) & (lambda_g > 0)
        bb_mask = (sx_data.MASK["ANEUPLOID"]) & (lambda_g > 0)

        # NB inputs
        X_gn = sx_data.X[nb_mask, :]  # (G, N)
        T_n = sx_data.T  # (N,)

        # BB inputs
        Y_gn = sx_data.Y[bb_mask, :]  # (G, N)
        D_gn = sx_data.D[bb_mask, :]  # (G, N)

        # update tumor proportion with full likelihoods
        if not fix_params.get("theta", True):
            inv_phi_g = params[f"{data_type}-inv_phi"][nb_mask]
            tau_g = params[f"{data_type}-tau"][bb_mask]

            theta_arr = np.zeros_like(params["theta"], dtype=np.float32)
            for n in range(self.N):

                def neg_Q_theta(theta):
                    theta = np.array([theta], dtype=float)
                    ll_nb = cond_negbin_logpmf_theta(
                        X=X_gn[:, n : n + 1],
                        T=np.array([T_n[n]], dtype=float),
                        lam_g=lambda_g[nb_mask],
                        inv_phi=inv_phi_g,
                        rdrs_gk=rdrs_gk[nb_mask],
                        theta=theta,
                    )  # (Gtot, 1, K)

                    ll_bb = cond_betabin_logpmf_theta(
                        Y=Y_gn[:, n : n + 1],
                        D=D_gn[:, n : n + 1],
                        tau=tau_g,
                        p=p_gk[bb_mask],
                        rdrs_gk=rdrs_gk[bb_mask],
                        theta=theta,
                    )

                    Q = np.sum(ll_nb[:, 0, :] * gamma[n][None, :])
                    Q += np.sum(ll_bb[:, 0, :] * gamma[n][None, :])
                    return -Q

                res = minimize_scalar(
                    neg_Q_theta,
                    bounds=(1e-4, 1.0 - 1e-4),
                    method="bounded",
                    options={"xatol": 1e-4},
                )
                theta_arr[n] = np.clip(res.x, 1e-4, 1.0 - 1e-4)
            params["theta"] = theta_arr
        if not fix_params.get(f"{data_type}-inv_phi", True):
            X_gnk = X_gn[nb_mask][:, :, None]
            T_gnk = T_n[None, :, None]
            lam_gnk = lambda_g[nb_mask][:, None, None]
            rdrs_gnk = rdrs_gk[nb_mask][:, None, :]
            theta_gnk = params["theta"][None, :, None]

            mu_gnk = T_gnk * lam_gnk * (theta_gnk * rdrs_gnk + (1.0 - theta_gnk))
            mu_gnk = np.clip(mu_gnk, eps, None)

            if share_invphi:
                invphi_hat = mle_invphi(X_gnk, mu_gnk, gamma_gnk, invphi_bounds)
                params[f"{data_type}-inv_phi"][:] = invphi_hat
            else:
                # learn cnv-segment specific invphi
                for _, idx in (
                    bin_info.reset_index(drop=True)
                    .groupby("HB", sort=False)
                    .groups.items()
                ):
                    idx = np.asarray(idx, dtype=int)
                    invphi_hat = mle_invphi(
                        X_gnk[idx], mu_gnk[idx], gamma_gnk[idx], invphi_bounds
                    )
                    params[f"{data_type}-inv_phi"][idx] = invphi_hat

        if not fix_params.get(f"{data_type}-tau", True):
            Y_gnk = Y_gn[bb_mask][:, :, None]
            D_gnk = D_gn[bb_mask][:, :, None]
            rdrs_gnk = rdrs_gk[bb_mask][:, None, :]
            theta_gnk = params["theta"][None, :, None]
            p_gnk = p_gk[bb_mask][:, None, :]

            denom = rdrs_gnk * theta_gnk + (1.0 - theta_gnk)
            num = p_gnk * rdrs_gnk * theta_gnk + 0.5 * (1.0 - theta_gnk)
            p_hat = num / np.clip(denom, eps, None)
            p_hat = np.clip(p_hat, eps, 1.0 - eps)

            if share_tau:
                tau_hat = mle_tau(Y_gnk, D_gnk, p_hat, gamma_gnk, logtau_bounds)
                params[f"{data_type}-tau"][:] = tau_hat
            else:
                for _, idx in (
                    sx_data.bin_info[bb_mask]
                    .reset_index(drop=True)
                    .groupby("HB", sort=False)
                    .groups.items()
                ):
                    idx = np.asarray(idx, dtype=int)
                    tau_hat = mle_tau(
                        Y_gnk[idx],
                        D_gnk[idx],
                        p_hat[idx],
                        gamma_gnk[idx],
                        logtau_bounds,
                    )
                    params[f"{data_type}-tau"][idx] = tau_hat
        return

    def print_params(self, params: dict, mode="hybrid"):
        print("pi: ", params["pi"])
        for data_type in self.data_types:
            if mode != "allele_only":
                print(
                    f"{data_type}-inv_phi",
                    np.unique(params[f"{data_type}-inv_phi"]),
                )
            if mode != "total_only":
                print(f"{data_type}-tau", np.unique(params[f"{data_type}-tau"]))

    def fit(
        self,
        mode="spot",
        fix_params=None,
        init_params=None,
        max_iter=100,
        tol=1e-4,
        eps=1e-10,
        share_invphi=True,
        share_tau=True,
    ):
        """
        Parameters:
            1. per-spot tumor proportion \theta
            2. mixing weights \pi
            3. dispersion \phi and \tau
            4. baseline proportions \lambda
        Latent variable:
            1. per-spot clone label in {0...K}

        normal spots -> baseline proportion
        1. init baseline proportions
        2. init per-spot tumor proportion
        3. E-step, compute posterior
        4. M-step, update per-spot proportion via 1d optimization
        """
        print(f"Start inference")
        # Parameters
        params, fix_params = self._init_params(mode, fix_params, init_params)
        if self.verbose:
            self.print_params(params, mode)

        ll_trace = []
        param_trace = []
        prev_ll = -np.inf
        for t in range(1, max_iter):
            param_trace.append(copy.deepcopy(params))
            gamma = self._e_step(mode, params, t)
            self._m_step(
                mode,
                gamma,
                params,
                fix_params,
                share_invphi=share_invphi,
                share_tau=share_tau,
                t=t,
            )

            ll, _, _ = self.compute_log_likelihood(mode, params)
            ll_trace.append(ll)
            if self.verbose:
                print(f"iter={t:03d} log-likelihood = {ll:.6f}")
                self.print_params(params, mode)

            if t > 1:
                rel_change = np.abs(ll - prev_ll) / (np.abs(prev_ll) + eps)
                if rel_change < tol:
                    print(f"Converged at iteration {t} (delta = {rel_change:.2e})")
                    break
            prev_ll = ll

        # potentially plot the LL here TODO ll_trace
        # print(param_trace)
        return params

    def predict(
        self,
        mode: str,
        params: dict,
        label: str = "cell_label",
        posterior_thres: float = 0.5,  # min max posterior
        margin_thres: float = 0.1,  # min gap between top-1 and top-2
    ):
        print("Decode labels with MAP")
        posteriors = self._e_step(mode, params)  # (N, K)
        anns = self.barcodes.copy(deep=True)
        anns.loc[:, self.clones] = posteriors
        anns["tumor"] = 1 - anns["normal"]

        # compute max posterior and margin
        probs = anns[self.clones].to_numpy()  # (N, K)
        probs_sorted = np.sort(probs, axis=1)
        anns["max_posterior"] = probs_sorted[:, -1]
        second_post = probs_sorted[:, -2]
        anns["margin_delta"] = anns["max_posterior"] - second_post

        # MAP label
        anns[label] = anns[self.clones].idxmax(axis=1)

        # reject (NA) if low max_posterior or small margin_delta or
        mask_na = (anns["max_posterior"] < posterior_thres) | (
            anns["margin_delta"] < margin_thres
        )
        anns.loc[mask_na, label] = "NA"
        # see if NA labels can be assigned to normal vs tumor
        # normal_tumor_margin = np.abs(anns["normal"] - anns["tumor"])
        # anns.loc[
        #     mask_na
        #     & (anns["normal"] >= posterior_thres)
        #     & (normal_tumor_margin >= margin_thres),
        #     label,
        # ] = "NA/normal"
        # anns.loc[
        #     mask_na & (anns["tumor"] >= posterior_thres) & (normal_tumor_margin >= margin_thres),
        #     label,
        # ] = "NA/tumor"

        # clone proportions among all barcodes (including NA)
        clone_props = {
            clone: np.mean(anns[label].to_numpy() == clone) for clone in self.clones
        }
        return anns, clone_props
