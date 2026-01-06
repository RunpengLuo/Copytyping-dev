import os
import sys
import copy

import numpy as np
import pandas as pd

from copytyping.utils import *
from copytyping.sx_data.sx_data import *
from copytyping.inference.model_utils import *
from copytyping.inference.likelihood_funcs import *

from scipy.optimize import minimize_scalar
from scipy.special import softmax, expit, betaln, digamma, gammaln, logsumexp


##################################################
class Cell_Model:
    """Single-cell EM model, no spatial information, tumor purity=1 for each cell."""

    def __init__(
        self,
        barcodes: pd.DataFrame,
        haplo_blocks: pd.DataFrame,
        data_types: list,
        mod_dirs: dict,
        modality: str,
        verbose=1,
    ) -> None:
        self.barcodes = barcodes
        self.data_types = data_types
        self.N = self.num_barcodes = len(barcodes)
        self.modality = modality
        self.data_sources = {}
        for data_type in self.data_types:
            self.data_sources[data_type] = SX_Data(
                barcodes, haplo_blocks, mod_dirs[data_type], data_type
            )
        self.clones = self.data_sources[self.data_types[0]].clones
        self.K = self.num_clones = len(self.clones)
        self.verbose = verbose
        return

    ##################################################
    def _init_params(
        self,
        mode: str,
        init_fix_params=None,
        init_params=None,
        default_tau=50,
        default_phi=30,
    ):
        params = {}
        if not init_params is None:
            params["pi"] = init_params.get("pi", None)
            default_phi = init_params.get("phi0", default_phi)
            default_tau = init_params.get("tau0", default_tau)

        if params.get("pi", None) is None:
            params["pi"] = np.ones(self.K) / self.K

        # initialize baseline proportion if not allele_only mode
        if mode != "allele_only":
            for data_type in self.data_types:
                if params.get(f"{data_type}-lambda", None) is None:
                    params[f"{data_type}-lambda"] = (
                        self.initialize_baseline_proportions(data_type)
                    )
                if params.get(f"{data_type}-inv_phi", None) is None:
                    params[f"{data_type}-inv_phi"] = np.full(
                        self.data_sources[data_type].nrows_eff_feat,
                        fill_value=1 / default_phi,
                        dtype=np.float32,
                    )

        if mode != "total_only":
            for data_type in self.data_types:
                if params.get(f"{data_type}-tau", None) is None:
                    params[f"{data_type}-tau"] = np.full(
                        self.data_sources[data_type].nrows_eff_allele,
                        fill_value=default_tau,
                        dtype=np.float32,
                    )
        if any(data_type.startswith("VISIUM") for data_type in self.data_types):
            assert mode == "allele_only"
            data_type = self.data_types[0]
            if params.get(f"{data_type}-tau", None) is None:
                params[f"{data_type}-tau"] = np.full(
                    self.data_sources[data_type].nrows_eff_allele,
                    fill_value=default_tau,
                    dtype=np.float32,
                )

        fix_params = {key: False for key in params.keys()}
        if not init_fix_params is None:
            for key in init_fix_params.keys():
                fix_params[key] = init_fix_params[key]

        return params, fix_params

    def initialize_baseline_proportions(
        self,
        data_type: str,
        ref_label="cell_type",
        black_list=[
            "Tumor_cell",
            "tumor",
            "Tumor",  # tumor labels / variants
            "Doublet",
            "doublet",  # artifacts / ambiguous
            "Unknown",
            "NA",  # missing labels
        ],
    ):
        """this returns baseline proportions for all bins"""
        print(f"initialize baseline proportion for {data_type}")
        if ref_label in self.barcodes.columns:
            cell_types = self.barcodes["cell_type"].unique()
            print("All celltypes: ", cell_types)
            print(f"Black list: ", black_list)
            is_normal_cell = (~self.barcodes["cell_type"].isin(black_list)).to_numpy()
        else:
            print("infer normal cells using allele model")
            # TODO allele model
            is_normal_cell = np.full(self.num_barcodes, fill_value=False)
            pass

        num_normal_cells = np.sum(is_normal_cell)
        print(f"#estimated normal cells={num_normal_cells}/{self.num_barcodes}")
        # TODO better normal cell selection
        assert num_normal_cells > 0
        base_props = compute_baseline_proportions(
            self.data_sources[data_type].X,
            self.data_sources[data_type].T,
            is_normal_cell,
        )
        return base_props

    ##################################################
    def compute_log_likelihood(self, mode: str, params: dict):
        global_lls = np.zeros((self.N, self.K), dtype=np.float32)
        # sum over all modalities
        for data_type in self.data_types:
            sx_data: SX_Data = self.data_sources[data_type]
            # allele log-probs
            if mode != "total_only":
                M = sx_data.apply_mask_shallow(mask_id="IMBALANCED")
                allele_ll_mat = cond_betabin_logpmf(
                    M["Y"], M["D"], params[f"{data_type}-tau"], M["BAF"]
                )
                allele_lls = allele_ll_mat.sum(axis=0)  # (N,K)
                global_lls += allele_lls

            # total log-probs
            if mode != "allele_only":
                lambda_g = params[f"{data_type}-lambda"]
                props_gk = clone_pi_gk(lambda_g, sx_data.C)
                props_gk_cnv = props_gk[(sx_data.MASK["ANEUPLOID"]) & (lambda_g > 0), :]

                mask = lambda_g[sx_data.MASK["ANEUPLOID"]] > 0
                M = sx_data.apply_mask_shallow(mask_id="ANEUPLOID")
                total_ll_mat = cond_negbin_logpmf(
                    M["X"][mask],
                    sx_data.T,
                    props_gk_cnv,
                    params[f"{data_type}-inv_phi"][mask],
                )
                total_lls = total_ll_mat.sum(axis=0)  # (N,K)
                global_lls += total_lls

        global_lls += np.log(params["pi"])[None, :]  # (N,K)
        log_marg = logsumexp(global_lls, axis=1)  # (N,1)
        ll = np.sum(log_marg)
        return ll, log_marg, global_lls

    def _e_step(self, mode: str, params: dict, t=0) -> np.ndarray:
        """compute allele and total log-probs, summed over modalities

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
    ):
        """m-step

        Args:
            gammas (np.ndarray): (N,K)
            params (dict): parameter values from previous iteration.
            fix_params (dict): which parameter is fixed.
        """
        if not fix_params["pi"]:
            # update mixing density for clone assignments
            params["pi"] = np.sum(gamma, axis=0) / self.N

        gamma_gnk = gamma[None, :, :]  # (1, N, K)
        for data_type in self.data_types:
            sx_data: SX_Data = self.data_sources[data_type]
            if (
                not fix_params.get(f"{data_type}-inv_phi", True)
                and mode != "allele_only"
            ):
                # update NB over-dispersion
                lambda_g = params[f"{data_type}-lambda"]
                props_gk = clone_pi_gk(lambda_g, sx_data.C)[
                    (sx_data.MASK["ANEUPLOID"]) & (lambda_g > 0), :
                ]
                M = sx_data.apply_mask_shallow(mask_id="ANEUPLOID")
                X_gnk = M["X"][lambda_g[sx_data.MASK["ANEUPLOID"]] > 0][
                    :, :, None
                ]  # (G, N, 1)
                mu_gnk = props_gk[:, None, :] * sx_data.T[None, :, None]  # (G, 1, K)
                if share_invphi:
                    const_logfact = -gammaln(X_gnk + 1.0)

                    def neg_E_loglik(invphi):
                        if invphi <= 0.0:
                            return np.inf

                        r = 1.0 / invphi  # r in NB
                        log_r = np.log(r)
                        log_r_plus_mu = np.log(r + mu_gnk)

                        log_pmf = (
                            gammaln(X_gnk + r)
                            - gammaln(r)
                            + const_logfact
                            + r * (log_r - log_r_plus_mu)
                            + X_gnk * (np.log(mu_gnk) - log_r_plus_mu)
                        )

                        E_loglik = np.sum(gamma_gnk * log_pmf)
                        return -E_loglik

                    res = minimize_scalar(
                        neg_E_loglik,
                        bounds=invphi_bounds,
                        method="bounded",
                        options={"xatol": 1e-8},
                    )
                    invphi_hat = float(
                        np.clip(res.x, invphi_bounds[0], invphi_bounds[1])
                    )
                    params[f"{data_type}-inv_phi"][:] = invphi_hat
                else:
                    # learn cnv-segment specific invphi
                    for k, v in (
                        M["bin_info"]
                        .reset_index(drop=True)
                        .groupby("HB", sort=False)
                        .groups.items()
                    ):
                        const_logfact = -gammaln(X_gnk[v] + 1.0)

                        def neg_E_loglik(invphi):
                            if invphi <= 0.0:
                                return np.inf

                            r = 1.0 / invphi  # r in NB
                            log_r = np.log(r)
                            log_r_plus_mu = np.log(r + mu_gnk[v])

                            log_pmf = (
                                gammaln(X_gnk[v] + r)
                                - gammaln(r)
                                + const_logfact
                                + r * (log_r - log_r_plus_mu)
                                + X_gnk[v] * (np.log(mu_gnk[v]) - log_r_plus_mu)
                            )

                            E_loglik = np.sum(gamma_gnk * log_pmf)
                            return -E_loglik

                        res = minimize_scalar(
                            neg_E_loglik,
                            bounds=invphi_bounds,
                            method="bounded",
                            options={"xatol": 1e-8},
                        )
                        invphi_hat = float(
                            np.clip(res.x, invphi_bounds[0], invphi_bounds[1])
                        )
                        params[f"{data_type}-inv_phi"][v] = invphi_hat
            if not fix_params.get(f"{data_type}-tau", True) and mode != "total_only":
                # update BB over-dispersion tau
                M = sx_data.apply_mask_shallow(mask_id="IMBALANCED")
                p_gnk = M["BAF"][:, None, :]  # (G, 1, K)
                Y_gnk = M["Y"][:, :, None]  # (G, N, 1)
                D_gnk = M["D"][:, :, None]  # (G, N, 1)
                X_gnk = D_gnk - Y_gnk  # (G, N, 1)
                if share_tau:
                    log_binom_const = (
                        gammaln(D_gnk + 1.0)
                        - gammaln(Y_gnk + 1.0)
                        - gammaln(X_gnk + 1.0)
                    )

                    def neg_Q_logtau(logtau):
                        tau = np.exp(logtau)
                        alpha = tau * p_gnk
                        beta = tau * (1.0 - p_gnk)

                        log_pmf = (
                            log_binom_const
                            + betaln(Y_gnk + alpha, X_gnk + beta)
                            - betaln(alpha, beta)
                        )

                        Q = np.sum(gamma_gnk * log_pmf)
                        return -Q  # minimize negative expected loglik

                    res = minimize_scalar(
                        neg_Q_logtau,
                        bounds=logtau_bounds,
                        method="bounded",
                        options={"xatol": 1e-6},
                    )
                    tau_hat = np.exp(res.x)
                    params[f"{data_type}-tau"][:] = tau_hat
                else:
                    # learn cnv-segment specific tau
                    for k, v in (
                        M["bin_info"]
                        .reset_index(drop=True)
                        .groupby("HB", sort=False)
                        .groups.items()
                    ):
                        log_binom_const = (
                            gammaln(D_gnk[v] + 1.0)
                            - gammaln(Y_gnk[v] + 1.0)
                            - gammaln(X_gnk[v] + 1.0)
                        )

                        def neg_Q_logtau(logtau):
                            tau = np.exp(logtau)
                            alpha = tau * p_gnk[v]
                            beta = tau * (1.0 - p_gnk[v])

                            log_pmf = (
                                log_binom_const
                                + betaln(Y_gnk[v] + alpha, X_gnk[v] + beta)
                                - betaln(alpha, beta)
                            )

                            Q = np.sum(gamma_gnk * log_pmf)
                            return -Q  # minimize negative expected loglik

                        res = minimize_scalar(
                            neg_Q_logtau,
                            bounds=logtau_bounds,
                            method="bounded",
                            options={"xatol": 1e-6},
                        )
                        tau_hat = np.exp(res.x)
                        params[f"{data_type}-tau"][v] = tau_hat

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
        mode="hybrid",
        fix_params=None,
        init_params=None,
        max_iter=100,
        tol=1e-4,
        eps=1e-10,
        share_invphi=True,
        share_tau=True,
    ):
        assert mode in ["hybrid", "allele_only", "total_only"]
        print(f"Start inference, mode={mode}")
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
        self.params = params
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
        # clone proportions among all barcodes (including NA)
        clone_props = {
            clone: np.mean(anns[label].to_numpy() == clone) for clone in self.clones
        }
        return anns, clone_props
