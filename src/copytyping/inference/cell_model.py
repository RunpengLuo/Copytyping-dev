import logging

import numpy as np
from scipy.special import logsumexp

from copytyping.inference.base_model import Base_Model
from copytyping.inference.model_utils import clone_pi_gk
from copytyping.inference.likelihoods import (
    cond_betabin_logpmf,
    cond_negbin_logpmf,
    expand_state_map,
    mle_invphi,
    mle_invphi_per_state,
    mle_tau,
    mle_tau_per_state,
)


class Cell_Model(Base_Model):
    """Single-cell EM model. Assumes tumor purity=1 for each cell.
    Posteriors over all K clones (including normal).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logging.info("=" * 20 + " Start Cell_Model " + "=" * 20)

    def _init_params(self, fit_mode: str) -> dict:
        # global clone mixture self.model_params["pi"] (shape (K,)) set in __init__

        # reference cells (skip in allele to avoid sub-EM recursion)
        is_reference, ref_clone = None, 0
        init_labeling = {
            "labels": np.full(self.num_barcodes, "NA"),
            "max_posterior": np.zeros(self.num_barcodes),
        }
        if fit_mode != "allele":
            is_reference, ref_clone, init_labeling = self._estimate_reference_cells()
        if fit_mode in {"total", "allele_total"}:
            self._init_lambda(is_reference, ref_clone)

        # Hard-EM init: dispersions at max bound (BB->Binomial, NB->Poisson).
        # *_states holds the per-CNA-state dispersion map (None until first M-step
        # in per-state mode; stays None in --share_dispersion mode).
        for assay_type in self.assay_types:
            if fit_mode in {"total", "allele_total"}:
                self.model_params[f"{assay_type}-inv_phi"] = self.invphi_bounds[1]
                self.model_params[f"{assay_type}-inv_phi_states"] = None
            if fit_mode in {"allele", "allele_total"}:
                self.model_params[f"{assay_type}-tau"] = self.tau_bounds[1]
                self.model_params[f"{assay_type}-tau_states"] = None

        return init_labeling

    def compute_log_likelihood(
        self, fit_mode: str
    ) -> tuple[float, np.ndarray, np.ndarray]:
        params = self.model_params
        global_lls = params["ll_global"]
        global_lls[:] = 0.0

        for assay_type in self.assay_types:
            count_data = self.count_data[assay_type]
            allele_mask = count_data.allele_mask[self.allele_mask_id]
            ll_a = params[f"{assay_type}-ll_allele"]
            ll_t = params[f"{assay_type}-ll_total"]
            ll_a[:] = 0.0
            ll_t[:] = 0.0

            if fit_mode in {"allele", "allele_total"}:
                # per-state tau is a (G, K) array -> slice to the masked bins;
                # shared tau is a scalar -> pass through
                tau = params[f"{assay_type}-tau"]
                tau = tau[allele_mask] if np.ndim(tau) == 2 else tau
                ll_a[allele_mask] = cond_betabin_logpmf(
                    count_data.count_B[allele_mask],
                    count_data.count_C[allele_mask],
                    tau,
                    count_data.cn_BAF[allele_mask],
                )
                global_lls += ll_a.sum(axis=0)

            if fit_mode in {"total", "allele_total"}:
                lambda_g = params[f"{assay_type}-lambda"]
                total_mask = count_data.total_mask[self.total_mask_id] & (lambda_g > 0)
                props_gk = clone_pi_gk(lambda_g, count_data.cn_C)[total_mask, :]
                inv_phi = params[f"{assay_type}-inv_phi"]
                inv_phi = inv_phi[total_mask] if np.ndim(inv_phi) == 2 else inv_phi
                ll_t[total_mask] = cond_negbin_logpmf(
                    count_data.count_X[total_mask],
                    self.count_T[assay_type],
                    props_gk,
                    inv_phi,
                )
                global_lls += ll_t.sum(axis=0)

        # global pi prior: log pi[k] added per cell
        global_lls += np.log(np.maximum(params["pi"], 1e-30))[None, :]
        log_marg = logsumexp(global_lls, axis=1)
        return np.sum(log_marg), log_marg, global_lls

    def _m_step(self, fit_mode: str, gamma: np.ndarray, t: int = 0):
        params = self.model_params
        self._update_pi(gamma, self.num_barcodes, self.num_clones)

        for assay_type in self.assay_types:
            count_data = self.count_data[assay_type]

            if fit_mode in {"allele", "allele_total"} and self.update_tau:
                if self.share_dispersion:
                    # one tau, pooled over the imbalanced bins only
                    am = count_data.allele_mask[self.allele_mask_id]
                    params[f"{assay_type}-tau"] = mle_tau(
                        count_data.count_B[am][:, :, None],
                        count_data.count_C[am][:, :, None],
                        count_data.cn_BAF[am][:, None, :],
                        gamma[None, :, :],
                        tau_bounds=self.tau_bounds,
                    )
                    params[f"{assay_type}-tau_states"] = None
                else:
                    # per-state tau estimated over ALL clusters (so neutral
                    # states use their full genome-wide signal), then broadcast
                    tau_states = mle_tau_per_state(
                        count_data.count_B,
                        count_data.count_C,
                        count_data.cn_BAF,
                        gamma,
                        count_data.cn_A,
                        count_data.cn_B,
                        tau_bounds=self.tau_bounds,
                    )
                    params[f"{assay_type}-tau"] = expand_state_map(
                        count_data.cn_A, count_data.cn_B, tau_states, self.tau_bounds[1]
                    )
                    params[f"{assay_type}-tau_states"] = tau_states

            if fit_mode in {"total", "allele_total"} and self.update_invphi:
                lambda_g = params[f"{assay_type}-lambda"]
                count_T = self.count_T[assay_type]
                props_gk = clone_pi_gk(lambda_g, count_data.cn_C)
                if self.share_dispersion:
                    # one inv_phi, pooled over the aneuploid bins only
                    tm = count_data.total_mask[self.total_mask_id] & (lambda_g > 0)
                    params[f"{assay_type}-inv_phi"] = mle_invphi(
                        count_data.count_X[tm][:, :, None],
                        props_gk[tm][:, None, :] * count_T[None, :, None],
                        gamma[None, :, :],
                        invphi_bounds=self.invphi_bounds,
                    )
                    params[f"{assay_type}-inv_phi_states"] = None
                else:
                    # per-state inv_phi over all lambda>0 bins, then broadcast
                    valid = lambda_g > 0
                    invphi_states = mle_invphi_per_state(
                        count_data.count_X[valid],
                        props_gk[valid],
                        count_T,
                        gamma,
                        count_data.cn_A[valid],
                        count_data.cn_B[valid],
                        invphi_bounds=self.invphi_bounds,
                    )
                    params[f"{assay_type}-inv_phi"] = expand_state_map(
                        count_data.cn_A,
                        count_data.cn_B,
                        invphi_states,
                        self.invphi_bounds[1],
                    )
                    params[f"{assay_type}-inv_phi_states"] = invphi_states
