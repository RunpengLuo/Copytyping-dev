import numpy as np
from scipy.special import logsumexp

from copytyping.inference.base_model import Base_Model
from copytyping.inference.model_utils import clone_pi_gk
from copytyping.inference.likelihoods import (
    cond_betabin_logpmf,
    cond_negbin_logpmf,
    mle_invphi,
    mle_tau,
)


class Cell_Model(Base_Model):
    """Single-cell EM model. Assumes tumor purity=1 for each cell.
    Posteriors over all K clones (including normal).
    """

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

        # Hard-EM init: dispersions at max bound (BB->Binomial, NB->Poisson)
        for assay_type in self.assay_types:
            if fit_mode in {"total", "allele_total"}:
                self.model_params[f"{assay_type}-inv_phi"] = self.invphi_bounds[1]
            if fit_mode in {"allele", "allele_total"}:
                self.model_params[f"{assay_type}-tau"] = self.tau_bounds[1]

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
                ll_a[allele_mask] = cond_betabin_logpmf(
                    count_data.count_B[allele_mask],
                    count_data.count_C[allele_mask],
                    params[f"{assay_type}-tau"],
                    count_data.cn_BAF[allele_mask],
                )
                global_lls += ll_a.sum(axis=0)

            if fit_mode in {"total", "allele_total"}:
                lambda_g = params[f"{assay_type}-lambda"]
                total_mask = count_data.total_mask[self.total_mask_id] & (lambda_g > 0)
                props_gk = clone_pi_gk(lambda_g, count_data.cn_C)[total_mask, :]
                ll_t[total_mask] = cond_negbin_logpmf(
                    count_data.count_X[total_mask],
                    self.T[assay_type],
                    props_gk,
                    params[f"{assay_type}-inv_phi"],
                )
                global_lls += ll_t.sum(axis=0)

        # global pi prior: log pi[k] added per cell
        global_lls += np.log(np.maximum(params["pi"], 1e-30))[None, :]
        log_marg = logsumexp(global_lls, axis=1)
        return np.sum(log_marg), log_marg, global_lls

    def _m_step(self, fit_mode: str, gamma: np.ndarray, t: int = 0) -> None:
        params = self.model_params
        self._update_pi(gamma, self.num_barcodes, self.num_clones)
        fix_params = self.fix_model_params

        for assay_type in self.assay_types:
            count_data = self.count_data[assay_type]

            if fit_mode in {"allele", "allele_total"} and not fix_params.get(
                f"{assay_type}-tau", True
            ):
                am = count_data.allele_mask[self.allele_mask_id]
                Y_am = count_data.count_B[am]
                D_am = count_data.count_C[am]
                BAF_am = count_data.cn_BAF[am]
                params[f"{assay_type}-tau"] = mle_tau(
                    Y_am[:, :, None],
                    D_am[:, :, None],
                    BAF_am[:, None, :],
                    gamma[None, :, :],
                    tau_bounds=self.tau_bounds,
                )

            if fit_mode in {"total", "allele_total"} and not fix_params.get(
                f"{assay_type}-inv_phi", True
            ):
                lambda_g = params[f"{assay_type}-lambda"]
                total_mask = count_data.total_mask[self.total_mask_id] & (lambda_g > 0)
                props_gk = clone_pi_gk(lambda_g, count_data.cn_C)[total_mask, :]
                X_tm = count_data.count_X[total_mask]
                T = self.T[assay_type]
                params[f"{assay_type}-inv_phi"] = mle_invphi(
                    X_tm[:, :, None],
                    props_gk[:, None, :] * T[None, :, None],
                    gamma[None, :, :],
                    invphi_bounds=self.invphi_bounds,
                )
