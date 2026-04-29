import numpy as np
from scipy.special import logsumexp

from copytyping.inference.base_model import Base_Model
from copytyping.inference.model_utils import (
    clone_pi_gk,
    cond_betabin_logpmf,
    cond_negbin_logpmf,
    mle_invphi,
    mle_tau,
)


class Cell_Model(Base_Model):
    """Single-cell EM model. Assumes tumor purity=1 for each cell.
    Posteriors over all K clones (including normal).
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

    def _init_params(self, fit_mode, init_fix_params, init_params):
        params = {
            "pi": init_params.get("pi", np.ones(self.K) / self.K),
        }

        # Identify normals (skip in allele_only to avoid recursion
        # from _identify_normal_cells inner sub-EM)
        is_normal, init_labeling = (
            None,
            {"labels": np.full(self.N, "NA"), "max_posterior": np.zeros(self.N)},
        )
        if fit_mode != "allele_only":
            is_normal, init_labeling = self._identify_normal_cells(
                init_fix_params,
                init_params,
            )
        if fit_mode in {"total_only", "hybrid"}:
            self._init_lambda(params, is_normal)

        for data_type in self.data_types:
            tau_bounds = init_params["tau_bounds"]
            invphi_bounds = init_params["invphi_bounds"]

            if fit_mode in {"total_only", "hybrid"} and is_normal is not None:
                lambda_g = params[f"{data_type}-lambda"]
                params[f"{data_type}-inv_phi"] = self._init_invphi_from_normals(
                    data_type, lambda_g, is_normal, invphi_bounds
                )

            if fit_mode in {"allele_only", "hybrid"}:
                if is_normal is not None:
                    params[f"{data_type}-tau"] = self._init_tau_from_normals(
                        data_type, is_normal, tau_bounds
                    )
                else:
                    # allele_only mode: no normals yet, use geometric mean of bounds
                    params[f"{data_type}-tau"] = np.sqrt(tau_bounds[0] * tau_bounds[1])

        fix_params = {key: False for key in params}
        if init_fix_params is not None:
            for key in init_fix_params:
                if key in fix_params:
                    fix_params[key] = init_fix_params[key]

        return params, fix_params, init_labeling

    def compute_log_likelihood(self, fit_mode: str, params: dict):
        global_lls = params["ll_global"]
        global_lls[:] = 0.0

        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]
            mask_n = self.modality_masks[data_type]
            allele_mask = sx_data.MASK[self.allele_mask_id]
            ll_a = params[f"{data_type}-ll_allele"]
            ll_t = params[f"{data_type}-ll_total"]
            ll_a[:] = 0.0
            ll_t[:] = 0.0

            if fit_mode in {"allele_only", "hybrid"}:
                MA, _ = sx_data.apply_mask_shallow(mask_id=self.allele_mask_id)
                ll_a[allele_mask] = cond_betabin_logpmf(
                    MA["Y"], MA["D"], params[f"{data_type}-tau"], MA["BAF"]
                )
                ll_a[:, ~mask_n, :] = 0.0
                global_lls += ll_a.sum(axis=0)

            if fit_mode in {"total_only", "hybrid"}:
                lambda_g = params[f"{data_type}-lambda"]
                total_mask = sx_data.MASK[self.total_mask_id] & (lambda_g > 0)
                props_gk = clone_pi_gk(lambda_g, sx_data.C)[total_mask, :]
                ll_t[total_mask] = cond_negbin_logpmf(
                    sx_data.X[total_mask],
                    sx_data.T,
                    props_gk,
                    params[f"{data_type}-inv_phi"],
                )
                ll_t[:, ~mask_n, :] = 0.0
                global_lls += ll_t.sum(axis=0)

        global_lls += np.log(np.maximum(params["pi"], 1e-30))[None, :]
        log_marg = logsumexp(global_lls, axis=1)
        return np.sum(log_marg), log_marg, global_lls

    def _m_step(self, fit_mode, gamma, params, fix_params, t=0, eps=1e-10):
        self._update_pi(gamma, params, fix_params, self.N, self.K)

        gamma_gnk = gamma[None, :, :]  # (1, N, K)
        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]

            if fit_mode in {"allele_only", "hybrid"} and not fix_params.get(
                f"{data_type}-tau", True
            ):
                MA, _ = sx_data.apply_mask_shallow(mask_id=self.allele_mask_id)
                params[f"{data_type}-tau"] = mle_tau(
                    MA["Y"][:, :, None],
                    MA["D"][:, :, None],
                    MA["BAF"][:, None, :],
                    gamma_gnk,
                    tau_bounds=self._tau_bounds,
                )

            if fit_mode in {"total_only", "hybrid"} and not fix_params.get(
                f"{data_type}-inv_phi", True
            ):
                lambda_g = params[f"{data_type}-lambda"]
                total_mask = sx_data.MASK[self.total_mask_id] & (lambda_g > 0)
                props_gk = clone_pi_gk(lambda_g, sx_data.C)[total_mask, :]
                params[f"{data_type}-inv_phi"] = mle_invphi(
                    sx_data.X[total_mask][:, :, None],
                    props_gk[:, None, :] * sx_data.T[None, :, None],
                    gamma_gnk,
                    invphi_bounds=self._invphi_bounds,
                )
