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
        args=None,
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
            args=args,
        )

    def _init_params(self, fit_mode):
        # pi per rep: shape (R, K)
        pi_per_rep = np.tile(self.model_params["pi"], (self.num_reps, 1))  # (R, K)
        params = {"pi": pi_per_rep}

        # reference cells (skip in allele_only to avoid sub-EM recursion)
        is_reference, ref_clone, init_labeling = (
            None,
            0,
            {
                "labels": np.full(self.num_barcodes, "NA"),
                "max_posterior": np.zeros(self.num_barcodes),
            },
        )
        if fit_mode != "allele_only":
            is_reference, ref_clone, init_labeling = self._estimate_reference_cells()
        if fit_mode in {"total_only", "hybrid"}:
            self._init_lambda(params, is_reference, ref_clone)

        # Hard-EM init: dispersions at max bound (BB->Binomial, NB->Poisson)
        tau_bounds = self.model_params["tau_bounds"]
        invphi_bounds = self.model_params["invphi_bounds"]
        for data_type in self.data_types:
            if fit_mode in {"total_only", "hybrid"}:
                params[f"{data_type}-inv_phi"] = np.full(
                    self.num_reps, invphi_bounds[1]
                )
            if fit_mode in {"allele_only", "hybrid"}:
                params[f"{data_type}-tau"] = np.full(self.num_reps, tau_bounds[1])

        self._finalize_fix_params(params)
        return params, init_labeling

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
                tau_per_spot = params[f"{data_type}-tau"][self.rep_idx]  # (N,)
                ll_a[allele_mask] = cond_betabin_logpmf(
                    MA["Y"], MA["D"], tau_per_spot, MA["BAF"]
                )
                ll_a[:, ~mask_n, :] = 0.0
                global_lls += ll_a.sum(axis=0)

            if fit_mode in {"total_only", "hybrid"}:
                lambda_g = params[f"{data_type}-lambda"]
                total_mask = sx_data.MASK[self.total_mask_id] & (lambda_g > 0)
                props_gk = clone_pi_gk(lambda_g, sx_data.C)[total_mask, :]
                invphi_per_spot = params[f"{data_type}-inv_phi"][self.rep_idx]  # (N,)
                ll_t[total_mask] = cond_negbin_logpmf(
                    sx_data.X[total_mask],
                    sx_data.T,
                    props_gk,
                    invphi_per_spot,
                )
                ll_t[:, ~mask_n, :] = 0.0
                global_lls += ll_t.sum(axis=0)

        # Per-rep pi prior: log pi[rep_idx[n], k] added per cell
        pi = params["pi"]  # (R, K)
        global_lls += np.log(np.maximum(pi[self.rep_idx], 1e-30))
        log_marg = logsumexp(global_lls, axis=1)
        return np.sum(log_marg), log_marg, global_lls

    def _m_step(self, fit_mode, gamma, params, t=0):
        self._update_pi(gamma, params, self.num_barcodes, self.num_clones)
        fix_params = self.fix_model_params

        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]

            if fit_mode in {"allele_only", "hybrid"} and not fix_params.get(
                f"{data_type}-tau", True
            ):
                MA, _ = sx_data.apply_mask_shallow(mask_id=self.allele_mask_id)
                tau_arr = params[f"{data_type}-tau"].copy()
                for r in range(self.num_reps):
                    mask = self.rep_idx == r
                    if mask.sum() == 0:
                        continue
                    tau_arr[r] = mle_tau(
                        MA["Y"][:, mask][:, :, None],
                        MA["D"][:, mask][:, :, None],
                        MA["BAF"][:, None, :],
                        gamma[mask][None, :, :],
                        tau_bounds=self._tau_bounds,
                    )
                params[f"{data_type}-tau"] = tau_arr

            if fit_mode in {"total_only", "hybrid"} and not fix_params.get(
                f"{data_type}-inv_phi", True
            ):
                lambda_g = params[f"{data_type}-lambda"]
                total_mask = sx_data.MASK[self.total_mask_id] & (lambda_g > 0)
                props_gk = clone_pi_gk(lambda_g, sx_data.C)[total_mask, :]
                invphi_arr = params[f"{data_type}-inv_phi"].copy()
                for r in range(self.num_reps):
                    mask = self.rep_idx == r
                    if mask.sum() == 0:
                        continue
                    invphi_arr[r] = mle_invphi(
                        sx_data.X[total_mask][:, mask][:, :, None],
                        props_gk[:, None, :] * sx_data.T[mask][None, :, None],
                        gamma[mask][None, :, :],
                        invphi_bounds=self._invphi_bounds,
                    )
                params[f"{data_type}-inv_phi"] = invphi_arr
