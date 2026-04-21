import logging

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
        is_normal = None
        if fit_mode != "allele_only":
            is_normal = self._identify_normal_cells(
                init_fix_params,
                init_params,
            )
        if fit_mode in {"total_only", "hybrid"}:
            self._init_lambda(params, is_normal)

        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]

            if fit_mode in {"total_only", "hybrid"}:
                n_total = int(sx_data.MASK[self.total_mask_id].sum())
                params[f"{data_type}-inv_phi"] = np.full(
                    n_total,
                    1 / init_params["phi0"],
                    dtype=np.float32,
                )

            if fit_mode in {"allele_only", "hybrid"}:
                n_allele = int(sx_data.MASK[self.allele_mask_id].sum())
                params[f"{data_type}-tau"] = np.full(
                    n_allele,
                    init_params["tau0"],
                    dtype=np.float32,
                )

        fix_params = {key: False for key in params}
        if init_fix_params is not None:
            for key in init_fix_params:
                if key in fix_params:
                    fix_params[key] = init_fix_params[key]

        return params, fix_params

    def compute_log_likelihood(self, fit_mode: str, params: dict):
        global_lls = np.zeros((self.N, self.K), dtype=np.float32)

        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]
            mask_n = self.modality_masks[data_type]

            if fit_mode in {"allele_only", "hybrid"}:
                MA, _ = sx_data.apply_mask_shallow(mask_id=self.allele_mask_id)
                allele_ll = cond_betabin_logpmf(
                    MA["Y"], MA["D"], params[f"{data_type}-tau"], MA["BAF"]
                )
                contrib = allele_ll.sum(axis=0)
                contrib[~mask_n, :] = 0.0
                global_lls += contrib

            if fit_mode in {"total_only", "hybrid"}:
                lambda_g = params[f"{data_type}-lambda"]
                total_mask = sx_data.MASK[self.total_mask_id] & (lambda_g > 0)
                props_gk = clone_pi_gk(lambda_g, sx_data.C)[total_mask, :]
                inv_phis = params[f"{data_type}-inv_phi"][
                    lambda_g[sx_data.MASK[self.total_mask_id]] > 0
                ]
                total_ll = cond_negbin_logpmf(
                    sx_data.X[total_mask],
                    sx_data.T,
                    props_gk,
                    inv_phis,
                )
                contrib = total_ll.sum(axis=0)
                contrib[~mask_n, :] = 0.0
                global_lls += contrib

        global_lls += np.log(np.maximum(params["pi"], 1e-30))[None, :]
        log_marg = logsumexp(global_lls, axis=1)
        return np.sum(log_marg), log_marg, global_lls

    def _m_step(self, fit_mode, gamma, params, fix_params, t=0, eps=1e-10):
        self._update_pi(gamma, params, fix_params, self.N, self.K)

        gamma_gnk = gamma[None, :, :]  # (1, N, K)
        for data_type in self.data_types:
            sx_data = self.data_sources[data_type]

            # NB dispersion (shared across all bins)
            if (
                fit_mode in {"total_only", "hybrid"}
                and not fix_params[f"{data_type}-inv_phi"]
                and sx_data.MASK[self.total_mask_id].sum() > 0
            ):
                lambda_g = params[f"{data_type}-lambda"]
                total_mask = sx_data.MASK[self.total_mask_id] & (lambda_g > 0)
                props_gk = clone_pi_gk(lambda_g, sx_data.C)[total_mask]
                mu_gnk = props_gk[:, None, :] * sx_data.T[None, :, None]
                X_gnk = sx_data.X[total_mask][:, :, None]
                invphi_est = mle_invphi(
                    X_gnk,
                    mu_gnk,
                    gamma_gnk,
                    invphi_bounds=self._invphi_bounds,
                )
                params[f"{data_type}-inv_phi"][:] = invphi_est
                logging.debug(
                    f"{data_type} inv_phi={invphi_est:.4f} (phi={1 / invphi_est:.2f})"
                )

            # BB dispersion (shared across all bins)
            if (
                fit_mode in {"allele_only", "hybrid"}
                and not fix_params[f"{data_type}-tau"]
                and sx_data.MASK[self.allele_mask_id].sum() > 0
            ):
                MA, _ = sx_data.apply_mask_shallow(mask_id=self.allele_mask_id)
                p_gnk = MA["BAF"][:, None, :]
                Y_gnk = MA["Y"][:, :, None]
                D_gnk = MA["D"][:, :, None]
                tau_est = mle_tau(
                    Y_gnk,
                    D_gnk,
                    p_gnk,
                    gamma_gnk,
                    tau_bounds=self._tau_bounds,
                )
                params[f"{data_type}-tau"][:] = tau_est
                logging.debug(f"{data_type} tau={tau_est:.2f}")
