import numpy as np
import pandas as pd
from scipy.special import logsumexp

from copytyping.inference.base_model import Base_Model
from copytyping.inference.count_data import Count_Data
from copytyping.inference.model_utils import (
    clone_rdr_gk,
    estimate_tumor_proportion,
)
from copytyping.inference.likelihoods import (
    cond_betabin_logpmf_theta,
    cond_negbin_logpmf_theta,
)


class Spot_Model(Base_Model):
    """Purity-only spot model for spatial data (no normal clone in EM).

    Latent variables per spot:
    - z_n in {clone1, ..., cloneK} (tumor clones only)
    - θ_n in [0, 1] (tumor purity, 0 = pure normal)

    Normal/tumor assignment is decided post-EM by thresholding θ.
    This avoids the identifiability issue of having both a normal clone
    and θ→0 explain the same observation.
    """

    def __init__(
        self,
        count_data: dict[str, Count_Data],
        platform: str,
        assay_types: list[str],
        **kwargs,
    ):
        super().__init__(count_data, platform, assay_types, **kwargs)
        self.num_em_clones = len(self.tumor_clones)

    def _init_params(self, fit_mode: str) -> dict:
        assert not self.no_normal, "no_normal is single-cell only"
        is_reference, ref_clone, init_labeling = self._estimate_reference_cells()
        params = self.model_params
        tumor_pi = params["pi"][1:]
        params["pi"] = tumor_pi / tumor_pi.sum()
        self._init_lambda(is_reference, ref_clone)

        # Hard-EM init: dispersions at max bound (BB->Binomial, NB->Poisson)
        for assay in self.assay_types:
            count_data = self.count_data[assay]
            lg = params[f"{assay}-lambda"]
            tau_init = invphi_init = None
            if fit_mode in {"allele", "allele_total"}:
                params[f"{assay}-tau"] = self.tau_bounds[1]
                tau_init = float(self.tau_bounds[1])
            if fit_mode in {"total", "allele_total"}:
                params[f"{assay}-inv_phi"] = self.invphi_bounds[1]
                invphi_init = float(self.invphi_bounds[1])
            params[f"{assay}-theta"] = estimate_tumor_proportion(
                count_data, self.T[assay], lg, tau_init, invphi_init, fit_mode=fit_mode
            )
            # Precompute: rdrs only for tumor clones (exclude normal column)
            rdrs_full = clone_rdr_gk(lg, count_data.cn_C)
            params[f"{assay}-rdrs"] = rdrs_full[:, 1:]  # (G, K_tumor)
            params[f"{assay}-allele_mask"] = count_data.allele_mask[
                self.allele_mask_id
            ] & (lg > 0)
            params[f"{assay}-total_mask"] = count_data.total_mask[
                self.total_mask_id
            ] & (lg > 0)

        return init_labeling

    def compute_log_likelihood(
        self, fit_mode: str
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute log-likelihood over tumor clones only (K_tumor components)."""
        params = self.model_params
        global_lls = params["ll_global"]
        global_lls[:] = 0.0

        for assay in self.assay_types:
            count_data = self.count_data[assay]
            theta = params[f"{assay}-theta"]
            rdrs = params[f"{assay}-rdrs"]  # (G, K_tumor)
            allele_mask = params[f"{assay}-allele_mask"]
            total_mask = params[f"{assay}-total_mask"]
            ll_a = params[f"{assay}-ll_allele"]
            ll_t = params[f"{assay}-ll_total"]
            ll_a[:] = 0.0
            ll_t[:] = 0.0

            if fit_mode in {"allele", "allele_total"} and allele_mask.any():
                ll_a[allele_mask] = cond_betabin_logpmf_theta(
                    count_data.count_B[allele_mask],
                    count_data.count_C[allele_mask],
                    params[f"{assay}-tau"],
                    count_data.cn_BAF[allele_mask][:, 1:],  # tumor clone BAFs only
                    rdrs[allele_mask],
                    theta,
                )
                global_lls += ll_a.sum(axis=0)

            if fit_mode in {"total", "allele_total"} and total_mask.any():
                ll_t[total_mask] = cond_negbin_logpmf_theta(
                    count_data.count_X[total_mask],
                    self.T[assay],
                    params[f"{assay}-lambda"][total_mask],
                    params[f"{assay}-inv_phi"],
                    rdrs[total_mask],
                    theta,
                )
                global_lls += ll_t.sum(axis=0)

        # global pi prior: log pi[k] added per spot
        global_lls += np.log(np.maximum(params["pi"], 1e-30))[None, :]

        log_marg = logsumexp(global_lls, axis=1)
        return np.sum(log_marg), log_marg, global_lls

    def _m_step(self, fit_mode: str, gamma: np.ndarray, t: int = 0):
        # pi simplex update; theta fixed after init (_e_step inherited from Base_Model)
        self._update_pi(gamma, self.num_barcodes, self.num_em_clones)

    def _map_estimation(
        self, gamma: np.ndarray, label: str, as_df: bool = True
    ) -> pd.DataFrame | dict:
        """MAP over tumor clones only — labels are always a tumor clone.

        Normal vs tumor is NOT decided here; ``tumor_purity`` (θ) is reported and
        the purity-cutoff sweep in ``analysis/validate.py`` relabels low-purity
        spots as "normal" downstream.
        """
        N = len(gamma)
        clone_names = np.array(self.tumor_clones)
        map_k = gamma.argmax(axis=1)
        labels = clone_names[map_k]
        max_post = gamma[np.arange(N), map_k]

        if not as_df:
            return {"labels": labels, "max_posterior": max_post}

        anns = self.barcodes.copy(deep=True)
        anns.loc[:, self.tumor_clones] = gamma
        anns["max_posterior"] = max_post
        anns[label] = labels
        return anns

    def predict(self, fit_mode: str, label: str, **kwargs) -> tuple[pd.DataFrame, dict]:
        """Predict clone labels via MAP. Purity reported but does not affect labels.

        Clone MAP: z_n = argmax_k gamma_nk (over tumor clones).
        """
        gamma = self._e_step(fit_mode)
        theta = self.model_params[f"{self.assay_types[0]}-theta"]

        clone_names = np.array(self.tumor_clones)
        map_k = gamma.argmax(axis=1)
        labels = clone_names[map_k]
        max_post = gamma[np.arange(self.num_barcodes), map_k]

        anns = self.barcodes.copy(deep=True)
        anns.loc[:, self.tumor_clones] = gamma
        anns["max_posterior"] = max_post
        anns["tumor_purity"] = theta
        anns[label] = labels

        clone_props = {c: np.mean(anns[label].to_numpy() == c) for c in self.clones}
        self._log_posterior_stats(anns, label)
        return anns, clone_props
