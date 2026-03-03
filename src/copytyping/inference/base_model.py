import logging

import numpy as np
import pandas as pd

from copytyping.utils import *
from copytyping.sx_data.sx_data import *
from copytyping.inference.model_utils import *
from copytyping.inference.likelihood_funcs import *

allowed_fit_mode = {"hybrid", "allele_only", "total_only"}


class Base_Model:
    def __init__(
        self,
        barcodes: pd.DataFrame,
        assay_type: str,
        data_types: list,
        data_sources: dict[str, SX_Data],
        work_dir=None,
        prefix="copytyping",
        verbose=1,
    ) -> None:
        self.barcodes = barcodes
        self.data_types = data_types
        self.assay_type = assay_type
        self.data_sources = data_sources
        self.clones = self.data_sources[self.data_types[0]].clones
        self.N = self.num_barcodes = len(barcodes)
        self.K = self.num_clones = len(self.clones)
        self.work_dir = work_dir
        self.prefix = prefix
        self.verbose = verbose
        return

    def _init_base_params(
        self,
        fit_mode: str,
        init_params: dict,
    ):
        params = {}

        params["pi"] = init_params.get("pi", None)
        if params.get("pi", None) is None:
            params["pi"] = np.ones(self.K) / self.K

        ##################################################
        # initialize NB dispersions
        if fit_mode in {"total_only", "hybrid"}:
            for data_type in self.data_types:
                if params.get(f"{data_type}-inv_phi", None) is None:
                    params[f"{data_type}-inv_phi"] = np.full(
                        self.data_sources[data_type].nrows_aneuploid,
                        fill_value=1 / init_params["phi0"],
                        dtype=np.float32,
                    )

        ##################################################
        # initialize BB dispersions
        if fit_mode in {"allele_only", "hybrid"}:
            for data_type in self.data_types:
                if params.get(f"{data_type}-tau", None) is None:
                    params[f"{data_type}-tau"] = np.full(
                        self.data_sources[data_type].nrows_imbalanced,
                        fill_value=init_params["tau0"],
                        dtype=np.float32,
                    )
        return params

    def compute_log_likelihood(self, fit_mode: str, params: dict):
        raise NotImplementedError("not implemented")

    def _e_step(self, fit_mode: str, params: dict, t=0) -> np.ndarray:
        """compute posterior probabilities per cell per clone. (N, K)"""
        ll, log_marg, global_lls = self.compute_log_likelihood(fit_mode, params)

        # normalize by softmax
        gamma = np.exp(global_lls - logsumexp(global_lls, axis=1, keepdims=True))
        return gamma

    def print_params(self, params: dict, fit_mode="hybrid"):
        for param_key, param_val in params.items():
            logging.info(f"{param_key}: {param_val}")
        return

    def predict(
        self,
        fit_mode: str,
        params: dict,
        label: str,
        posterior_thres: float = 0.5,  # minimum required max posterior value
        margin_thres: float = 0.1,  # minimum gap between 1st and 2nd highest MAP value
    ):
        logging.info("Decode labels with MAP estimation")
        posteriors = self._e_step(fit_mode, params)  # (N, K)
        anns = self.barcodes.copy(deep=True)
        anns.loc[:, self.clones] = posteriors

        # posterior probs being tumor cell.
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
