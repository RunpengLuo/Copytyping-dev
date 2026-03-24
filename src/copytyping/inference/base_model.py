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

    def save_param_trace(self, param_trace: list):
        if not param_trace or not self.work_dir:
            return
        for key in param_trace[0]:
            vals = [p[key] for p in param_trace]
            if isinstance(vals[0], np.ndarray):
                df = pd.DataFrame(vals)
            else:
                df = pd.DataFrame({"value": vals})
            df.index.name = "iter"
            out_path = os.path.join(self.work_dir, f"{self.prefix}.trace.{key}.tsv")
            df.to_csv(out_path, sep="\t")
        logging.info(f"saved {len(param_trace[0])} parameter traces to {self.work_dir}")

    def predict(
        self,
        fit_mode: str,
        params: dict,
        label: str,
        posterior_thres: float = 0.5,  # minimum required max posterior value
        margin_thres: float = 0.1,  # minimum gap between 1st and 2nd highest MAP value
        tumorprop_threshold: float = 0.5,  # for spatial: set spots to normal if purity < threshold
    ):
        logging.info("Decode labels with MAP estimation")
        posteriors = self._e_step(fit_mode, params)
        anns = self.barcodes.copy(deep=True)

        if self.assay_type in SPATIAL_ASSAYS:
            # posteriors are over tumor clones only; normal is modeled via 1-theta
            tumor_clones = self.clones[1:]
            anns.loc[:, tumor_clones] = posteriors

            theta_key = f"{self.data_types[0]}-theta"
            anns["tumor_purity"] = params[theta_key]

            probs = anns[tumor_clones].to_numpy()
            probs_sorted = np.sort(probs, axis=1)
            anns["max_posterior"] = probs_sorted[:, -1]
            if probs_sorted.shape[1] > 1:
                anns["margin_delta"] = probs_sorted[:, -1] - probs_sorted[:, -2]
            else:
                anns["margin_delta"] = 1.0

            anns[label] = anns[tumor_clones].idxmax(axis=1)

            mask_normal = anns["tumor_purity"] < tumorprop_threshold
            anns.loc[mask_normal, label] = "normal"
            anns.loc[mask_normal, "max_posterior"] = 0.0
            anns.loc[mask_normal, "margin_delta"] = 0.0
            logging.info(
                f"set {mask_normal.sum()} spots to normal "
                f"(purity < {tumorprop_threshold})"
            )
        else:
            anns.loc[:, self.clones] = posteriors
            anns["tumor"] = 1 - anns["normal"]

            probs = anns[self.clones].to_numpy()
            probs_sorted = np.sort(probs, axis=1)
            anns["max_posterior"] = probs_sorted[:, -1]
            anns["margin_delta"] = probs_sorted[:, -1] - probs_sorted[:, -2]

            anns[label] = anns[self.clones].idxmax(axis=1)

            # reject ambiguous assignments below confidence thresholds
            mask_na = (anns["max_posterior"] < posterior_thres) | (
                anns["margin_delta"] < margin_thres
            )
            anns.loc[mask_na, label] = "NA"

        clone_props = {
            clone: np.mean(anns[label].to_numpy() == clone) for clone in self.clones
        }
        return anns, clone_props
