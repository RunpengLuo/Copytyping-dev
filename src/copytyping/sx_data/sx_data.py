import os
import sys

import numpy as np
import pandas as pd

from copytyping.utils import *
from copytyping.io_utils import *


class SX_Data:
    def __init__(
        self,
        barcodes: pd.DataFrame,
        haplo_blocks: pd.DataFrame,
        mod_dir: str,
        data_type: str,
        verbose=1,
    ) -> None:
        assert os.path.isdir(mod_dir)
        self.N = len(barcodes)
        self.G = len(haplo_blocks)

        self.barcodes = barcodes
        self.bin_info = haplo_blocks.copy(deep=True)
        self.clones, self.A, self.B, self.C, self.BAF = parse_cnv_profile(
            self.bin_info, laplace=0.01
        )
        self.K = len(self.clones)
        self.MASK = get_cnp_mask(self.A, self.B, self.C)
        self.nrows_eff_allele = np.sum(self.MASK["IMBALANCED"])
        self.nrows_eff_feat = np.sum(self.MASK["ANEUPLOID"])

        # (G, N)
        self.X: np.ndarray = (
            sparse.load_npz(os.path.join(mod_dir, "X_count.npz")).toarray().astype(dtype=np.int32)
        )
        self.Y: np.ndarray = (
            sparse.load_npz(os.path.join(mod_dir, "Y_count.npz")).toarray().astype(dtype=np.int32)
        )
        self.D: np.ndarray = (
            sparse.load_npz(os.path.join(mod_dir, "D_count.npz")).toarray().astype(dtype=np.int32)
        )
        assert self.X.shape == (self.G, self.N)
        assert self.Y.shape == (self.G, self.N)
        assert self.D.shape == (self.G, self.N)

        self.Tn = np.sum(self.X, axis=0)

        if verbose:
            print(f"{data_type} data is loaded #cells={self.N}, #bins={self.G}")
            print(f"#effective imbalanced CNA bins={self.nrows_eff_allele}")
            print(f"#effective CNA features={self.nrows_eff_feat}")
        return

    def apply_allele_mask_shallow(self, mask_id="CNP"):
        cnp_mask = self.MASK[mask_id]
        M = {
            "A": self.A[cnp_mask, :],
            "B": self.B[cnp_mask, :],
            "C": self.C[cnp_mask, :],
            "BAF": self.BAF[cnp_mask, :],
            "Y": self.Y[cnp_mask, :],
            "D": self.D[cnp_mask, :],
            "bin_info": self.bin_info[cnp_mask],
        }
        return M

    def apply_feat_mask_shallow(self, mask_id="CNP"):
        cnp_mask = self.MASK[mask_id]
        M = {
            "A": self.A[cnp_mask, :],
            "B": self.B[cnp_mask, :],
            "C": self.C[cnp_mask, :],
            "X": self.X[cnp_mask, :],
            "bin_info": self.bin_info[cnp_mask],
        }
        return M


def get_cnp_mask(A, B, C, and_mask=None):
    """return 1d mask, False if the bin should be discarded during modelling"""
    tumor_mask = np.any(A != 1, axis=1) | np.any(
        B != 1, axis=1
    )  # not purely normal cell
    ai_mask = np.any(A != B, axis=1)  # at least one clone is allelic imbalanced
    c_mask = np.any(C != 2, axis=1)  # at least one clone has total copy != 2
    tumor_mask = ai_mask | c_mask  # either allelic imbalanced or non-diploid
    neutral_mask = ~tumor_mask

    clonal_loh_mask = np.all(B[:, 1:] == 0, axis=1) & np.all(A[:, 1:] > 0, axis=1)
    subclonal_loh_mask = np.any(B[:, 1:] == 0, axis=1) & np.all(A[:, 1:] > 0, axis=1)
    subclonal_mask = np.copy(tumor_mask)
    if A.shape[1] > 2:
        subclonal_mask = np.any(A[:, 2:] != A[:, 1][:, None], axis=1) | np.any(
            B[:, 2:] != B[:, 1][:, None], axis=1
        )
    if not and_mask is None:
        tumor_mask &= and_mask
        clonal_loh_mask &= and_mask
        subclonal_loh_mask &= and_mask
        ai_mask &= and_mask
        subclonal_mask &= and_mask
    return {
        "CNP": tumor_mask,
        "IMBALANCED": ai_mask,
        "ANEUPLOID": c_mask,
        "SUBCLONAL": subclonal_mask,
        "CLONAL_LOH": clonal_loh_mask,
        "SUBCLONAL_LOH": subclonal_loh_mask,
        "NEUTRAL": neutral_mask,
    }
