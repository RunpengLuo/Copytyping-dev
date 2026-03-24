import os
import sys
import logging

import numpy as np
import pandas as pd

from copytyping.utils import *
from copytyping.io_utils import *


class SX_Data:
    def __init__(
        self,
        bc_file: str,
        cnp_file: str,
        x_count_file: str,
        y_count_file: str,
        d_count_file: str,
        data_type: str,
        laplace=0.01,
        verbose=1,
    ) -> None:
        self.data_type = data_type
        self.barcodes = pd.read_table(
            bc_file, sep="\t", header=None, names=["BARCODE"], dtype=str
        )
        # Parse REP_ID from barcode suffix (format: {barcode}_{rep_id})
        self.barcodes["REP_ID"] = self.barcodes["BARCODE"].str.rsplit("_", n=1).str[-1]
        self.cnv_blocks = pd.read_table(cnp_file, sep="\t")
        self.clones, self.A, self.B, self.C, self.BAF = parse_cnv_profile(
            self.cnv_blocks, laplace=laplace
        )
        self.N = len(self.barcodes)
        self.G = len(self.cnv_blocks)
        self.K = len(self.clones)
        self.MASK = get_cnp_mask(self.A, self.B, self.C)
        self.nrows_imbalanced = np.sum(self.MASK["IMBALANCED"])
        self.nrows_aneuploid = np.sum(self.MASK["ANEUPLOID"])

        # (G, N)
        self.X: np.ndarray = (
            sparse.load_npz(x_count_file).toarray().astype(dtype=np.int32)
        )
        self.Y: np.ndarray = (
            sparse.load_npz(y_count_file).toarray().astype(dtype=np.int32)
        )
        self.D: np.ndarray = (
            sparse.load_npz(d_count_file).toarray().astype(dtype=np.int32)
        )
        assert self.X.shape == (self.G, self.N)
        assert self.Y.shape == (self.G, self.N)
        assert self.D.shape == (self.G, self.N)

        self.T = np.sum(self.X, axis=0)

        if verbose:
            logging.info(f"{data_type} data is loaded #cells={self.N}, #bins={self.G}")
            logging.info(f"#effective imbalanced CNA bins={self.nrows_imbalanced}")
            logging.info(f"#effective aneuploid CNA bins={self.nrows_aneuploid}")
        return

    def apply_mask_shallow(self, mask_id="CNP", additional_mask=None):
        if additional_mask is None:
            additional_mask = np.ones(self.G, dtype=bool)

        mask = self.MASK[mask_id] & additional_mask
        M = {
            "A": self.A[mask, :],
            "B": self.B[mask, :],
            "C": self.C[mask, :],
            "BAF": self.BAF[mask, :],
            "X": self.X[mask, :],
            "Y": self.Y[mask, :],
            "D": self.D[mask, :],
            "cnv_blocks": self.cnv_blocks[mask],
        }
        return M, mask


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
    clonal_loh_mask |= np.all(A[:, 1:] == 0, axis=1) & np.all(B[:, 1:] > 0, axis=1)

    subclonal_loh_mask = np.any(B[:, 1:] == 0, axis=1) & np.all(A[:, 1:] > 0, axis=1)
    subclonal_loh_mask |= np.any(A[:, 1:] == 0, axis=1) & np.all(B[:, 1:] > 0, axis=1)

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
