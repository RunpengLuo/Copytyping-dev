import os
import sys

import numpy as np
import pandas as pd

from copytyping.utils import *
from copytyping.io_utils import *


class SX_Data:
    def __init__(
        self,
        nbarcodes: int,
        mod_dir: str,
        data_type: str,
        allele_mask=None,
        total_mask=None,
        verbose=1,
    ) -> None:
        assert os.path.isdir(mod_dir)
        self.num_barcodes = nbarcodes

        ##################################################
        # allele bin inputs
        self.bin_info = pd.read_table(
            os.path.join(mod_dir, "allele_bins.tsv"), sep="\t"
        )
        self.num_bins = len(self.bin_info)
        # load allele matrix
        clones, allele_A, allele_B, allele_C, allele_BAF = parse_allele_cnp(
            self.bin_info, laplace=0.01
        )
        self.clones = clones
        self.K = self.num_clones = len(clones)
        self.allele_A = allele_A
        self.allele_B = allele_B
        self.allele_C = allele_C
        self.allele_BAF = allele_BAF

        # TODO filter BAF close to 0.5.
        # allele_mask = (self.bin_info["BAF"] < 0.45) & (self.bin_info["D"] >= 30)
        self.ALLELE_MASK = get_cnp_mask(
            allele_A, allele_B, allele_C, and_mask=allele_mask
        )
        self.nrows_eff_allele = np.sum(self.ALLELE_MASK["IMBALANCED"])
        self.allele_hb2index = {
            k: np.array(v)
            for k, v in self.bin_info.groupby("HB", sort=False).groups.items()
        }
        a_allele_mat, b_allele_mat, t_allele_mat = load_allele_input(mod_dir)
        assert a_allele_mat.shape == (self.num_bins, self.num_barcodes), (
            a_allele_mat.shape
        )
        assert b_allele_mat.shape == (self.num_bins, self.num_barcodes), (
            b_allele_mat.shape
        )
        assert t_allele_mat.shape == (self.num_bins, self.num_barcodes), (
            t_allele_mat.shape
        )
        self.X = a_allele_mat
        self.Y = b_allele_mat
        self.D = t_allele_mat

        ##################################################
        # feature bin inputs
        self.feat_info = pd.read_table(
            os.path.join(mod_dir, "feature_bins.tsv"), sep="\t"
        )
        self.num_features = len(self.feat_info)

        _, feat_A, feat_B, feat_C = parse_total_cnp(self.feat_info)
        self.feat_A = feat_A
        self.feat_B = feat_B
        self.feat_C = feat_C

        self.FEAT_MASK = get_cnp_mask(feat_A, feat_B, feat_C, and_mask=total_mask)
        self.nrows_eff_feat = np.sum(self.FEAT_MASK["ANEUPLOID"])
        self.feat_hb2index = {
            k: np.array(v)
            for k, v in self.feat_info.groupby("HB", sort=False).groups.items()
        }

        count_mat = load_count_input(mod_dir)
        assert count_mat.shape == (self.num_features, self.num_barcodes), (
            count_mat.shape
        )
        self.T = count_mat
        self.Tn = np.sum(count_mat, axis=0)

        if verbose:
            print(
                f"{data_type} data is loaded #features={self.num_features} #allele bins={self.num_bins}"
            )
            print(f"#effective imbalanced CNA bins={self.nrows_eff_allele}")
            print(f"#effective CNA features={self.nrows_eff_feat}")
        return

    def apply_allele_mask_shallow(self, mask_id="CNP"):
        cnp_mask = self.ALLELE_MASK[mask_id]
        M = {
            "A": self.allele_A[cnp_mask, :],
            "B": self.allele_B[cnp_mask, :],
            "C": self.allele_C[cnp_mask, :],
            "BAF": self.allele_BAF[cnp_mask, :],
            "X": self.X[cnp_mask, :],
            "Y": self.Y[cnp_mask, :],
            "D": self.D[cnp_mask, :],
            "bin_info": self.bin_info[cnp_mask],
        }
        return M

    def apply_feat_mask_shallow(self, mask_id="CNP"):
        cnp_mask = self.FEAT_MASK[mask_id]
        M = {
            "A": self.feat_A[cnp_mask, :],
            "B": self.feat_B[cnp_mask, :],
            "C": self.feat_C[cnp_mask, :],
            "T": self.T[cnp_mask, :],
            "feat_info": self.feat_info[cnp_mask],
        }
        return M

    def subset_allele_matrix(self, cnp_ids: np.ndarray):
        M = {
            "A": self.allele_A[cnp_ids, :],
            "B": self.allele_B[cnp_ids, :],
            "C": self.allele_C[cnp_ids, :],
            "BAF": self.allele_BAF[cnp_ids, :],
            "X": self.X[cnp_ids, :],
            "Y": self.Y[cnp_ids, :],
            "D": self.D[cnp_ids, :],
        }
        return M

    def subset_feat_matrix(self, cnp_ids: np.ndarray):
        M = {
            "A": self.feat_A[cnp_ids, :],
            "B": self.feat_B[cnp_ids, :],
            "C": self.feat_C[cnp_ids, :],
            "T": self.T[cnp_ids, :],
        }
        return M

    # def get_cnp_shared_ids(self, x_info, masks, apply_cnp_mask=True, mask_id="CNP"):
    #     """per CNP group, get bins index"""
    #     assert "CNP_ID" in x_info
    #     if apply_cnp_mask:
    #         cnp_mask = masks[mask_id]
    #         cnp_groups = x_info.loc[cnp_mask, :].groupby("CNP_ID", sort=False).groups
    #     else:
    #         cnp_groups = x_info.groupby("CNP_ID", sort=False).groups

    #     cnp_groups = {k: np.array(v) for k, v in cnp_groups.items()}
    #     return cnp_groups


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
