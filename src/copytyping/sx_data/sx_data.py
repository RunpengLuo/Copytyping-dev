import logging

import numpy as np
import pandas as pd

from copytyping.io_utils import parse_cnv_profile


class SX_Data:
    """Container for segment-level (or cluster-level) count data and CNV profiles."""

    def __init__(
        self,
        barcodes_df: pd.DataFrame,
        seg_df: pd.DataFrame,
        X: np.ndarray,
        Y: np.ndarray,
        D: np.ndarray,
        laplace=0.01,
        verbose=1,
    ) -> None:
        """Construct from pre-loaded segment-level data.

        Args:
            barcodes_df: DataFrame with BARCODE, REP_ID columns.
            seg_df: Segment DataFrame with CNP, PROPS columns.
            X, Y, D: Dense int32 count matrices (G, N).
            laplace: BAF clipping parameter.
        """
        self.barcodes = barcodes_df
        self.N = len(self.barcodes)

        self.cnv_blocks = seg_df
        self.X = X
        self.Y = Y
        self.D = D

        self.clones, self.A, self.B, self.C, self.BAF = parse_cnv_profile(
            self.cnv_blocks, laplace=laplace
        )
        self.G = len(self.cnv_blocks)
        self.K = len(self.clones)
        self.MASK = get_cnp_mask(self.A, self.B, self.C)
        self.nrows_imbalanced = int(np.sum(self.MASK["IMBALANCED"]))
        self.nrows_aneuploid = int(np.sum(self.MASK["ANEUPLOID"]))

        assert self.X.shape == (self.G, self.N), (
            f"X shape {self.X.shape} != ({self.G}, {self.N})"
        )
        assert self.Y.shape == (self.G, self.N)
        assert self.D.shape == (self.G, self.N)

        self.T = np.sum(self.X, axis=0)

        logging.debug(f"data loaded #cells={self.N}, #bins={self.G}")
        logging.debug(f"#effective imbalanced CNA bins={self.nrows_imbalanced}")
        logging.debug(f"#effective aneuploid CNA bins={self.nrows_aneuploid}")

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

    def to_cluster_level(self):
        """Return a new SX_Data-like object at CNP-cluster level.

        Merges segments with identical (A, B) copy-number profiles: X/Y/D
        counts are summed. The returned object has the same interface as
        SX_Data (attributes + apply_mask_shallow) but fewer rows (G_clust).

        An additional ``cluster_ids`` attribute (shape G_seg) maps each
        original segment index to its cluster index.
        """
        clust = self.aggregate_to_cnp_clusters()
        return clust

    def aggregate_to_cnp_clusters(self):
        """Group segments by identical (A, B) copy-number pattern across all clones.

        Segments that share the same per-clone (A, B) vector are merged: their
        X/Y/D count matrices are summed across the merged rows. The resulting
        namespace mirrors the SX_Data attribute layout so that model code
        (``compute_log_likelihood``, ``_m_step``) can use it interchangeably.

        The returned namespace does **not** carry a ``cnv_blocks`` attribute and
        its ``apply_mask_shallow`` closure does not return a ``"cnv_blocks"`` key.
        Callers that need genomic coordinates must retain the original SX_Data.

        Returns:
            SimpleNamespace with attributes: X, Y, D, T, A, B, C, BAF, MASK,
            G, N, K, clones, nrows_imbalanced, nrows_aneuploid, cluster_ids,
            and a bound ``apply_mask_shallow(mask_id, additional_mask)`` method.
        """
        from types import SimpleNamespace

        cnp_keys = []
        for g in range(self.G):
            a_row = self.A[g].tolist()
            b_row = self.B[g].tolist()
            cnp_keys.append(tuple(a_row + b_row))

        unique_keys = list(dict.fromkeys(cnp_keys))
        key_to_cid = {key: cid for cid, key in enumerate(unique_keys)}
        cluster_ids = np.array([key_to_cid[key] for key in cnp_keys])
        G_c = len(unique_keys)

        X_c = np.zeros((G_c, self.N), dtype=self.X.dtype)
        Y_c = np.zeros((G_c, self.N), dtype=self.Y.dtype)
        D_c = np.zeros((G_c, self.N), dtype=self.D.dtype)
        for cid in range(G_c):
            members = np.where(cluster_ids == cid)[0]
            X_c[cid] = self.X[members].sum(axis=0)
            Y_c[cid] = self.Y[members].sum(axis=0)
            D_c[cid] = self.D[members].sum(axis=0)

        first_members = [np.where(cluster_ids == cid)[0][0] for cid in range(G_c)]
        A_c = self.A[first_members]
        B_c = self.B[first_members]
        C_c = self.C[first_members]
        BAF_c = self.BAF[first_members]
        MASK_c = get_cnp_mask(A_c, B_c, C_c)

        clust = SimpleNamespace(
            X=X_c,
            Y=Y_c,
            D=D_c,
            T=self.T,
            A=A_c,
            B=B_c,
            C=C_c,
            BAF=BAF_c,
            MASK=MASK_c,
            G=G_c,
            N=self.N,
            K=self.K,
            clones=self.clones,
            nrows_imbalanced=int(MASK_c["IMBALANCED"].sum()),
            nrows_aneuploid=int(MASK_c["ANEUPLOID"].sum()),
            cluster_ids=cluster_ids,
            barcodes=self.barcodes,
        )

        def _apply_mask_shallow(mask_id="CNP", additional_mask=None):
            if additional_mask is None:
                additional_mask = np.ones(G_c, dtype=bool)
            mask = MASK_c[mask_id] & additional_mask
            M = {
                "A": A_c[mask],
                "B": B_c[mask],
                "C": C_c[mask],
                "BAF": BAF_c[mask],
                "X": X_c[mask],
                "Y": Y_c[mask],
                "D": D_c[mask],
            }
            return M, mask

        clust.apply_mask_shallow = _apply_mask_shallow

        logging.info(
            f"aggregated {self.G} segments -> {G_c} CNP clusters, "
            f"#imbalanced={clust.nrows_imbalanced}, #aneuploid={clust.nrows_aneuploid}"
        )
        return clust


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

    if A.shape[1] > 2:
        subclonal_mask = np.any(A[:, 2:] != A[:, 1][:, None], axis=1) | np.any(
            B[:, 2:] != B[:, 1][:, None], axis=1
        )
    else:
        # single tumor clone: nothing is subclonal
        subclonal_mask = np.zeros(A.shape[0], dtype=bool)
    if and_mask is not None:
        tumor_mask &= and_mask
        clonal_loh_mask &= and_mask
        subclonal_loh_mask &= and_mask
        ai_mask &= and_mask
        subclonal_mask &= and_mask
    return {
        "CNP": tumor_mask,
        "IMBALANCED": ai_mask,
        "CLONAL_IMBALANCED": ai_mask & ~subclonal_mask,
        "ANEUPLOID": c_mask,
        "SUBCLONAL": subclonal_mask,
        "CLONAL_LOH": clonal_loh_mask,
        "SUBCLONAL_LOH": subclonal_loh_mask,
        "NEUTRAL": neutral_mask,
    }
