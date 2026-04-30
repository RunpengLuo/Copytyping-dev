import logging

import numpy as np
import pandas as pd

from scipy import sparse

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
        baf_clip=0.01,
        verbose=1,
    ) -> None:
        """Construct from pre-loaded segment-level data.

        Args:
            barcodes_df: DataFrame with BARCODE, REP_ID columns.
            seg_df: Segment DataFrame with CNP, PROPS columns.
            X, Y, D: Dense int32 count matrices (G, N).
            baf_clip: BAF clipping parameter.
        """
        self.barcodes = barcodes_df
        self.N = len(self.barcodes)

        self.cnv_blocks = seg_df
        self.X = X
        self.Y = Y
        self.D = D

        self.clones, self.A, self.B, self.C, self.BAF = parse_cnv_profile(
            self.cnv_blocks, baf_clip=baf_clip
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

    def apply_adaptive_smoothing(self, W_per_rep, max_k, min_umi, min_snp_umi):
        r"""Adaptive per-spot spatial smoothing, in-place.

        For each spot, progressively smooth k=1,2,...,max_k until per-spot
        total UMI >= min_umi AND total allele >= min_snp_umi, or max_k reached.
        Spots already meeting thresholds are not smoothed.

        Args:
            W_per_rep: dict[rep_id -> {"BARCODE": array, "W": sparse}].
            max_k: maximum smoothing level.
            min_umi: minimum total UMI threshold per spot.
            min_snp_umi: minimum total allele (D) threshold per spot.
        """
        if max_k <= 0:
            return
        T_before = self.T.copy()
        D_before = self.D.sum(axis=0).copy()
        bc_arr = self.barcodes["BARCODE"].values
        spot_k = np.zeros(self.N, dtype=int)  # smoothing level per spot

        for rep_id, sg in W_per_rep.items():
            W = sg["W"]
            sg_bcs = sg["BARCODE"]
            N_rep = len(sg_bcs)
            A = sparse.eye(N_rep, format="csr") + sparse.csr_matrix(W > 0).astype(np.int8)

            bc_set = set(sg_bcs)
            col_idx = np.array([i for i, bc in enumerate(bc_arr) if bc in bc_set])
            if len(col_idx) != N_rep:
                continue

            # Precompute reachability matrices for k=1..max_k
            reach_list = []
            reach = A
            for kk in range(max_k):
                if kk > 0:
                    reach = reach @ A
                reach_list.append((reach > 0).astype(np.int8))

            # Original counts for this rep
            X_orig = self.X[:, col_idx].copy()
            Y_orig = self.Y[:, col_idx].copy()
            D_orig = self.D[:, col_idx].copy()

            for kk in range(max_k):
                # Check which spots still need smoothing
                cur_T = self.X[:, col_idx].sum(axis=0)
                cur_D = self.D[:, col_idx].sum(axis=0)
                # Flatten to 1D if matrix
                cur_T = np.asarray(cur_T).ravel()
                cur_D = np.asarray(cur_D).ravel()
                needs_smooth = (cur_T < min_umi) | (cur_D < min_snp_umi)
                if not needs_smooth.any():
                    break

                # Apply k=(kk+1) smoothing to spots that need it
                R = reach_list[kk]
                X_smooth = X_orig @ R.T
                Y_smooth = Y_orig @ R.T
                D_smooth = D_orig @ R.T

                for j in range(N_rep):
                    if needs_smooth[j]:
                        self.X[:, col_idx[j]] = np.asarray(X_smooth[:, j]).ravel()
                        self.Y[:, col_idx[j]] = np.asarray(Y_smooth[:, j]).ravel()
                        self.D[:, col_idx[j]] = np.asarray(D_smooth[:, j]).ravel()
                        spot_k[col_idx[j]] = kk + 1

        self.T = np.sum(self.X, axis=0)
        D_after = self.D.sum(axis=0)

        n_smoothed = (spot_k > 0).sum()
        smoothed = spot_k > 0
        k_counts = {k: int((spot_k == k).sum()) for k in range(max_k + 1) if (spot_k == k).any()}
        logging.info(
            f"adaptive smoothing (max_k={max_k}, min_umi={min_umi}, min_snp_umi={min_snp_umi}): "
            f"{n_smoothed}/{self.N} spots smoothed, k distribution: {k_counts}"
        )
        logging.info(
            f"  all spots: UMI median {int(np.median(T_before))}->{int(np.median(self.T))}, "
            f"allele median {int(np.median(D_before))}->{int(np.median(D_after))}"
        )
        if n_smoothed > 0:
            logging.info(
                f"  smoothed spots: UMI median {int(np.median(T_before[smoothed]))}->{int(np.median(self.T[smoothed]))}, "
                f"allele median {int(np.median(D_before[smoothed]))}->{int(np.median(D_after[smoothed]))}"
            )

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

        # build cluster-level cnv_blocks with SEGMENTS column
        clust_rows = []
        for cid in range(G_c):
            members = np.where(cluster_ids == cid)[0]
            segs = ";".join(
                f"{self.cnv_blocks['#CHR'].iloc[m]}:{self.cnv_blocks['START'].iloc[m]}-{self.cnv_blocks['END'].iloc[m]}"
                for m in members
            )
            row = {"cluster": cid, "SEGMENTS": segs}
            row["CNP"] = self.cnv_blocks["CNP"].iloc[members[0]]
            for col in ("#BBC", "#SNPS", "#gene", "LENGTH"):
                if col in self.cnv_blocks.columns:
                    row[col] = int(self.cnv_blocks[col].iloc[members].sum())
            clust_rows.append(row)
        clust.cnv_blocks = pd.DataFrame(clust_rows)

        x_zero = (clust.X == 0).sum()
        y_zero = (clust.Y == 0).sum()
        d_zero = (clust.D == 0).sum()
        total = clust.G * clust.N
        logging.info(
            f"aggregated {self.G} segments -> {G_c} CNP clusters, "
            f"#imbalanced={clust.nrows_imbalanced}, #aneuploid={clust.nrows_aneuploid}"
        )
        logging.info(
            f"cluster-level sparsity: X={x_zero}/{total} ({100 * x_zero / total:.1f}%), "
            f"Y={y_zero}/{total} ({100 * y_zero / total:.1f}%), "
            f"D={d_zero}/{total} ({100 * d_zero / total:.1f}%)"
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
