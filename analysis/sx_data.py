import logging

import numpy as np
import pandas as pd

from scipy import sparse

from copytyping.inference.count_data import get_cnp_mask
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
    ):
        """Construct from pre-loaded segment-level data.

        Args:
            barcodes_df: DataFrame with BARCODE, REP_ID columns.
            seg_df: Segment DataFrame with CNP, PROPS columns.
            X, Y, D: Dense int32 count matrices (G, N).
            baf_clip: cn_BAF clipping parameter.
        """
        self.barcodes = barcodes_df
        self.N = len(self.barcodes)

        self.cnv_blocks = seg_df
        self.X = X
        self.Y = Y
        self.D = D

        self.clones, self.cn_A, self.cn_B, self.cn_C, self.cn_BAF = parse_cnv_profile(
            self.cnv_blocks, baf_clip=baf_clip
        )
        self.G = len(self.cnv_blocks)
        self.K = len(self.clones)
        self.MASK = get_cnp_mask(self.cn_A, self.cn_B, self.cn_C)
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

    def subset_by_rep(self, rep_id):
        """Return (new SX_Data, mask) restricted to barcodes where REP_ID == rep_id.

        Reuses G-axis attributes (cnv_blocks, cn_A/cn_B/cn_C/cn_BAF/MASK, etc.) and only
        recomputes N-axis attributes (X/Y/D/T/barcodes/N).
        """
        mask = (self.barcodes["REP_ID"] == rep_id).to_numpy()
        new = SX_Data.__new__(SX_Data)
        new.barcodes = self.barcodes[mask].reset_index(drop=True)
        new.N = int(mask.sum())
        new.cnv_blocks = self.cnv_blocks
        new.X = self.X[:, mask]
        new.Y = self.Y[:, mask]
        new.D = self.D[:, mask]
        new.clones = self.clones
        new.cn_A = self.cn_A
        new.cn_B = self.cn_B
        new.cn_C = self.cn_C
        new.cn_BAF = self.cn_BAF
        new.G = self.G
        new.K = self.K
        new.MASK = self.MASK
        new.nrows_imbalanced = self.nrows_imbalanced
        new.nrows_aneuploid = self.nrows_aneuploid
        new.T = np.sum(new.X, axis=0)
        return new, mask

    def apply_mask_shallow(self, mask_id="CNP", additional_mask=None):
        if additional_mask is None:
            additional_mask = np.ones(self.G, dtype=bool)

        mask = self.MASK[mask_id] & additional_mask
        M = {
            "cn_A": self.cn_A[mask, :],
            "cn_B": self.cn_B[mask, :],
            "cn_C": self.cn_C[mask, :],
            "cn_BAF": self.cn_BAF[mask, :],
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
            A = sparse.eye(N_rep, format="csr") + sparse.csr_matrix(W > 0).astype(
                np.int8
            )

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
        k_counts = {
            k: int((spot_k == k).sum()) for k in range(max_k + 1) if (spot_k == k).any()
        }
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
            SimpleNamespace with attributes: X, Y, D, T, A, B, C, cn_BAF, MASK,
            G, N, K, clones, nrows_imbalanced, nrows_aneuploid, cluster_ids,
            and a bound ``apply_mask_shallow(mask_id, additional_mask)`` method.
        """
        from types import SimpleNamespace

        cnp_keys = []
        for g in range(self.G):
            a_row = self.cn_A[g].tolist()
            b_row = self.cn_B[g].tolist()
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
        cn_A_c = self.cn_A[first_members]
        cn_B_c = self.cn_B[first_members]
        cn_C_c = self.cn_C[first_members]
        cn_BAF_c = self.cn_BAF[first_members]
        MASK_c = get_cnp_mask(cn_A_c, cn_B_c, cn_C_c)

        clust = SimpleNamespace(
            X=X_c,
            Y=Y_c,
            D=D_c,
            T=self.T,
            cn_A=cn_A_c,
            cn_B=cn_B_c,
            cn_C=cn_C_c,
            cn_BAF=cn_BAF_c,
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
                "cn_A": cn_A_c[mask],
                "cn_B": cn_B_c[mask],
                "cn_C": cn_C_c[mask],
                "cn_BAF": cn_BAF_c[mask],
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


def adaptive_bin_bbc(
    bbc_df,
    X_bbc,
    Y_bbc,
    D_bbc,
    seg_sx,
    min_snp_count=300,
    max_bin_length=5_000_000,
):
    """Merge adjacent BBC bins within the same segment to reduce sparsity.

    Walks BBC bins in genomic order per chromosome, grouping consecutive bins
    that share the same seg_id until the pseudobulk SNP count reaches
    min_snp_count or the combined length exceeds max_bin_length.

    Args:
        bbc_df: BBC-level DataFrame with #CHR, START, END, seg_id, CNP columns.
        X_bbc, Y_bbc, D_bbc: (G_bbc, N) sparse or dense count matrices.
        seg_sx: segment-level SX_Data (for barcodes).
        min_snp_count: minimum pseudobulk D sum per merged bin.
        max_bin_length: maximum merged bin length in bp.

    Returns:
        SX_Data with aggregated bins.
    """
    bbc_df = bbc_df.reset_index(drop=True)

    # Keep X/Y/D sparse (CSR) — avoids ~30 GB densification of 66K × 38K matrices.
    # Per-group sum on sparse row slices is cheap; final aggregated matrix is small.
    X = X_bbc.tocsr() if sparse.issparse(X_bbc) else sparse.csr_matrix(X_bbc)
    Y = Y_bbc.tocsr() if sparse.issparse(Y_bbc) else sparse.csr_matrix(Y_bbc)
    D = D_bbc.tocsr() if sparse.issparse(D_bbc) else sparse.csr_matrix(D_bbc)

    D_total = np.asarray(D.sum(axis=1)).ravel()
    # Pre-extract as numpy arrays so the walk loop avoids pandas iloc per access.
    bbc_chr = bbc_df["#CHR"].to_numpy()
    seg_ids = bbc_df["seg_id"].to_numpy()
    starts = bbc_df["START"].to_numpy()
    ends = bbc_df["END"].to_numpy()
    cnps = bbc_df["CNP"].to_numpy()

    # bbc_df is already chr-pos sorted (HATCHet writes it that way); preserve that
    # row order via pd.unique rather than np.unique (which would lex-sort).
    groups = []  # list of (chr, start, end, seg_id, cnp, [bbc_indices])
    for chrom in pd.unique(bbc_chr):
        chr_idx = np.where(bbc_chr == chrom)[0]
        if chr_idx.size == 0:
            continue
        order = chr_idx[np.argsort(starts[chr_idx])]

        cur_seg = seg_ids[order[0]]
        cur_start = starts[order[0]]
        cur_end = ends[order[0]]
        cur_cnp = cnps[order[0]]
        cur_indices = [order[0]]
        cur_d = D_total[order[0]]

        for i in range(1, len(order)):
            bi = order[i]
            bi_seg = seg_ids[bi]
            bi_start = starts[bi]
            bi_end = ends[bi]
            bi_length = bi_end - cur_start

            same_seg = bi_seg == cur_seg and cur_seg >= 0
            fits_length = bi_length <= max_bin_length
            needs_more = cur_d < min_snp_count

            if same_seg and fits_length and needs_more:
                cur_indices.append(bi)
                cur_end = bi_end
                cur_d += D_total[bi]
            else:
                groups.append(
                    (chrom, cur_start, cur_end, cur_seg, cur_cnp, cur_indices)
                )
                cur_seg = bi_seg
                cur_start = bi_start
                cur_end = bi_end
                cur_cnp = cnps[bi]
                cur_indices = [bi]
                cur_d = D_total[bi]

        groups.append((chrom, cur_start, cur_end, cur_seg, cur_cnp, cur_indices))

    # build aggregated arrays — sum sparse rows per group, then densify the result
    n_agg = len(groups)
    N = X.shape[1]
    X_agg = np.zeros((n_agg, N), dtype=np.int32)
    Y_agg = np.zeros((n_agg, N), dtype=np.int32)
    D_agg = np.zeros((n_agg, N), dtype=np.int32)
    rows = []

    for gi, (chrom, start, end, sid, cnp, indices) in enumerate(groups):
        idx = np.asarray(indices)
        X_agg[gi] = np.asarray(X[idx].sum(axis=0)).ravel()
        Y_agg[gi] = np.asarray(Y[idx].sum(axis=0)).ravel()
        D_agg[gi] = np.asarray(D[idx].sum(axis=0)).ravel()
        rows.append(
            {"#CHR": chrom, "START": start, "END": end, "seg_id": sid, "CNP": cnp}
        )

    agg_df = pd.DataFrame(rows)

    # Drop unmapped bins (seg_id=-1, no CNP)
    mapped = agg_df["seg_id"] >= 0
    if not mapped.all():
        n_drop = (~mapped).sum()
        logging.warning(f"adaptive_bin_bbc: dropping {n_drop} unmapped bins")
        keep = mapped.to_numpy()
        agg_df = agg_df[keep].reset_index(drop=True)
        X_agg = X_agg[keep]
        Y_agg = Y_agg[keep]
        D_agg = D_agg[keep]

    lengths = (agg_df["END"] - agg_df["START"]).to_numpy()
    d_sums = D_agg.sum(axis=1)
    x_sums = X_agg.sum(axis=1)
    logging.info(
        f"adaptive_bin_bbc: {len(bbc_df)} -> {n_agg} bins, "
        f"length median={np.median(lengths):.0f} mean={np.mean(lengths):.0f}, "
        f"snp_count median={np.median(d_sums):.0f} mean={np.mean(d_sums):.0f}, "
        f"total_count median={np.median(x_sums):.0f} mean={np.mean(x_sums):.0f}"
    )

    return SX_Data(seg_sx.barcodes, agg_df, X_agg, Y_agg, D_agg)
