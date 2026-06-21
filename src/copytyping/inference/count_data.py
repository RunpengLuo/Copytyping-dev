import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import sparse

from copytyping.io_utils import exclude_barcodes
from copytyping.inference.inference_utils import merge_celltype_into_barcodes
from copytyping.sx_data.sx_data import get_cnp_mask


@dataclass
class CountData:
    """Segment-by-cell count container with an optional per-row CN profile.

    Count matrices are ``(num_segment, num_cell)``: rows are genomic segments
    (aligned to ``coordinates``), columns are cells/spots (aligned to
    ``barcodes``). ``cn_A/cn_B/cn_C/cn_BAF`` are ``(num_segment, num_clone)``
    per-row copy-number profiles, populated by ``annotate_cnps`` (None until then).

    Attributes:
        barcodes: per-cell metadata, one row per matrix column (num_cell).
        coordinates: per-segment metadata, one row per matrix row (num_segment).
        X: read depth / total feature counts.
        A: A-allele counts.
        B: B-allele counts.
        cn_A/cn_B/cn_C/cn_BAF: per-row, per-clone copy numbers + B-allele freq.
        clones: clone names (``["normal", "clone1", ...]``), populated by
            ``annotate_cnps`` (None until then).
        allele_mask/total_mask: per-row boolean masks selecting informative rows
            for the allele (BAF) and total (RDR) likelihood. ``allele_mask`` keys
            ``IMBALANCED``/``CLONAL_IMBALANCED``; ``total_mask`` keys
            ``ANEUPLOID``/``SUBCLONAL``. Populated by ``segment_count_data``
            (None until then).
        num_segment: matrix row count (set on init).
        num_cell: matrix column count (set on init).
    """

    barcodes: pd.DataFrame
    coordinates: pd.DataFrame
    X: np.ndarray | sparse.csr_matrix
    A: np.ndarray | sparse.csr_matrix
    B: np.ndarray | sparse.csr_matrix
    cn_A: np.ndarray | None = None
    cn_B: np.ndarray | None = None
    cn_C: np.ndarray | None = None
    cn_BAF: np.ndarray | None = None
    clones: list[str] | None = None
    allele_mask: dict[str, np.ndarray] | None = None
    total_mask: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.num_segment, self.num_cell = self.X.shape
        shape = (self.num_segment, self.num_cell)
        assert self.A.shape == shape, f"A shape {self.A.shape} != {shape}"
        assert self.B.shape == shape, f"B shape {self.B.shape} != {shape}"
        assert len(self.barcodes) == self.num_cell, (
            f"barcodes rows ({len(self.barcodes)}) != num_cell ({self.num_cell})"
        )
        assert len(self.coordinates) == self.num_segment, (
            f"coordinates rows ({len(self.coordinates)}) != "
            f"num_segment ({self.num_segment})"
        )

    def to_dense(self) -> None:
        """Densify X/A/B in place (no-op for already-dense matrices)."""
        if sparse.issparse(self.X):
            self.X = np.asarray(self.X.todense())
        if sparse.issparse(self.A):
            self.A = np.asarray(self.A.todense())
        if sparse.issparse(self.B):
            self.B = np.asarray(self.B.todense())

    def annotate_cnps(
        self,
        bbc_phases: pd.DataFrame,
        seg_df: pd.DataFrame,
        clones: list[str],
        cn_A: np.ndarray,
        cn_B: np.ndarray,
        cn_C: np.ndarray,
        cn_BAF: np.ndarray,
    ) -> None:
        """Phase-correct A/B and attach the per-row CN profile (in place).

        ``bbc_phases`` (exact #CHR/START/END match) gives the per-row phase
        (1=keep, 0=swap A/B). The bulk profile arrays (``clones`` +
        ``cn_A/cn_B/cn_C/cn_BAF`` from ``load_bulk_cnprofile``, aligned to
        ``seg_df`` rows) are mapped to each row by midpoint containment.
        """
        coords = self.coordinates

        # phase per row from the (bbc-level) phasing table
        ph = coords.merge(
            bbc_phases[["#CHR", "START", "END", "PHASE"]],
            on=["#CHR", "START", "END"],
            how="left",
        )["PHASE"]
        assert ph.notna().all(), "some rows have no matching phase in bbc_phases"
        ph = ph.to_numpy()[:, None]  # 1=keep, 0=swap
        A, B = self.A, self.B
        if sparse.issparse(A):
            self.A = A.multiply(ph) + B.multiply(1 - ph)
            self.B = B.multiply(ph) + A.multiply(1 - ph)
        else:
            self.A = A * ph + B * (1 - ph)
            self.B = B * ph + A * (1 - ph)

        # map each row's midpoint to its bulk segment, index the parsed CN arrays
        idx = _map_bulk_seg_index(coords, seg_df)
        self.clones = clones
        self.cn_A, self.cn_B, self.cn_C, self.cn_BAF = (
            cn_A[idx],
            cn_B[idx],
            cn_C[idx],
            cn_BAF[idx],
        )


def _map_bulk_seg_index(
    coordinates: pd.DataFrame, bulk_profiles: pd.DataFrame
) -> np.ndarray:
    """Map each row's midpoint to the containing ``bulk_profiles`` row index."""
    mid = (
        (coordinates["START"].to_numpy() + coordinates["END"].to_numpy()) / 2
    ).astype(np.int64)
    chrom = coordinates["#CHR"].to_numpy()
    out = np.full(len(coordinates), -1, dtype=np.int64)

    bulk = bulk_profiles.reset_index(drop=True)
    bulk_chr = bulk["#CHR"].to_numpy()
    for c in pd.unique(chrom):
        m = chrom == c
        sub = bulk[bulk_chr == c].sort_values("START")
        starts, ends = sub["START"].to_numpy(), sub["END"].to_numpy()
        bulk_idx = sub.index.to_numpy()
        idx = np.searchsorted(starts, mid[m], side="right") - 1
        safe = idx.clip(min=0)
        valid = (idx >= 0) & (mid[m] < ends[safe])
        rows = np.where(m)[0]
        out[rows[valid]] = bulk_idx[idx[valid]]

    assert (out >= 0).all(), "some rows fall outside bulk_profiles"
    return out


def initialize_count_data(
    barcodes_path: str,
    x_count_path: str,
    a_allele_path: str,
    b_allele_path: str,
    cnv_segments_path: str,
    assay_type: str,
    cell_type_df: pd.DataFrame | None = None,
    ref_label: str | None = None,
    exclude_labels: set[str] | None = None,
) -> CountData:
    """Read one modality's BBC files into a CountData.

    Reads barcodes (+REP_ID), the sparse X/A/B count matrices, and the segment
    table; merges cell types and drops excluded labels when provided.
    """
    barcodes_df = pd.read_table(
        barcodes_path, sep="\t", header=None, names=["BARCODE"], dtype=str
    )
    # REP_ID is everything after the first underscore ("ACGT-1_U1" -> "U1").
    barcodes_df["REP_ID"] = barcodes_df["BARCODE"].str.split("_", n=1).str[1]

    X = sparse.load_npz(x_count_path)
    A = sparse.load_npz(a_allele_path)
    B = sparse.load_npz(b_allele_path)
    coordinates_df = pd.read_table(cnv_segments_path, sep="\t")

    if cell_type_df is not None and ref_label is not None:
        barcodes_df = merge_celltype_into_barcodes(
            barcodes_df, cell_type_df, ref_label, assay_type
        )
        if exclude_labels:
            barcodes_df, X, A, B = exclude_barcodes(
                barcodes_df, exclude_labels, ref_label, X, A, B
            )

    return CountData(
        barcodes=barcodes_df,
        coordinates=coordinates_df,
        X=X,
        A=A,
        B=B,
    )


def _adaptive_segment_ids(
    chrom: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    cn: np.ndarray,
    snp_count: np.ndarray,
    min_snp_count: int,
    max_bin_length: int,
) -> np.ndarray:
    """Per-row segment id: merge contiguous same-CN rows per chromosome until the
    pseudobulk ``snp_count`` reaches ``min_snp_count`` or the span would exceed
    ``max_bin_length`` bp. Returns genomic-order cumulative ids."""
    order = np.lexsort((start, chrom))
    seg = np.empty(len(order), dtype=np.int64)
    sid = -1
    cur_chrom = cur_cn = None
    cur_start = cur_snp = 0
    for i in order:
        start_new = (
            sid < 0
            or chrom[i] != cur_chrom
            or bool(np.any(cn[i] != cur_cn))
            or (end[i] - cur_start) > max_bin_length
            or cur_snp >= min_snp_count
        )
        if start_new:
            sid += 1
            cur_chrom, cur_cn, cur_start, cur_snp = chrom[i], cn[i], start[i], 0
        seg[i] = sid
        cur_snp += snp_count[i]
    return seg


def segment_count_data(
    count_data: dict[str, CountData],
    agg_level: str = "cnp_cluster",
    min_snp_count: int = 300,
    max_bin_length: int = 5_000_000,
) -> dict[str, CountData]:
    """Aggregate a dict of CNP-annotated CountData to segment/cluster level.

    All modalities must share the same rows (coordinates + per-row CN); only the
    counts/cells differ. The grouping is computed **once** and applied to every
    modality, so the segmentation is JOINT across data_types — for ``cnp_bin`` the
    SNP-count cap pools the pseudobulk allele count over ALL cells of ALL modalities.
    Each modality's X/A/B are summed with one sparse matmul; the grouped CN profile
    is carried onto every result.

    - ``cnp_bin``: contiguous same-CN rows per chromosome, capped by ``min_snp_count``
      / ``max_bin_length`` (a CN run split into smaller bins).
    - ``cnp_segment``: all contiguous same-CN rows per chromosome (full CN segments,
      no cap). Both yield per-segment ``segment_id`` + #CHR/START/END coordinates.
    - ``cnp_cluster``: all same-CN rows genome-wide (like to_cluster_level);
      coordinates carry only ``segment_id``.

    All modes name the grouped row id ``segment_id``. Returns ``{assay: CountData}``.
    """
    assays = list(count_data)
    ref = count_data[assays[0]]
    assert ref.cn_A is not None, "call annotate_cnps before segmenting"
    n = ref.num_segment
    assert all(count_data[a].num_segment == n for a in assays), (
        "all modalities must share the same rows for joint segmentation"
    )
    cn = np.concatenate([ref.cn_A, ref.cn_B], axis=1)  # CN identity (shared)

    if agg_level == "cnp_cluster":
        segment_ids = np.unique(cn, axis=0, return_inverse=True)[1].ravel()
        coords = pd.DataFrame({"segment_id": np.arange(int(segment_ids.max()) + 1)})
    elif agg_level in ("cnp_segment", "cnp_bin"):
        c = ref.coordinates
        chrom, start, end = (
            c["#CHR"].to_numpy(),
            c["START"].to_numpy(),
            c["END"].to_numpy(),
        )
        if agg_level == "cnp_bin":
            # joint pseudobulk allele count: pooled over all modalities' cells
            snp_count = sum(
                np.asarray((count_data[a].A + count_data[a].B).sum(axis=1)).ravel()
                for a in assays
            )
            segment_ids = _adaptive_segment_ids(
                chrom, start, end, cn, snp_count, min_snp_count, max_bin_length
            )
        else:
            # full segments: every contiguous same-CN run per chromosome
            order = np.lexsort((start, chrom))
            cn_o, chrom_o = cn[order], chrom[order]
            is_new = np.empty(n, dtype=bool)
            is_new[0] = True
            is_new[1:] = (chrom_o[1:] != chrom_o[:-1]) | np.any(
                cn_o[1:] != cn_o[:-1], axis=1
            )
            segment_ids = np.empty(n, dtype=np.int64)
            segment_ids[order] = np.cumsum(is_new) - 1
        coords = (
            pd.DataFrame(
                {"segment_id": segment_ids, "#CHR": chrom, "START": start, "END": end}
            )
            .groupby("segment_id", sort=True)
            .agg(
                **{
                    "#CHR": ("#CHR", "first"),
                    "START": ("START", "min"),
                    "END": ("END", "max"),
                }
            )
            .reset_index()
        )
    else:
        raise ValueError(f"unknown agg_level: {agg_level!r}")

    G = int(segment_ids.max()) + 1
    indicator = sparse.csr_matrix(
        (np.ones(n, dtype=np.int32), (segment_ids, np.arange(n))), shape=(G, n)
    )
    rep = np.unique(segment_ids, return_index=True)[1]  # representative row per group
    cn_A, cn_B, cn_C, cn_BAF = (
        ref.cn_A[rep],
        ref.cn_B[rep],
        ref.cn_C[rep],
        ref.cn_BAF[rep],
    )

    # per-segment informative-row masks (shared across modalities; CN identical)
    m = get_cnp_mask(cn_A, cn_B, cn_C)
    allele_mask = {k: m[k] for k in ("IMBALANCED", "CLONAL_IMBALANCED")}
    total_mask = {k: m[k] for k in ("ANEUPLOID", "SUBCLONAL")}

    def _sum(mat: np.ndarray | sparse.csr_matrix) -> sparse.csr_matrix:
        return indicator @ (mat if sparse.issparse(mat) else sparse.csr_matrix(mat))

    return {
        a: CountData(
            barcodes=count_data[a].barcodes,
            coordinates=coords.copy(),
            X=_sum(count_data[a].X),
            A=_sum(count_data[a].A),
            B=_sum(count_data[a].B),
            cn_A=cn_A,
            cn_B=cn_B,
            cn_C=cn_C,
            cn_BAF=cn_BAF,
            clones=ref.clones,
            allele_mask={k: v.copy() for k, v in allele_mask.items()},
            total_mask={k: v.copy() for k, v in total_mask.items()},
        )
        for a in assays
    }


def restrict_masks_to_cnp(
    count_data: dict[str, CountData], keep_cn_row: str | None
) -> None:
    """Restrict every CountData's allele/total masks to a CNP whitelist (in place).

    ``keep_cn_row`` is a comma-separated list of CNP strings (e.g.
    ``"1|1;2|1,2|2;3|1"``); rows whose CNP is not in the set are masked out of
    inference. No-op when falsy. CNP per row is reconstructed from the shared
    ``cn_A``/``cn_B``.
    """
    if not keep_cn_row:
        return
    keep_set = {r.strip() for r in keep_cn_row.split(",") if r.strip()}
    ref = count_data[next(iter(count_data))]
    cn_A, cn_B, G = ref.cn_A, ref.cn_B, ref.num_segment
    cnp = np.array(
        [";".join(f"{a}|{b}" for a, b in zip(cn_A[g], cn_B[g])) for g in range(G)]
    )
    keep = np.isin(cnp, list(keep_set))
    n_kept = int(keep.sum())
    assert n_kept > 0, "keep_cn_row matched 0 CNP clusters; check format vs CNP"
    for cd in count_data.values():
        cd.allele_mask = {k: v & keep for k, v in cd.allele_mask.items()}
        cd.total_mask = {k: v & keep for k, v in cd.total_mask.items()}
    logging.info(f"keep_cn_row: inference uses {n_kept}/{G} CNP clusters")


def smooth_spatial_neighbors(
    count_data: dict[str, CountData],
    spatial_graphs: dict[str, dict],
    max_k: int,
    min_umi: int,
    min_snp_umi: int,
) -> dict[str, CountData]:
    """Adaptive per-spot spatial smoothing of X/A/B (returns a new CountData dict).

    For each spot, progressively pool its k=1..max_k hop neighborhood (binary
    reachability from the spatial graph) until per-spot total UMI >= ``min_umi``
    AND total allele (A+B) >= ``min_snp_umi``, or ``max_k`` is reached. Spots
    already meeting both thresholds are left untouched. No-op (returns the input
    dict) when ``max_k <= 0``; an assay absent from ``spatial_graphs`` is passed
    through unchanged.

    Args:
        count_data: per-assay CountData (smoothing is independent per assay).
        spatial_graphs: ``{assay -> {rep -> {"BARCODE": array, "W": sparse}}}``.
        max_k: maximum neighborhood radius (0 disables smoothing).
        min_umi: minimum per-spot total UMI threshold.
        min_snp_umi: minimum per-spot total allele (A+B) threshold.
    """
    if max_k <= 0:
        return count_data

    out: dict[str, CountData] = {}
    for assay, cd in count_data.items():
        W_per_rep = spatial_graphs.get(assay)
        if W_per_rep is None:
            out[assay] = cd
            continue

        def _dense_copy(m: np.ndarray | sparse.csr_matrix) -> np.ndarray:
            return np.asarray(m.todense()) if sparse.issparse(m) else np.array(m)

        X, A, B = _dense_copy(cd.X), _dense_copy(cd.A), _dense_copy(cd.B)
        pos = {bc: i for i, bc in enumerate(cd.barcodes["BARCODE"].to_numpy())}
        spot_k = np.zeros(cd.num_cell, dtype=int)  # smoothing radius per spot

        for sg in W_per_rep.values():
            sg_bcs = sg["BARCODE"]
            N_rep = len(sg_bcs)
            col_idx = np.array([pos[bc] for bc in sg_bcs if bc in pos])
            if len(col_idx) != N_rep:  # rep not fully present in this assay
                continue

            # binary reachability (self + neighbors)^(kk+1) for kk=0..max_k-1
            adj = sparse.eye(N_rep, format="csr") + sparse.csr_matrix(
                sg["W"] > 0
            ).astype(np.int8)
            reach_list, reach = [], adj
            for kk in range(max_k):
                if kk > 0:
                    reach = reach @ adj
                reach_list.append((reach > 0).astype(np.int8))

            X_orig, A_orig, B_orig = X[:, col_idx], A[:, col_idx], B[:, col_idx]
            for kk in range(max_k):
                cur_T = X[:, col_idx].sum(axis=0)
                cur_D = (A[:, col_idx] + B[:, col_idx]).sum(axis=0)
                needs = (cur_T < min_umi) | (cur_D < min_snp_umi)
                if not needs.any():
                    break
                R = reach_list[kk].T  # (N_rep, N_rep); column j = its kk-hop nbhd
                sel = col_idx[needs]
                X[:, sel] = (X_orig @ R)[:, needs]
                A[:, sel] = (A_orig @ R)[:, needs]
                B[:, sel] = (B_orig @ R)[:, needs]
                spot_k[sel] = kk + 1

        n_smoothed = int((spot_k > 0).sum())
        logging.info(
            f"[{assay}] spatial smoothing (max_k={max_k}, min_umi={min_umi}, "
            f"min_snp_umi={min_snp_umi}): {n_smoothed}/{cd.num_cell} spots smoothed"
        )
        out[assay] = replace(cd, X=X, A=A, B=B)
    return out


def save_count_data(count_data: dict[str, CountData], prefix: str) -> None:
    """Save each assay's CountData (coordinates + count matrices) to disk.

    For every ``assay`` writes ``{prefix}.{assay}.tsv.gz`` (coordinates) and
    ``{prefix}.{assay}.{X,B,C}.npz`` where X = read depth, B = B-allele,
    C = total allele (A + B).
    """
    for assay, cd in count_data.items():
        p = f"{prefix}.{assay}"
        cd.coordinates.to_csv(f"{p}.tsv.gz", sep="\t", index=False, compression="gzip")
        sparse.save_npz(f"{p}.X.npz", sparse.csr_matrix(cd.X))
        sparse.save_npz(f"{p}.B.npz", sparse.csr_matrix(cd.B))
        sparse.save_npz(f"{p}.C.npz", sparse.csr_matrix(cd.A + cd.B))
