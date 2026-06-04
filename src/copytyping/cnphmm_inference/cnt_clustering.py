"""Copy-Number-Transformation (CNT) distance between cells and clustering /
trees over them.

The CNT distance is the minimum number of segmental amplification / deletion
events (each adds +1 or -1 to a contiguous run of segments) that transforms one
integer copy-number profile into another. It was introduced and solved in linear
time by:

    Ron Zeira, Meirav Zehavi, Ron Shamir.
    "A Linear-Time Algorithm for the Copy Number Transformation Problem."
    Journal of Computational Biology 24(12):1179-1194, 2017
    (preliminary version: CPM 2016).

This module implements the *unconstrained* interval-event distance (intermediate
copy numbers may go negative), which for a single allele equals the up-variation
of the positive part of the difference vector plus that of its negative part — a
closed form, symmetric, and a faithful proxy of the Zeira et al. model. Copy-
number events do not span chromosomes, so the distance is **summed per
chromosome** (a 0 boundary is inserted at every chromosome edge so no event can
cross it). The allele-specific distance sums the A- and B-allele CNT distances.

The CNT distance is a *custom, non-Euclidean* metric, so it is fed to clustering
as a precomputed distance matrix (condensed via ``scipy.spatial.distance``):

  * ``cnt_upgma``    — average linkage (UPGMA). The recommended default for a
                       custom distance: it averages cross-cluster distances and
                       makes no Euclidean assumption.
  * ``cnt_complete`` — complete linkage. More conservative (merges only when all
                       cross-distances are small); useful to avoid chaining.
  * ``cnt_nj``       — neighbor-joining (scikit-bio; additive unrooted tree). For
                       small N only — it is slow and needs the full square matrix.

Ward linkage is intentionally **not** offered: it assumes Euclidean feature
vectors and minimizes within-cluster variance, which is meaningless on an
arbitrary precomputed distance matrix. Single linkage is omitted for its chaining
behavior.
"""

import logging
import sys

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list, to_tree, fcluster
from scipy.spatial.distance import squareform
from skbio import DistanceMatrix
from skbio.tree import nj as _skbio_nj

# heatmap cell-ordering methods (lexsort is handled by the caller, not here)
CLUSTER_METHODS = ("lexsort", "cnt_nj", "cnt_upgma", "cnt_complete")

# method -> scipy linkage method (agglomerative methods only; cnt_nj is skbio)
_LINKAGE_METHOD = {"cnt_upgma": "average", "cnt_complete": "complete"}


def _cnt_one_allele(M: np.ndarray, chrom: np.ndarray) -> np.ndarray:
    """All-pairs unconstrained CNT distance for one allele's CN matrix, summed
    per chromosome.

    ``M`` is ``(n, G)`` integer copy numbers; ``chrom`` is ``(G,)`` per-segment
    chromosome labels. A 0 column is inserted at every chromosome edge so the
    up-variation accumulates independently within each chromosome (events cannot
    span boundaries). Returns the symmetric ``(n, n)`` distance, computed row by
    row to keep memory at ``O(n * G)``.
    """
    n, G = M.shape
    chrom = np.asarray(chrom)
    change = np.flatnonzero(chrom[1:] != chrom[:-1]) + 1
    insert_at = np.concatenate(([0], change, [G]))  # 0-pad each chromosome block
    Mp = np.insert(M.astype(np.int64), insert_at, 0, axis=1)  # (n, G + n_chrom + 1)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        w = Mp[i][None, :] - Mp
        cpos = np.maximum(w, 0)
        cneg = np.maximum(-w, 0)
        D[i] = np.maximum(np.diff(cpos, axis=1), 0).sum(axis=1) + np.maximum(
            np.diff(cneg, axis=1), 0
        ).sum(axis=1)
    return D


def compute_cnt_dist(A: np.ndarray, B: np.ndarray, chrom: np.ndarray) -> np.ndarray:
    """Symmetric all-pairs Copy-Number-Transformation distance over cells.

    ``A`` / ``B`` are the per-cell allele-specific CN matrices ``(n, G)``;
    ``chrom`` is the ``(G,)`` per-segment chromosome label. Returns the ``(n, n)``
    CNT distance ``D[i,j] = CNT(A_i, A_j) + CNT(B_i, B_j)``, summed over
    chromosomes (Zeira, Zehavi & Shamir 2017; see module docstring).

    Cost is ``O(n^2 G)`` time and ``O(n^2)`` memory — intended for a modest number
    of cells / unique profiles (≲ a few thousand), not all cells of a large set.
    """
    assert A.shape == B.shape, (A.shape, B.shape)
    return _cnt_one_allele(A, chrom) + _cnt_one_allele(B, chrom)


def _skbio_nj_tree(dist: np.ndarray, ids: list[str]) -> tuple[str, np.ndarray]:
    """Run scikit-bio neighbor-joining and return ``(newick, leaf_order)`` with
    ``leaf_order`` the tip traversal (original row indices)."""
    dm = DistanceMatrix(np.asarray(dist, dtype=np.float64), ids=ids)
    tree = _skbio_nj(dm)
    newick = str(tree).strip()
    idx_of = {name: i for i, name in enumerate(ids)}
    leaf_order = np.array([idx_of[t.name] for t in tree.tips()], dtype=np.int64)
    return newick, leaf_order


def _linkage_newick(Z: np.ndarray, ids: list[str]) -> str:
    """Convert a scipy linkage matrix to a Newick string (branch length = height
    difference between a node and its parent)."""
    sys.setrecursionlimit(max(10000, 4 * len(ids)))
    root = to_tree(Z, rd=False)

    def rec(node, parent_h: float) -> str:
        bl = max(parent_h - node.dist, 0.0)
        if node.is_leaf():
            return f"{ids[node.id]}:{bl:.6f}"
        return f"({rec(node.left, node.dist)},{rec(node.right, node.dist)}):{bl:.6f}"

    return f"({rec(root.left, root.dist)},{rec(root.right, root.dist)});"


def build_cnt_tree(
    dist: np.ndarray, method: str, labels: list[str] | None = None
) -> tuple[str, np.ndarray, np.ndarray | None]:
    """Build a tree / clustering over a symmetric distance matrix and return
    ``(newick, leaf_order, linkage)``. ``method``:

      * ``cnt_upgma``    — average linkage (UPGMA; scipy; ultrametric dendrogram).
      * ``cnt_complete`` — complete linkage (scipy; dendrogram).
      * ``cnt_nj``       — neighbor-joining (scikit-bio; additive unrooted tree;
                           ``linkage`` is ``None`` — not a dendrogram).

    ``leaf_order`` is the tip / leaf ordering (original row indices) for ordering
    heatmap rows; ``linkage`` is the scipy linkage matrix for the agglomerative
    methods (for drawing a dendrogram), ``None`` for ``cnt_nj``. The agglomerative
    methods consume the distance as a condensed matrix via
    ``squareform(..., checks=False)``. Requires ``m >= 3``.
    """
    m = dist.shape[0]
    ids = [str(i) for i in range(m)] if labels is None else [str(x) for x in labels]
    if method == "cnt_nj":
        newick, order = _skbio_nj_tree(dist, ids)
        link = None
    elif method in _LINKAGE_METHOD:
        link = linkage(
            squareform(np.asarray(dist, dtype=np.float64), checks=False),
            method=_LINKAGE_METHOD[method],
        )
        order = leaves_list(link).astype(np.int64)
        newick = _linkage_newick(link, ids)
    else:
        raise ValueError(f"unknown tree method: {method}")
    logging.info(f"build_cnt_tree: {method} tree over {m} taxa")
    return newick, order, link


def cluster_cnt(
    dist: np.ndarray, n_clusters: int, method: str = "cnt_upgma"
) -> np.ndarray:
    """Flat clustering of a precomputed CNT distance matrix into ``n_clusters``
    groups. Builds an agglomerative linkage (``cnt_upgma`` = average, default, or
    ``cnt_complete`` = complete) over the condensed distance and cuts it with
    ``fcluster(..., criterion="maxclust")``. Returns the ``(m,)`` integer cluster
    labels (1-based, as SciPy returns). Ward is not supported (Euclidean-only)."""
    if method not in _LINKAGE_METHOD:
        raise ValueError(f"cluster_cnt supports {tuple(_LINKAGE_METHOD)}, got {method}")
    link = linkage(
        squareform(np.asarray(dist, dtype=np.float64), checks=False),
        method=_LINKAGE_METHOD[method],
    )
    return fcluster(link, t=n_clusters, criterion="maxclust")


def assign_clones(
    Z: np.ndarray,
    rep_id: np.ndarray,
    cn_states: np.ndarray,
    chrom: np.ndarray,
    n_clones: int,
    method: str = "cnt_upgma",
    max_taxa: int = 2000,
) -> np.ndarray:
    """Per-REP flat clone assignment over the decoded CN paths.

    Within each REP_ID the unique decoded profiles are clustered by cutting the
    CNT-distance hierarchy (``method`` = ``cnt_upgma`` average, default, or
    ``cnt_complete``) into ``min(n_clones, U_rep)`` groups via
    :func:`cluster_cnt`; every cell inherits its profile's label. Reps whose
    unique-profile count exceeds ``max_taxa`` are left unclustered (the all-pairs
    CNT distance is ``O(U^2)`` — the same cap used for heatmap ordering).

    ``Z`` is the ``(N, G)`` decoded state-index matrix; ``cn_states`` the
    ``(K, 2)`` state -> ``(A, B)`` map; ``chrom`` the ``(G,)`` per-segment
    chromosome label. Returns the ``(N,)`` per-cell clone label (1-based **within
    a rep**, so disambiguate with REP_ID; ``-1`` where the rep was unclustered).
    """
    rep_id = np.asarray(rep_id)
    clone = np.full(rep_id.shape[0], -1, dtype=np.int64)
    for rep in np.unique(rep_id):
        idx = np.flatnonzero(rep_id == rep)
        uniq, inv = np.unique(Z[idx], axis=0, return_inverse=True)
        inv = inv.ravel()
        U = uniq.shape[0]
        if U == 1:
            clone[idx] = 1
            continue
        if U > max_taxa:
            logging.info(
                f"assign_clones: REP {rep} has {U} unique profiles > {max_taxa} "
                f"cap; left unclustered"
            )
            continue
        dist = compute_cnt_dist(cn_states[uniq, 0], cn_states[uniq, 1], chrom)
        labels = cluster_cnt(dist, min(n_clones, U), method=method)
        clone[idx] = labels[inv]
        logging.info(
            f"assign_clones: REP {rep} -> {len(np.unique(labels))} clones "
            f"from {U} unique profiles"
        )
    return clone
