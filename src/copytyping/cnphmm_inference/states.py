"""CN-state space (full grid masked to bulk-observed states) and the
bulk-CNP-informed Dirichlet transition prior for the factorial CNP-HMM.

State indices in this module live in the *masked* index space, distinct from
the *bulk* index space used by ``data_io.load_bulk_cnp``. The
``bulk_to_masked`` map returned by :func:`build_state_space` is the single
bridge between the two; remap every bulk-indexed array (cna_profile,
rdr_baf_params, ...) through it exactly once.
"""

import logging

import numpy as np


def make_state_transmat(t: float, n_states: int) -> np.ndarray:
    """K×K doubly-stochastic CN-state transition matrix. ``t`` is the per-row
    off-diagonal total mass (split uniformly across the K−1 other states);
    diagonal = ``1 - t``. Used as the sticky baseline ``q`` in the bulk-anchored
    transition prior (:func:`build_transition_prior`).
    """
    if n_states == 1:
        return np.ones((1, 1), dtype=np.float64)
    offdiag = t / (n_states - 1)
    transmat = np.full((n_states, n_states), offdiag, dtype=np.float64)
    np.fill_diagonal(transmat, 1.0 - t)
    return transmat


def _full_grid(c_max: int) -> np.ndarray:
    """All ``(a, b)`` with ``a, b >= 0`` and ``0 < a + b <= c_max``."""
    grid = [
        (a, b) for a in range(c_max + 1) for b in range(c_max + 1) if 0 < a + b <= c_max
    ]
    return np.array(grid, dtype=np.int32)


def build_state_space(
    c_max: int,
    mask_mode: str,
    cna_int_states: np.ndarray,
    cna_profile_seg: np.ndarray,
    baf_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the masked CN-state space and remap the bulk CN profile into it.

    ``cna_int_states`` (K_bulk, 2) and ``cna_profile_seg`` (G, M0) are in bulk
    index space (from ``load_bulk_cnp`` + ``perform_segmentation``).

    ``mask_mode``:
      * ``"full"``          — keep the entire grid up to ``c_max``.
      * ``"bulk"``          — keep only ``(a, b)`` states observed in the bulk
                              profile.
      * ``"bulk_neighbor"`` — bulk-observed states plus their ``(a±1, b±1)``
                              grid neighbors.

    Returns ``(cn_states (K, 2) int, rdr_baf_cn (K, 2) float [rdr, baf],
    bulk_to_masked (K_bulk,) int, cna_profile_masked (G, M0) int)``. Bulk states
    absent from the masked set map to ``-1`` in ``bulk_to_masked``; this function
    asserts every state actually referenced by ``cna_profile_seg`` is present.
    """
    grid = _full_grid(c_max)
    grid_set = {(int(a), int(b)) for a, b in grid}

    observed_idx = np.unique(cna_profile_seg)
    observed = {tuple(int(x) for x in cna_int_states[i]) for i in observed_idx}

    # bulk states outside the c_max grid are DROPPED (not added). Segments whose
    # bulk clone sits in a dropped state map to -1 in cna_profile_masked and are
    # skipped downstream (transition prior, pi init).
    observed_in_grid = observed & grid_set
    if mask_mode == "full":
        keep = set(grid_set)
    elif mask_mode == "bulk":
        keep = set(observed_in_grid)
    elif mask_mode == "bulk_neighbor":
        keep = set(observed_in_grid)
        for a, b in observed_in_grid:
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    na, nb = a + da, b + db
                    if na >= 0 and nb >= 0 and 0 < na + nb <= c_max:
                        keep.add((na, nb))
    else:
        raise ValueError(f"unknown mask_mode: {mask_mode}")

    dropped = observed - grid_set
    if dropped:
        logging.warning(
            f"build_state_space: dropping {len(dropped)} bulk states outside "
            f"c_max={c_max} grid: {sorted(dropped)}"
        )

    # canonical order: by total CN (a + b) ascending, then by A descending
    # (a > b before a < b). Gives (1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),...
    cn_states = np.array(
        sorted(keep, key=lambda ab: (ab[0] + ab[1], -ab[0])), dtype=np.int32
    )
    K = len(cn_states)
    totals = cn_states[:, 0] + cn_states[:, 1]
    rdr = totals / 2.0
    baf = np.clip(cn_states[:, 1] / totals, baf_clip, 1.0 - baf_clip)
    rdr_baf_cn = np.column_stack([rdr, baf]).astype(np.float64)

    lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(cn_states)}
    bulk_to_masked = np.array(
        [lookup.get((int(a), int(b)), -1) for a, b in cna_int_states],
        dtype=np.int64,
    )
    cna_profile_masked = bulk_to_masked[cna_profile_seg]  # -1 where bulk state dropped
    n_dropped_entries = int((cna_profile_masked < 0).sum())

    logging.info(
        f"build_state_space: mask_mode={mask_mode}, c_max={c_max}, "
        f"{len(grid)} grid -> {K} masked states (bulk-observed {len(observed)}, "
        f"{n_dropped_entries} (seg,clone) entries dropped to -1)"
    )
    return cn_states, rdr_baf_cn, bulk_to_masked, cna_profile_masked


def build_transition_prior(
    cna_profile_masked: np.ndarray,
    K: int,
    s: float,
    omega: float,
    t: float,
    eps: float,
) -> np.ndarray:
    """Bulk-anchored Dirichlet concentration ``alpha`` per transition row.

    For each adjacent segment pair ``(g, g+1)`` and the ``M0`` bulk paths
    (columns of ``cna_profile_masked``):

        n0[c, c'] = # bulk paths transitioning c -> c'
        q[c, .]   = sticky baseline (diagonal t, off-diagonal (1-t)/(K-1))
        rho[c]    = 1{ sum_u n0[c, u] > 0 }            anchor indicator
        nhat[c,.] = (n0[c, .] + eps) / row-sum
        w[c]      = omega * rho[c]                     per-row bulk weight
        alpha[g]  = s * ((1 - w) * q + w * nhat)

    Anchored rows (state ``c`` observed in the bulk at segment ``g``) blend the
    sticky baseline with the bulk transitions; unanchored rows (``rho=0``) fall
    back to ``s * q`` and favor self-transition. ``q`` comes from
    ``make_state_transmat`` with the complementary argument. Returns ``alpha`` of
    shape ``(G-1, K, K)``.
    """
    G, _M0 = cna_profile_masked.shape
    q = make_state_transmat(1.0 - t, K)  # diagonal t, off-diag (1-t)/(K-1)
    alpha = np.empty((G - 1, K, K), dtype=np.float64)
    for g in range(G - 1):
        n0 = np.zeros((K, K), dtype=np.float64)
        a, b = cna_profile_masked[g], cna_profile_masked[g + 1]
        ok = (a >= 0) & (b >= 0)  # skip bulk paths in a dropped (out-of-grid) state
        np.add.at(n0, (a[ok], b[ok]), 1.0)
        rho = (n0.sum(axis=1) > 0).astype(np.float64)  # (K,) anchored states
        nhat = (n0 + eps) / (n0 + eps).sum(axis=1, keepdims=True)
        w = (omega * rho)[:, None]  # (K, 1) per-row bulk weight
        alpha[g] = s * ((1.0 - w) * q + w * nhat)
    return alpha


def prior_mean_transitions(alpha: np.ndarray) -> np.ndarray:
    """Row-normalized Dirichlet mean of each transition matrix, ``(G-1, K, K)``."""
    return alpha / alpha.sum(axis=2, keepdims=True)
