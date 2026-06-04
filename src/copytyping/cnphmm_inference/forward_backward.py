"""Batched log-space forward-backward for the CNP-HMM E-step.

Cells are i.i.d. given ``(Theta, H)``, so the E-step streams over cell chunks
and accumulates the cross-cell sufficient statistics — never materializing the
global ``(N, G, K)`` posterior. Per chunk it runs a vectorized log-space
forward-backward (``fb_chunk``) and reduces immediately into ``pi_acc``,
cell-summed ``Xi``, the phasing emission ``eH``, and the total log-likelihood.
"""

import numpy as np
from scipy import sparse
import numba
from numba import njit, prange

from copytyping.cnphmm_inference.emissions import build_log_emissions

_NEG = -1.0e300  # finite stand-in for -inf (fastmath-safe)


@njit(parallel=True, fastmath=True, cache=True)
def _fb_kernel(log_e, log_pi, logA, gamma, ll, Xi_threads):
    """Per-cell forward-backward in compiled parallel code.

    Loops cells with ``prange`` (cells are independent), writing ``gamma`` and
    ``ll`` and accumulating the transition posterior into per-thread ``Xi`` so
    the cross-cell reduction is lock-free. Each cell uses local ``(G, K)``
    forward/backward buffers; the log-sum-exp is done inline with the
    max-subtraction trick (no temporaries, no scipy overhead).
    """
    n, G, K = log_e.shape
    for i in prange(n):
        tid = numba.get_thread_id()
        la = np.empty((G, K))
        lb = np.empty((G, K))

        # forward
        for k in range(K):
            la[0, k] = log_pi[k] + log_e[i, 0, k]
        for g in range(1, G):
            for k in range(K):
                m = _NEG
                for c in range(K):
                    v = la[g - 1, c] + logA[g - 1, c, k]
                    if v > m:
                        m = v
                s = 0.0
                for c in range(K):
                    s += np.exp(la[g - 1, c] + logA[g - 1, c, k] - m)
                la[g, k] = m + np.log(s) + log_e[i, g, k]

        # backward
        for k in range(K):
            lb[G - 1, k] = 0.0
        for g in range(G - 2, -1, -1):
            for c in range(K):
                m = _NEG
                for k in range(K):
                    v = logA[g, c, k] + log_e[i, g + 1, k] + lb[g + 1, k]
                    if v > m:
                        m = v
                s = 0.0
                for k in range(K):
                    s += np.exp(logA[g, c, k] + log_e[i, g + 1, k] + lb[g + 1, k] - m)
                lb[g, c] = m + np.log(s)

        # marginal log-likelihood
        m = _NEG
        for k in range(K):
            if la[G - 1, k] > m:
                m = la[G - 1, k]
        s = 0.0
        for k in range(K):
            s += np.exp(la[G - 1, k] - m)
        lli = m + np.log(s)
        ll[i] = lli

        # gamma
        for g in range(G):
            for k in range(K):
                gamma[i, g, k] = np.exp(la[g, k] + lb[g, k] - lli)

        # xi -> thread-local accumulator
        for g in range(G - 1):
            for c in range(K):
                la_gc = la[g, c]
                for k in range(K):
                    Xi_threads[tid, g, c, k] += np.exp(
                        la_gc + logA[g, c, k] + log_e[i, g + 1, k] + lb[g + 1, k] - lli
                    )


def fb_chunk(
    log_e: np.ndarray,
    log_pi: np.ndarray,
    logA: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log-space forward-backward for a chunk of ``n`` cells (numba-accelerated,
    parallel over cells).

    ``log_e`` is ``(n, G, K)``, ``log_pi`` ``(K,)``, ``logA`` ``(G-1, K, K)``.
    Returns ``(gamma (n, G, K), Xi (G-1, K, K), ll (n,))`` where ``Xi`` is the
    cell-summed transition posterior and ``ll`` the per-cell marginal
    log-likelihood.
    """
    n, G, K = log_e.shape
    nthreads = numba.get_num_threads()
    gamma = np.empty((n, G, K), dtype=np.float64)
    ll = np.empty(n, dtype=np.float64)
    Xi_threads = np.zeros((nthreads, G - 1, K, K), dtype=np.float64)
    _fb_kernel(
        np.ascontiguousarray(log_e, dtype=np.float64),
        np.ascontiguousarray(log_pi, dtype=np.float64),
        np.ascontiguousarray(logA, dtype=np.float64),
        gamma,
        ll,
        Xi_threads,
    )
    return gamma, Xi_threads.sum(axis=0), ll


def _accumulate_eH_soft(
    eH: np.ndarray,
    gamma: np.ndarray,
    bb_pack: tuple,
) -> None:
    """Add this chunk's gamma-weighted phasing emission into ``eH`` in place.

    ``eH[g, h] += sum_i sum_k gamma[i, g, k] * bb_h[(i, g), k]`` over the BB
    nonzeros, for both orientations ``h``.
    """
    cg, cn, bb0, bb1 = bb_pack
    gw = gamma[cn, cg, :]  # (nnz, K)
    np.add.at(eH[:, 0], cg, (gw * bb0).sum(axis=1))
    np.add.at(eH[:, 1], cg, (gw * bb1).sum(axis=1))


def accumulate_eH_hard(
    eH: np.ndarray,
    Z_chunk: np.ndarray,
    bb_pack: tuple,
) -> None:
    """Hard-assignment phasing emission: ``eH[g, h] += sum_i bb_h[(i,g), Z_ig]``.

    ``Z_chunk`` is ``(n, G)`` hard states for the chunk; ``bb_pack`` the BB
    nonzero pack from ``build_log_emissions(return_bb_orient=True)``.
    """
    cg, cn, bb0, bb1 = bb_pack
    sel = Z_chunk[cn, cg]  # (nnz,)
    rows = np.arange(sel.size)
    np.add.at(eH[:, 0], cg, bb0[rows, sel])
    np.add.at(eH[:, 1], cg, bb1[rows, sel])


def estep(
    X: sparse.csr_matrix,
    B: sparse.csr_matrix,
    C: sparse.csr_matrix,
    T: np.ndarray,
    rdr_baf_cn: np.ndarray,
    rdr_baf_params: np.ndarray,
    base_props: np.ndarray,
    H: np.ndarray,
    log_pi: np.ndarray,
    logA: np.ndarray,
    fit_mode: str,
    chunk_size: int,
    eps: float = 1e-12,
    Z_out: np.ndarray | None = None,
    conf_out: np.ndarray | None = None,
    chrom_start: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Streamed soft E-step. Returns ``(pi_acc (K,), Xi (G-1, K, K), total_ll,
    eH (G, 2))``. If ``Z_out`` (preallocated ``(N, G)``) is given, fills it with
    the per-cell posterior argmax path (for the hard dispersion M-step / decode).
    If ``conf_out`` (preallocated ``(N,)``) is given, fills it with each cell's
    decode confidence = mean over segments of the max posterior marginal.

    ``pi_acc`` is the soft count of the initial state. With ``chrom_start`` (G,)
    bool — each chromosome's first segment — it is pooled over every chromosome
    start (each chromosome begins from ``pi``); otherwise only segment 0 is used.
    """
    G, N = X.shape
    K = rdr_baf_cn.shape[0]
    start_segs = np.array([0]) if chrom_start is None else np.flatnonzero(chrom_start)
    Xc, Bc, Cc = X.tocsc(), B.tocsc(), C.tocsc()

    pi_acc = np.zeros(K, dtype=np.float64)
    Xi = np.zeros((G - 1, K, K), dtype=np.float64)
    eH = np.zeros((G, 2), dtype=np.float64)
    total_ll = 0.0

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        X_sub = Xc[:, start:end].tocsr()
        B_sub = Bc[:, start:end].tocsr()
        C_sub = Cc[:, start:end].tocsr()
        T_sub = T[start:end]

        log_e, bb_pack = build_log_emissions(
            X_sub,
            B_sub,
            C_sub,
            T_sub,
            rdr_baf_cn,
            rdr_baf_params,
            base_props,
            H,
            fit_mode,
            eps=eps,
            return_bb_orient=True,
        )
        gamma, Xi_c, ll = fb_chunk(log_e, log_pi, logA)
        pi_acc += gamma[:, start_segs, :].sum(axis=(0, 1))
        Xi += Xi_c
        total_ll += float(ll.sum())
        if bb_pack is not None:
            _accumulate_eH_soft(eH, gamma, bb_pack)
        if Z_out is not None:
            Z_out[start:end] = gamma.argmax(axis=2)
        if conf_out is not None:
            conf_out[start:end] = gamma.max(axis=2).mean(axis=1)

    return pi_acc, Xi, total_ll, eH
