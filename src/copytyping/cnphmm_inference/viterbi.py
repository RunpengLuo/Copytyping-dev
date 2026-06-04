"""Viterbi decoders for the factorial CNP-HMM: per-cell CN paths (given the
shared phasing ``H``) and the shared 2-state phasing path (given expected/hard
per-segment phase emissions). Both DPs are numba kernels — the per-cell decoder
runs ``prange`` over cells with thread-local backpointer buffers (so memory is
``O(threads * G * K)`` rather than the ``O(n * G * K)`` of a materialized
trellis), and the phasing chain is a small sequential ``njit``.
"""

import math

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def _viterbi_z_kernel(
    log_e: np.ndarray,
    log_pi: np.ndarray,
    logA: np.ndarray,
    Z: np.ndarray,
) -> None:
    """Per-cell CN-path Viterbi DP, parallel over cells. Writes the MAP path into
    ``Z`` (n, G). Each cell uses a thread-local ``(G, K)`` backpointer buffer.
    Argmax ties keep the lowest state index (matches ``np.argmax``)."""
    n, G, K = log_e.shape
    for i in prange(n):
        dp_prev = np.empty(K, dtype=np.float64)
        dp_cur = np.empty(K, dtype=np.float64)
        bp = np.empty((G, K), dtype=np.int32)
        for k in range(K):
            dp_prev[k] = log_pi[k] + log_e[i, 0, k]
        for g in range(1, G):
            for kc in range(K):
                best = dp_prev[0] + logA[g - 1, 0, kc]
                arg = 0
                for kp in range(1, K):
                    val = dp_prev[kp] + logA[g - 1, kp, kc]
                    if val > best:
                        best = val
                        arg = kp
                dp_cur[kc] = best + log_e[i, g, kc]
                bp[g, kc] = arg
            for k in range(K):
                dp_prev[k] = dp_cur[k]
        # terminate: argmax over the last column (first max on ties)
        best = dp_prev[0]
        arg = 0
        for k in range(1, K):
            if dp_prev[k] > best:
                best = dp_prev[k]
                arg = k
        Z[i, G - 1] = arg
        for g in range(G - 2, -1, -1):
            arg = bp[g + 1, arg]
            Z[i, g] = arg


def viterbi_Z(
    log_e: np.ndarray,
    log_pi: np.ndarray,
    logA: np.ndarray,
) -> np.ndarray:
    """Per-cell CN-path Viterbi, parallel over the ``n`` cells in the chunk.

    ``log_e`` is ``(n, G, K)``, ``log_pi`` ``(K,)``, ``logA`` ``(G-1, K, K)``.
    Returns the MAP state path ``Z`` of shape ``(n, G)`` (int32).
    """
    n, G, _K = log_e.shape
    Z = np.empty((n, G), dtype=np.int32)
    _viterbi_z_kernel(
        np.ascontiguousarray(log_e, dtype=np.float64),
        np.ascontiguousarray(log_pi, dtype=np.float64),
        np.ascontiguousarray(logA, dtype=np.float64),
        Z,
    )
    return Z


@njit(cache=True)
def _viterbi_h_kernel(
    eH: np.ndarray,
    logp: np.ndarray,
    log1mp: np.ndarray,
    H: np.ndarray,
) -> None:
    """Shared 2-state phasing-path Viterbi DP (sequential over segments). Writes
    the MAP orientation into ``H`` (G,). Ties (stay vs switch, and the terminal
    argmax) prefer ``stay`` / the lower index, matching the numpy reference."""
    G = eH.shape[0]
    dp = np.empty((G, 2), dtype=np.float64)
    bp = np.empty((G, 2), dtype=np.int8)
    dp[0, 0] = math.log(0.5) + eH[0, 0]
    dp[0, 1] = math.log(0.5) + eH[0, 1]
    for g in range(1, G):
        for h in range(2):
            stay = dp[g - 1, h] + log1mp[g]
            switch = dp[g - 1, 1 - h] + logp[g]
            if stay >= switch:
                bp[g, h] = h
                dp[g, h] = stay + eH[g, h]
            else:
                bp[g, h] = 1 - h
                dp[g, h] = switch + eH[g, h]
    last = 0 if dp[G - 1, 0] >= dp[G - 1, 1] else 1
    H[G - 1] = last
    for g in range(G - 2, -1, -1):
        last = bp[g + 1, last]
        H[g] = last


def viterbi_H(
    eH: np.ndarray,
    switchprobs_seg: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Shared phasing-path Viterbi over the 2-state ``H`` chain.

    ``eH`` is the ``(G, 2)`` per-segment phase emission (log scale; column ``h``
    aggregates the data evidence for orientation ``h``). ``switchprobs_seg[g]``
    is the prob of flipping orientation entering segment ``g`` (``g >= 1``;
    index 0 unused). ``H_1`` is uniform. Returns ``H`` of shape ``(G,)`` (int8).
    """
    G = eH.shape[0]
    logp = np.log(np.clip(switchprobs_seg, eps, 1.0 - eps))
    log1mp = np.log(np.clip(1.0 - switchprobs_seg, eps, 1.0 - eps))
    H = np.empty(G, dtype=np.int8)
    _viterbi_h_kernel(
        np.ascontiguousarray(eH, dtype=np.float64),
        np.ascontiguousarray(logp, dtype=np.float64),
        np.ascontiguousarray(log1mp, dtype=np.float64),
        H,
    )
    return H
