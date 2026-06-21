"""Per-(cell, segment, state) log-emission tensors for the factorial CNP-HMM.

Unlike the per-clone ``(N, M)`` likelihoods in ``initialize`` (used only for
reference-cell seeding), the HMM needs per-state ``(n, G, K)`` emissions for a
chunk of ``n`` cells given the
current shared phasing path ``H``. The NB (read-depth) term is phase-independent;
the BB (allele) term flips its BAF when ``H_g = 0``.

The per-element math is done in numba kernels (parallel over cells / nonzeros):
the NB dense ``x=0`` baseline and sparse ``x>0`` adjustment, and the BB
log-PMF for both phase orientations via ``math.lgamma`` (the
``betaln(alpha, beta)`` term collapses to per-``k`` constants precomputed on the
host). This fuses away the ``(n, G, K)`` ``mu`` temporary and the scipy
``betaln``/``gammaln`` overhead.

NB mean convention (per-state generalization of the single-cell ``clone_norm``
branch): ``mu_{i,g,k} = T_i * base_props[g] * rdr_k``. ``base_props`` sums to 1,
so the genome-ploidy normalizer is implicitly 1 and a diploid genome sits at
``mu = T_i * base_props[g]``.
"""

import math

import numpy as np
from scipy import sparse
from scipy.special import gammaln
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def _nb_dense(log_e, T, base, rdr_k, invphi_k, log_invphi_k, eps):
    """Add the NB ``x=0`` baseline ``invphi*(log invphi - log(invphi+mu))`` over
    every ``(i, g, k)``; ``mu = T_i * base_g * rdr_k``. Parallel over cells."""
    n, G, K = log_e.shape
    for i in prange(n):
        Ti = T[i]
        for g in range(G):
            tb = Ti * base[g]
            for k in range(K):
                mu = tb * rdr_k[k]
                if mu < eps:
                    mu = eps
                log_e[i, g, k] += invphi_k[k] * (
                    log_invphi_k[k] - math.log(invphi_k[k] + mu)
                )


@njit(parallel=True, fastmath=True, cache=True)
def _nb_sparse(
    log_e, gx, nx, xval, logfact, invphi_k, lg_invphi_k, T, base, rdr_k, eps
):
    """Add the NB ``x>0`` adjustment at the X nonzeros (each ``(g, i)`` unique →
    race-free). Parallel over nonzeros."""
    nnz = gx.shape[0]
    K = invphi_k.shape[0]
    for e in prange(nnz):
        g = gx[e]
        i = nx[e]
        x = xval[e]
        lf = logfact[e]
        tb = T[i] * base[g]
        for k in range(K):
            mu = tb * rdr_k[k]
            if mu < eps:
                mu = eps
            ip = invphi_k[k]
            log_e[i, g, k] += (
                math.lgamma(x + ip)
                - lg_invphi_k[k]
                - lf
                + x * (math.log(mu) - math.log(ip + mu))
            )


@njit(parallel=True, fastmath=True, cache=True)
def _bb_kernel(
    log_e,
    cg,
    cn,
    ctot,
    bval,
    aval,
    comb,
    tau_k,
    a1,
    b1,
    a0,
    b0,
    const1,
    const0,
    Hg,
    bb0_out,
    bb1_out,
):
    """BB log-PMF for both orientations at the C nonzeros; write the H-selected
    value into ``log_e`` and store both into ``bb0_out``/``bb1_out``.

    ``v_h = comb + lgamma(b+alpha_h) + lgamma(a+beta_h) - lgamma(ctot+tau) +
    const_h`` where ``const_h = -(lgamma(alpha_h)+lgamma(beta_h)-lgamma(tau))``
    is precomputed per ``k``. Each ``(g, i)`` is unique → race-free. Parallel
    over nonzeros."""
    nnz = cg.shape[0]
    K = tau_k.shape[0]
    for e in prange(nnz):
        g = cg[e]
        i = cn[e]
        ce = ctot[e]
        be = bval[e]
        ae = aval[e]
        cm = comb[e]
        h = Hg[e]
        for k in range(K):
            lc = math.lgamma(ce + tau_k[k])
            v1 = cm + math.lgamma(be + a1[k]) + math.lgamma(ae + b1[k]) - lc + const1[k]
            v0 = cm + math.lgamma(be + a0[k]) + math.lgamma(ae + b0[k]) - lc + const0[k]
            bb1_out[e, k] = v1
            bb0_out[e, k] = v0
            log_e[i, g, k] += v1 if h == 1 else v0


def build_log_emissions(
    X_sub: sparse.csr_matrix,
    B_sub: sparse.csr_matrix,
    C_sub: sparse.csr_matrix,
    T_sub: np.ndarray,
    rdr_baf_cn: np.ndarray,
    rdr_baf_params: np.ndarray,
    base_props: np.ndarray,
    H: np.ndarray,
    fit_mode: str,
    eps: float = 1e-12,
    return_bb_orient: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple]:
    """Per-state log-emission tensor ``(n, G, K)`` for a chunk of ``n`` cells.

    ``X_sub`` / ``B_sub`` / ``C_sub`` are the chunk's ``(G, n)`` count matrices,
    ``T_sub`` the chunk's per-cell library sizes. ``rdr_baf_cn`` / ``rdr_baf_params``
    are per-state ``[rdr, baf]`` / ``[invphi, tau]`` ``(K, 2)`` tables. ``H`` is
    the ``(G,)`` shared phasing path (1=canonical, 0=flip).

    ``fit_mode``: ``allele_total`` (BB+NB), ``allele`` (BB), ``total`` (NB).

    If ``return_bb_orient``, also returns the BB nonzero pack
    ``(cg, cn, bb0, bb1)`` (or ``None`` when BB is disabled) for the H-update.
    """
    G, n = X_sub.shape
    K = rdr_baf_cn.shape[0]
    use_bb = fit_mode in ("allele_total", "allele")
    use_nb = fit_mode in ("allele_total", "total")

    rdr_k = np.ascontiguousarray(rdr_baf_cn[:, 0])
    baf_k = np.ascontiguousarray(rdr_baf_cn[:, 1])
    invphi_k = np.ascontiguousarray(rdr_baf_params[:, 0])
    tau_k = np.ascontiguousarray(rdr_baf_params[:, 1])
    base = np.ascontiguousarray(base_props, dtype=np.float64)
    T = np.ascontiguousarray(T_sub, dtype=np.float64)

    log_e = np.zeros((n, G, K), dtype=np.float64)

    if use_nb:
        _nb_dense(log_e, T, base, rdr_k, invphi_k, np.log(invphi_k), eps)
        X_coo = X_sub.tocoo()
        gx = X_coo.row.astype(np.int64)
        nx = X_coo.col.astype(np.int64)
        xval = X_coo.data.astype(np.float64)
        logfact = gammaln(xval + 1)
        _nb_sparse(
            log_e,
            gx,
            nx,
            xval,
            logfact,
            invphi_k,
            gammaln(invphi_k),
            T,
            base,
            rdr_k,
            eps,
        )

    bb_pack = None
    if use_bb:
        C_coo = C_sub.tocoo()
        cg = C_coo.row.astype(np.int64)
        cn = C_coo.col.astype(np.int64)
        ctot = C_coo.data.astype(np.float64)
        bval = np.asarray(B_sub.tocsr()[cg, cn]).ravel().astype(np.float64)
        aval = ctot - bval
        comb = gammaln(ctot + 1) - gammaln(bval + 1) - gammaln(aval + 1)
        a1 = tau_k * baf_k
        b1 = tau_k * (1.0 - baf_k)
        a0 = b1.copy()  # tau * (1 - baf)
        b0 = a1.copy()  # tau * baf
        const1 = -(gammaln(a1) + gammaln(b1) - gammaln(tau_k))
        const0 = -(gammaln(a0) + gammaln(b0) - gammaln(tau_k))
        Hg = H[cg].astype(np.int64)
        nnz = cg.shape[0]
        bb0 = np.empty((nnz, K), dtype=np.float64)
        bb1 = np.empty((nnz, K), dtype=np.float64)
        _bb_kernel(
            log_e,
            cg,
            cn,
            ctot,
            bval,
            aval,
            comb,
            tau_k,
            a1,
            b1,
            a0,
            b0,
            const1,
            const0,
            Hg,
            bb0,
            bb1,
        )
        if return_bb_orient:
            bb_pack = (cg, cn, bb0, bb1)

    if return_bb_orient:
        return log_e, bb_pack
    return log_e
