"""M-step updates for the CNP-HMM: MAP transition matrices, initial state
distribution, and per-state dispersion MLEs (BB ``tau`` / NB ``invphi``) from
the hard per-(cell, segment) state assignment.
"""

import numpy as np
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.special import betaln, gammaln

from copytyping.cnphmm_inference.states import prior_mean_transitions


def update_transitions(Xi: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """MAP transition update ``A[c, c'] = (Xi + alpha - 1)_+ / row-sum``.

    Negative numerators (when ``alpha < 1``) are clamped to 0; rows with no
    surviving mass fall back to the Dirichlet prior mean. ``Xi`` / ``alpha`` are
    ``(G-1, K, K)``; returns ``A`` of the same shape.
    """
    num = np.clip(Xi + alpha - 1.0, 0.0, None)
    rowsum = num.sum(axis=2, keepdims=True)
    prior_mean = prior_mean_transitions(alpha)
    A = np.where(rowsum > 0, num / np.where(rowsum > 0, rowsum, 1.0), prior_mean)
    return A


def update_pi(pi_acc: np.ndarray, pi_alpha: float, eps: float = 1e-10) -> np.ndarray:
    """MAP initial-state update under a symmetric ``Dir(pi_alpha)`` prior:
    ``pi_c ∝ (pi_acc_c + pi_alpha - 1)_+``. ``pi_acc`` is the (soft or hard)
    first-segment state count.
    """
    num = np.clip(pi_acc + pi_alpha - 1.0, eps, None)
    return num / num.sum()


def transition_logprior(A: np.ndarray, alpha: np.ndarray, eps: float = 1e-12) -> float:
    """Dirichlet log-prior ``sum_g sum_c (alpha - 1) . log A`` (up to a constant),
    for the complete-data MAP objective.
    """
    return float(((alpha - 1.0) * np.log(np.clip(A, eps, None))).sum())


def transition_entropy(
    A: np.ndarray,
    Xi: np.ndarray | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Normalized transition entropy of a ``(G-1, K, K)`` transition stack.

    Each row's entropy ``-sum_c' A log A`` is normalized by ``log K`` to [0, 1]
    (0 = deterministic, 1 = uniform). The per-segment value is the mean over
    from-states ``c``; if ``Xi`` (the transition posterior / hard counts) is
    given, rows are weighted by their outgoing data mass so only the states
    cells actually occupy at that segment contribute. Returns
    ``(seg_entropy (G-1,), global_mean)`` where ``global_mean`` is the unweighted
    mean over all rows (a single scalar for tracking EM convergence).
    """
    K = A.shape[-1]
    row_H = -(A * np.log(np.clip(A, eps, None))).sum(axis=-1) / np.log(K)  # (G-1, K)
    if Xi is not None:
        w = Xi.sum(axis=2)  # (G-1, K) outgoing mass per from-state
        wsum = w.sum(axis=1, keepdims=True)
        w = np.where(wsum > 0, w / np.clip(wsum, eps, None), 1.0 / K)
        seg = (w * row_H).sum(axis=1)
    else:
        seg = row_H.mean(axis=1)
    return seg, float(row_H.mean())


def _neg_Q_bb(
    log_tau: float,
    b: np.ndarray,
    a: np.ndarray,
    comb: np.ndarray,
    baf: np.ndarray,
) -> float:
    tau = np.exp(log_tau)
    alpha = tau * baf
    beta = tau * (1.0 - baf)
    return -float((comb + betaln(b + alpha, a + beta) - betaln(alpha, beta)).sum())


def update_dispersions_hard(
    Zhat: np.ndarray,
    X: sparse.csr_matrix,
    B: sparse.csr_matrix,
    C: sparse.csr_matrix,
    T: np.ndarray,
    rdr_baf_cn: np.ndarray,
    rdr_baf_params: np.ndarray,
    base_props: np.ndarray,
    H: np.ndarray,
    em_kwargs: dict,
) -> None:
    """Per-state 1-D bounded MLE for BB ``tau`` and NB ``invphi`` from the hard
    per-(cell, segment) state ``Zhat`` (N, G). Mutates ``rdr_baf_params`` in
    place. Honors ``update_tau`` / ``update_invphi`` flags; a no-op otherwise.

    Each nonzero (BB) / each assigned pair (NB) is attributed to its own
    ``Zhat[i, g]`` state, so unlike the per-clone version the NB normalization
    set is an explicit pair list, not a Cartesian grid.
    """
    update_tau = em_kwargs["update_tau"]
    update_invphi = em_kwargs["update_invphi"]
    eps = em_kwargs["eps"]

    if update_tau:
        C_coo = C.tocoo()
        gc, nc = C_coo.row, C_coo.col
        ctot = C_coo.data.astype(np.float64)
        bval = np.asarray(B.tocsr()[gc, nc]).ravel().astype(np.float64)
        aval = ctot - bval
        comb = gammaln(ctot + 1) - gammaln(bval + 1) - gammaln(aval + 1)
        state_nz = Zhat[nc, gc]
        baf_canon = rdr_baf_cn[:, 1]
        baf_eff = np.where(H[gc] == 1, baf_canon[state_nz], 1.0 - baf_canon[state_nz])
        log_bounds = (np.log(em_kwargs["min_tau"]), np.log(em_kwargs["max_tau"]))
        for state in np.unique(state_nz):
            m = state_nz == state
            res = minimize_scalar(
                lambda lt: _neg_Q_bb(lt, bval[m], aval[m], comb[m], baf_eff[m]),
                bounds=log_bounds,
                method="bounded",
            )
            rdr_baf_params[state, 1] = float(np.exp(np.clip(res.x, *log_bounds)))

    if update_invphi:
        N, G = Zhat.shape
        rdr_state = rdr_baf_cn[:, 0]
        Zf = Zhat.reshape(-1)

        X_coo = X.tocoo()
        gx, nx = X_coo.row, X_coo.col
        xval = X_coo.data.astype(np.float64)
        logfact = gammaln(xval + 1)
        state_read = Zhat[nx, gx]
        bounds_ip = (em_kwargs["min_invphi"], em_kwargs["max_invphi"])

        for state in np.unique(Zf):
            rs = float(rdr_state[state])
            if rs == 0.0:
                continue
            # assigned (i, g) pairs for this state, computed without the 3x N*G
            # Cartesian arrays: i = idx // G, g = idx % G
            idx = np.flatnonzero(Zf == state)
            mu_p = np.clip(T[idx // G] * base_props[idx % G] * rs, eps, None)
            rmask = state_read == state
            x_s = xval[rmask]
            lf_s = logfact[rmask]
            mu_r = np.clip(T[nx[rmask]] * base_props[gx[rmask]] * rs, eps, None)

            def neg_Q(invphi: float) -> float:
                zero = (invphi * (np.log(invphi) - np.log(invphi + mu_p))).sum()
                adj = (
                    gammaln(x_s + invphi)
                    - gammaln(invphi)
                    - lf_s
                    + x_s * (np.log(mu_r) - np.log(invphi + mu_r))
                ).sum()
                return -float(zero + adj)

            res = minimize_scalar(neg_Q, bounds=bounds_ip, method="bounded")
            rdr_baf_params[state, 0] = float(np.clip(res.x, *bounds_ip))
