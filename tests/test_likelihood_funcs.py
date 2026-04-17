"""Unit tests for copytyping.inference.likelihood_funcs."""

import numpy as np
import pytest
from scipy.special import gammaln

from copytyping.inference.model_utils import (
    cond_betabin_logpmf,
    cond_negbin_logpmf,
    cond_betabin_logpmf_theta,
    cond_negbin_logpmf_theta,
    mle_invphi,
    mle_tau,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bb_inputs(rng):
    G, N, K = 8, 6, 3
    D = rng.integers(5, 30, size=(G, N)).astype(np.float64)
    Y = (rng.random(size=(G, N)) * D).astype(np.float64)
    tau = np.full(G, 50.0)
    p = rng.random(size=(G, K)) * 0.9 + 0.05  # (0.05, 0.95)
    return Y, D, tau, p


@pytest.fixture
def nb_inputs(rng):
    G, N, K = 8, 6, 3
    X = rng.integers(0, 20, size=(G, N)).astype(np.float64)
    T = rng.integers(50, 200, size=N).astype(np.float64)
    pi_gk = rng.dirichlet(np.ones(K), size=G).astype(np.float64)  # rows sum to 1
    inv_phi = np.full(G, 1 / 30.0)
    return X, T, pi_gk, inv_phi


# ---------------------------------------------------------------------------
# cond_betabin_logpmf
# ---------------------------------------------------------------------------


def test_cond_betabin_logpmf_shape(bb_inputs):
    Y, D, tau, p = bb_inputs
    G, N = Y.shape
    K = p.shape[1]
    ll = cond_betabin_logpmf(Y, D, tau, p)
    assert ll.shape == (G, N, K)


def test_cond_betabin_logpmf_values_not_nan(bb_inputs):
    Y, D, tau, p = bb_inputs
    ll = cond_betabin_logpmf(Y, D, tau, p)
    assert np.all(np.isfinite(ll))


def test_cond_betabin_logpmf_uniform_p(rng):
    """With p=0.5 and Y=D/2, the likelihood is the same for all clones at each bin/cell."""
    G, N, K = 5, 4, 3
    D = np.full((G, N), 10.0)
    Y = np.full((G, N), 5.0)
    tau = np.full(G, 50.0)
    p = np.full((G, K), 0.5)
    ll = cond_betabin_logpmf(Y, D, tau, p)
    # All clones should yield the same log-likelihood at every (g, n)
    assert np.allclose(ll[:, :, 0], ll[:, :, 1], atol=1e-6)
    assert np.allclose(ll[:, :, 0], ll[:, :, 2], atol=1e-6)


# ---------------------------------------------------------------------------
# cond_negbin_logpmf
# ---------------------------------------------------------------------------


def test_cond_negbin_logpmf_shape(nb_inputs):
    X, T, pi_gk, inv_phi = nb_inputs
    G, N = X.shape
    K = pi_gk.shape[1]
    ll = cond_negbin_logpmf(X, T, pi_gk, inv_phi)
    assert ll.shape == (G, N, K)


def test_cond_negbin_logpmf_values_not_nan(nb_inputs):
    X, T, pi_gk, inv_phi = nb_inputs
    ll = cond_negbin_logpmf(X, T, pi_gk, inv_phi)
    assert np.all(np.isfinite(ll))


# ---------------------------------------------------------------------------
# theta variants
# ---------------------------------------------------------------------------


def test_cond_betabin_logpmf_theta_shape(rng):
    G, N, K = 6, 5, 3
    D = rng.integers(5, 30, size=(G, N)).astype(np.float64)
    Y = (rng.random(size=(G, N)) * D).astype(np.float64)
    tau = np.full(G, 50.0)
    p = rng.random(size=(G, K)) * 0.8 + 0.1
    rdrs_gk = np.ones((G, K))
    theta = rng.random(size=N) * 0.8 + 0.1
    ll = cond_betabin_logpmf_theta(Y, D, tau, p, rdrs_gk, theta)
    assert ll.shape == (G, N, K)


def test_cond_negbin_logpmf_theta_shape(rng):
    G, N, K = 6, 5, 3
    X = rng.integers(0, 20, size=(G, N)).astype(np.float64)
    T = rng.integers(50, 200, size=N).astype(np.float64)
    lam_g = rng.random(size=G) * 0.01 + 1e-4
    inv_phi = np.full(G, 1 / 30.0)
    rdrs_gk = np.ones((G, K))
    theta = rng.random(size=N) * 0.8 + 0.1
    ll = cond_negbin_logpmf_theta(X, T, lam_g, inv_phi, rdrs_gk, theta)
    assert ll.shape == (G, N, K)


# ---------------------------------------------------------------------------
# mle_invphi
# ---------------------------------------------------------------------------


def test_mle_invphi_returns_scalar(rng):
    G, N, K = 5, 20, 3
    X_gnk = rng.integers(0, 15, size=(G, N, K)).astype(np.float64)
    mu_gnk = np.clip(rng.random(size=(G, N, K)) * 5 + 0.5, 1e-6, None)
    weights = np.ones((1, N, 1)) / (N * K)
    bounds = (1 / 100, 1 / 10)
    result = mle_invphi(X_gnk, mu_gnk, weights, bounds)
    assert isinstance(result, float)
    assert bounds[0] <= result <= bounds[1]


def test_mle_tau_returns_scalar(rng):
    G, N, K = 5, 20, 3
    D_gnk = rng.integers(5, 30, size=(G, N, K)).astype(np.float64)
    Y_gnk = (rng.random(size=(G, N, K)) * D_gnk).astype(np.float64)
    p_gnk = rng.random(size=(G, 1, K)) * 0.8 + 0.1
    weights = np.ones((1, N, 1)) / (N * K)
    logtau_bounds = (np.log(50), np.log(200))
    result = mle_tau(Y_gnk, D_gnk, p_gnk, weights, logtau_bounds)
    assert isinstance(result, float)
    assert np.exp(logtau_bounds[0]) <= result <= np.exp(logtau_bounds[1])


def test_mle_invphi_improves_likelihood(rng):
    """Fitted inv_phi should yield a higher (less negative) log-likelihood than the initial."""
    G, N, K = 4, 30, 2
    inv_phi_init = 1 / 30.0
    X_gnk = rng.integers(0, 20, size=(G, N, K)).astype(np.float64)
    mu_gnk = np.clip(rng.random(size=(G, N, K)) * 3 + 0.5, 1e-6, None)
    weights = np.ones((1, N, 1)) / (N * K)
    bounds = (1 / 100, 1 / 10)

    inv_phi_fitted = mle_invphi(X_gnk, mu_gnk, weights, bounds)

    def neg_ll(invphi):
        ll = (
            gammaln(X_gnk + invphi)
            - gammaln(invphi)
            - gammaln(X_gnk + 1.0)
            + invphi * np.log(invphi / (invphi + mu_gnk))
            + X_gnk * np.log(mu_gnk / (invphi + mu_gnk))
        )
        return -np.sum(weights * ll)

    ll_init = neg_ll(inv_phi_init)
    ll_fitted = neg_ll(inv_phi_fitted)
    # fitted should be at least as good (lower neg-ll)
    assert ll_fitted <= ll_init + 1e-6
