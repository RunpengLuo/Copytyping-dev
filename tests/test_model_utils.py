"""Unit tests for copytyping.inference.model_utils."""

import numpy as np
import pytest

from copytyping.inference.model_utils import (
    compute_baseline_proportions,
    clone_rdr_gk,
    clone_pi_gk,
    empirical_baf_gn,
)


# ---------------------------------------------------------------------------
# compute_baseline_proportions
# ---------------------------------------------------------------------------

def test_compute_baseline_proportions_shape(rng):
    G, N = 10, 8
    X = rng.integers(0, 20, size=(G, N)).astype(np.float64)
    T = X.sum(axis=0)
    normal_labels = np.array([True, True, False, True, False, True, False, False])
    result = compute_baseline_proportions(X, T, normal_labels)
    assert result.shape == (G,)


def test_compute_baseline_proportions_values(rng):
    G, N = 10, 8
    X = rng.integers(1, 20, size=(G, N)).astype(np.float64)
    T = X.sum(axis=0)
    normal_labels = np.array([True, True, False, True, False, True, False, False])
    result = compute_baseline_proportions(X, T, normal_labels)
    assert np.all(result >= 0)
    # proportions should roughly sum to 1 (they're normalized by total library size)
    assert abs(result.sum() - 1.0) < 0.05


# ---------------------------------------------------------------------------
# clone_rdr_gk
# ---------------------------------------------------------------------------

def test_clone_rdr_gk_shape(rng):
    G, K = 10, 3
    lambda_g = rng.random(size=G) * 0.01 + 1e-4
    C = rng.integers(1, 5, size=(G, K)).astype(np.float64)
    result = clone_rdr_gk(lambda_g, C)
    assert result.shape == (G, K)


def test_clone_rdr_gk_normalization(rng):
    """lambda_g dot rdr_gk[:, k] should equal 1.0 for each clone k."""
    G, K = 10, 3
    lambda_g = rng.random(size=G) * 0.01 + 1e-4
    C = rng.integers(1, 5, size=(G, K)).astype(np.float64)
    rdr = clone_rdr_gk(lambda_g, C)
    for k in range(K):
        dot = np.dot(lambda_g, rdr[:, k])
        assert abs(dot - 1.0) < 1e-6, f"clone {k}: dot={dot}"


# ---------------------------------------------------------------------------
# clone_pi_gk
# ---------------------------------------------------------------------------

def test_clone_pi_gk_shape(rng):
    G, K = 10, 3
    lambda_g = rng.random(size=G) * 0.01 + 1e-4
    C = rng.integers(1, 5, size=(G, K)).astype(np.float64)
    result = clone_pi_gk(lambda_g, C)
    assert result.shape == (G, K)


def test_clone_pi_gk_column_sum(rng):
    """Each column of pi_gk should sum to 1."""
    G, K = 10, 3
    lambda_g = rng.random(size=G) * 0.01 + 1e-4
    C = rng.integers(1, 5, size=(G, K)).astype(np.float64)
    pi = clone_pi_gk(lambda_g, C)
    col_sums = pi.sum(axis=0)
    assert np.allclose(col_sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# empirical_baf_gn
# ---------------------------------------------------------------------------

def test_empirical_baf_gn_shape(rng):
    G, N = 10, 8
    D = rng.integers(1, 30, size=(G, N)).astype(np.float64)
    Y = (rng.random(size=(G, N)) * D).astype(np.float64)
    result = empirical_baf_gn(Y, D)
    assert result.shape == (G, N)


def test_empirical_baf_gn_nan_where_zero_depth(rng):
    G, N = 6, 5
    D = rng.integers(1, 20, size=(G, N)).astype(np.float64)
    Y = (rng.random(size=(G, N)) * D).astype(np.float64)
    # Force some cells to have zero depth in row 0
    D[0, :2] = 0
    Y[0, :2] = 0
    result = empirical_baf_gn(Y, D)
    assert np.all(np.isnan(result[0, :2]))
    assert np.all(np.isfinite(result[0, 2:]))
