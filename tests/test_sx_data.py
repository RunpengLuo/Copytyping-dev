"""Unit tests for copytyping.sx_data.sx_data helpers."""

import numpy as np
import pytest

from copytyping.sx_data.sx_data import get_cnp_mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_acb(G, K, rng, diploid=False):
    """Return A, B, C arrays of shape (G, K)."""
    if diploid:
        A = np.ones((G, K), dtype=np.float32)
        B = np.ones((G, K), dtype=np.float32)
    else:
        A = rng.integers(0, 4, size=(G, K)).astype(np.float32) + 1
        B = rng.integers(0, 4, size=(G, K)).astype(np.float32)
    C = A + B
    return A, B, C


# ---------------------------------------------------------------------------
# get_cnp_mask
# ---------------------------------------------------------------------------

def test_get_cnp_mask_keys(rng):
    G, K = 10, 3
    A, B, C = _make_acb(G, K, rng)
    mask = get_cnp_mask(A, B, C)
    expected_keys = {"CNP", "IMBALANCED", "CLONAL_IMBALANCED", "ANEUPLOID", "SUBCLONAL", "CLONAL_LOH", "SUBCLONAL_LOH", "NEUTRAL"}
    assert set(mask.keys()) == expected_keys


def test_get_cnp_mask_neutral():
    """Diploid bins (A=B=1, C=2) across all clones → NEUTRAL=True, IMBALANCED=False, ANEUPLOID=False."""
    G, K = 5, 3
    A = np.ones((G, K), dtype=np.float32)
    B = np.ones((G, K), dtype=np.float32)
    C = A + B
    mask = get_cnp_mask(A, B, C)
    assert np.all(mask["NEUTRAL"])
    assert np.all(~mask["IMBALANCED"])
    assert np.all(~mask["ANEUPLOID"])


def test_get_cnp_mask_imbalanced():
    """Bins with A≠B in at least one clone → IMBALANCED=True."""
    G, K = 5, 3
    A = np.ones((G, K), dtype=np.float32)
    B = np.ones((G, K), dtype=np.float32)
    C = A + B
    # Make rows 1 and 3 imbalanced in clone 1
    A[1, 1] = 2.0
    B[1, 1] = 0.0
    A[3, 2] = 3.0
    B[3, 2] = 1.0
    C = A + B
    mask = get_cnp_mask(A, B, C)
    assert mask["IMBALANCED"][1]
    assert mask["IMBALANCED"][3]
    # Rows 0, 2, 4 are still balanced
    for i in [0, 2, 4]:
        assert not mask["IMBALANCED"][i]


def test_get_cnp_mask_aneuploid():
    """Bins with total copy ≠ 2 in at least one clone → ANEUPLOID=True."""
    G, K = 5, 3
    A = np.ones((G, K), dtype=np.float32)
    B = np.ones((G, K), dtype=np.float32)
    C = A + B  # all diploid initially
    # Make row 2 aneuploid in clone 1 (C=3)
    A[2, 1] = 2.0
    C = A + B
    mask = get_cnp_mask(A, B, C)
    assert mask["ANEUPLOID"][2]
    # Rows 0,1,3,4 still diploid
    for i in [0, 1, 3, 4]:
        assert not mask["ANEUPLOID"][i]


def test_get_cnp_mask_shape(rng):
    G, K = 10, 3
    A, B, C = _make_acb(G, K, rng)
    mask = get_cnp_mask(A, B, C)
    for key, arr in mask.items():
        assert arr.shape == (G,), f"{key} has wrong shape"
        assert arr.dtype == bool, f"{key} is not bool"


# ---------------------------------------------------------------------------
# apply_mask_shallow (via mock SX_Data)
# ---------------------------------------------------------------------------

def test_apply_mask_shallow_keys(make_sx_data_mock):
    sx = make_sx_data_mock()
    M, mask = sx.apply_mask_shallow(mask_id="IMBALANCED")
    expected_keys = {"A", "B", "C", "BAF", "X", "Y", "D", "cnv_blocks"}
    assert set(M.keys()) == expected_keys


def test_apply_mask_shallow_filters_rows(make_sx_data_mock):
    sx = make_sx_data_mock()
    # The fixture has some imbalanced bins (rows 3-7), so mask should reduce rows
    M, mask = sx.apply_mask_shallow(mask_id="IMBALANCED")
    n_selected = int(np.sum(mask))
    assert 0 < n_selected < sx.G
    for key in ("A", "B", "C", "BAF", "X", "Y", "D"):
        assert M[key].shape[0] == n_selected
