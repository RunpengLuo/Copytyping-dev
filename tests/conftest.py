"""Shared fixtures for copytyping unit tests."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def make_sx_data_mock(rng):
    """Factory that returns a MagicMock mimicking SX_Data with (G=10, N=5, K=3) arrays."""

    def _make(G=10, N=5, K=3):
        mock = MagicMock()

        mock.G = G
        mock.N = N
        mock.K = K

        # Clone names: "normal", "clone1", "clone2"
        mock.clones = ["normal"] + [f"clone{i}" for i in range(1, K)]

        # Copy number arrays (G, K): normal clone is diploid (1,1,2), others vary
        A = np.ones((G, K), dtype=np.float32)
        B = np.ones((G, K), dtype=np.float32)
        # Make some bins aneuploid/imbalanced for clones 1..K-1
        A[3:6, 1:] = 2.0
        B[3:6, 1:] = 0.0
        A[6:8, 1:] = 3.0
        B[6:8, 1:] = 1.0
        C = A + B

        mock.A = A
        mock.B = B
        mock.C = C

        # BAF = B / (A + B), with laplace smoothing 0.01
        laplace = 0.01
        BAF = (B + laplace) / (C + 2 * laplace)
        mock.BAF = BAF

        # Count matrices (G, N)
        X = rng.integers(0, 20, size=(G, N)).astype(np.int32)
        D = rng.integers(1, 30, size=(G, N)).astype(np.int32)
        Y = (rng.random(size=(G, N)) * D).astype(np.int32)
        mock.X = X
        mock.Y = Y
        mock.D = D

        # Library sizes
        mock.T = X.sum(axis=0).astype(np.float32)

        # Build MASK the same way get_cnp_mask would
        ai_mask = np.any(A != B, axis=1)
        c_mask = np.any(C != 2, axis=1)
        tumor_mask = ai_mask | c_mask
        neutral_mask = ~tumor_mask

        clonal_loh_mask = np.all(B[:, 1:] == 0, axis=1) & np.all(A[:, 1:] > 0, axis=1)
        clonal_loh_mask |= np.all(A[:, 1:] == 0, axis=1) & np.all(B[:, 1:] > 0, axis=1)
        subclonal_loh_mask = np.any(B[:, 1:] == 0, axis=1) & np.all(A[:, 1:] > 0, axis=1)
        subclonal_loh_mask |= np.any(A[:, 1:] == 0, axis=1) & np.all(B[:, 1:] > 0, axis=1)
        subclonal_mask = np.copy(tumor_mask)
        if K > 2:
            subclonal_mask = np.any(A[:, 2:] != A[:, 1][:, None], axis=1) | np.any(
                B[:, 2:] != B[:, 1][:, None], axis=1
            )

        mock.MASK = {
            "CNP": tumor_mask,
            "IMBALANCED": ai_mask,
            "ANEUPLOID": c_mask,
            "SUBCLONAL": subclonal_mask,
            "CLONAL_LOH": clonal_loh_mask,
            "SUBCLONAL_LOH": subclonal_loh_mask,
            "NEUTRAL": neutral_mask,
        }
        mock.nrows_imbalanced = int(np.sum(ai_mask))
        mock.nrows_aneuploid = int(np.sum(c_mask))

        # Wire apply_mask_shallow to behave like the real method
        def _apply_mask_shallow(mask_id="CNP", additional_mask=None):
            if additional_mask is None:
                additional_mask = np.ones(G, dtype=bool)
            mask = mock.MASK[mask_id] & additional_mask
            M = {
                "A": A[mask, :],
                "B": B[mask, :],
                "C": C[mask, :],
                "BAF": BAF[mask, :],
                "X": X[mask, :],
                "Y": Y[mask, :],
                "D": D[mask, :],
                "cnv_blocks": mask,  # simplified stand-in
            }
            return M, mask

        mock.apply_mask_shallow = _apply_mask_shallow

        return mock

    return _make
