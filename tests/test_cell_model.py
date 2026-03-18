"""Unit tests for copytyping.inference.cell_model.Cell_Model."""

import numpy as np
import pandas as pd
import pytest

from copytyping.inference.cell_model import Cell_Model


# ---------------------------------------------------------------------------
# Helper: build a Cell_Model from a mock SX_Data
# ---------------------------------------------------------------------------

def _build_cell_model(make_sx_data_mock, assay_type="scRNA"):
    sx = make_sx_data_mock(G=10, N=8, K=3)
    barcodes = pd.DataFrame({"BARCODE": [f"bc{i}" for i in range(sx.N)]})
    model = Cell_Model(
        barcodes=barcodes,
        assay_type=assay_type,
        data_types=[assay_type],
        data_sources={assay_type: sx},
        work_dir=None,
        verbose=0,
    )
    return model, sx


def _allele_only_params(model, sx):
    """Return a minimal params dict for allele_only mode."""
    return {
        "pi": np.ones(model.K) / model.K,
        f"{model.assay_type}-tau": np.full(sx.nrows_imbalanced, 50.0, dtype=np.float32),
    }


def _hybrid_params(model, sx):
    """Return a minimal params dict for hybrid mode."""
    lambda_g = sx.X.sum(axis=1) / (sx.T.sum() + 1e-12)
    return {
        "pi": np.ones(model.K) / model.K,
        f"{model.assay_type}-tau": np.full(sx.nrows_imbalanced, 50.0, dtype=np.float32),
        f"{model.assay_type}-inv_phi": np.full(sx.nrows_aneuploid, 1 / 30.0, dtype=np.float32),
        f"{model.assay_type}-lambda": lambda_g.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# E-step
# ---------------------------------------------------------------------------

def test_cell_model_e_step_shape(make_sx_data_mock):
    model, sx = _build_cell_model(make_sx_data_mock)
    params = _allele_only_params(model, sx)
    gamma = model._e_step("allele_only", params)
    assert gamma.shape == (model.N, model.K)


def test_cell_model_e_step_rows_sum_to_one(make_sx_data_mock):
    model, sx = _build_cell_model(make_sx_data_mock)
    params = _allele_only_params(model, sx)
    gamma = model._e_step("allele_only", params)
    row_sums = gamma.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# compute_log_likelihood
# ---------------------------------------------------------------------------

def test_cell_model_compute_log_likelihood_allele_only(make_sx_data_mock):
    model, sx = _build_cell_model(make_sx_data_mock)
    params = _allele_only_params(model, sx)
    ll, log_marg, global_lls = model.compute_log_likelihood("allele_only", params)
    assert np.isfinite(ll)
    assert np.all(np.isfinite(log_marg))


def test_cell_model_compute_log_likelihood_hybrid(make_sx_data_mock):
    model, sx = _build_cell_model(make_sx_data_mock)
    params = _hybrid_params(model, sx)
    ll, log_marg, global_lls = model.compute_log_likelihood("hybrid", params)
    assert np.isfinite(ll)
    assert np.all(np.isfinite(log_marg))


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def test_cell_model_fit_runs_allele_only(make_sx_data_mock):
    model, sx = _build_cell_model(make_sx_data_mock)
    init_params = {"tau0": 50.0, "phi0": 30.0}
    fix_params = {"pi": True}  # keep pi fixed to avoid updating
    params = model.fit(
        fit_mode="allele_only",
        fix_params=fix_params,
        init_params=init_params,
        share_params={},
        max_iter=5,
    )
    assert isinstance(params, dict)
    assert "pi" in params
    assert f"{model.assay_type}-tau" in params


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def test_cell_model_predict_labels(make_sx_data_mock):
    model, sx = _build_cell_model(make_sx_data_mock)
    params = _allele_only_params(model, sx)
    anns, clone_props = model.predict("allele_only", params, label="label")

    assert isinstance(anns, pd.DataFrame)
    assert "label" in anns.columns
    assert "max_posterior" in anns.columns
    assert "margin_delta" in anns.columns

    valid_labels = set(model.clones) | {"NA"}
    assert set(anns["label"].unique()).issubset(valid_labels)
    assert len(anns) == model.N
