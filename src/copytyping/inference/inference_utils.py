import os
import sys

import numpy as np
import pandas as pd

import scanpy as sc
from scanpy import AnnData
import squidpy as sq

from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy import sparse

from scipy import sparse, stats

from copytyping.utils import *
from copytyping.io_utils import *
from copytyping.external import *
from copytyping.sx_data.sx_data import SX_Data


def prepare_rdr_baf_features(
    sx_data: SX_Data, base_props: np.ndarray, aggregate_hb=True, norm=True
):
    # allele data
    Y = sx_data.Y
    D = sx_data.D
    baf_masks = sx_data.ALLELE_MASK["IMBALANCED"]
    # aggregate over CNV segments
    if aggregate_hb:
        hb = sx_data.bin_info["HB"].to_numpy()
        uniq_hb, inv = np.unique(hb, return_inverse=True)  # inv[g] in [0, K)
        Y_agg = np.zeros((len(uniq_hb), Y.shape[1]), dtype=Y.dtype)  # Y'
        D_agg = np.zeros((len(uniq_hb), D.shape[1]), dtype=D.dtype)  # D'
        np.add.at(Y_agg, inv, Y)
        np.add.at(D_agg, inv, D)
        Y = Y_agg
        D = D_agg
        baf_masks = mark_imbalanced(sx_data.bin_info.groupby(by="HB").first())
    baf_matrix = np.divide(
        Y, D, out=np.full_like(D, fill_value=np.nan, dtype=np.float32), where=D > 0
    )
    baf_matrix = baf_matrix.T
    baf_matrix = baf_matrix[:, baf_masks]
    perc_nan_baf = np.round(np.sum(np.isnan(baf_matrix)) / baf_matrix.size, 3)
    print(f"perc.nan.baf={perc_nan_baf:.3%}")

    # rdr data
    T = sx_data.T
    Tn = sx_data.Tn
    rdr_masks = sx_data.FEAT_MASK["ANEUPLOID"]
    # aggregate over CNV segments
    if aggregate_hb:
        hb = sx_data.feat_info["HB"].to_numpy()
        uniq_hb, inv = np.unique(hb, return_inverse=True)  # inv[g] in [0, K)
        T_agg = np.zeros((len(uniq_hb), Y.shape[1]), dtype=T.dtype)
        base_props_agg = np.zeros((len(uniq_hb), 1), dtype=base_props.dtype)
        np.add.at(T_agg, inv, T)
        np.add.at(base_props_agg, inv, base_props[:, None])
        T = T_agg
        base_props = base_props_agg.reshape(-1)
        rdr_masks = mark_aneuploid(sx_data.feat_info.groupby(by="HB").first())

    rdr_denom = base_props[:, None] @ Tn[None, :]  # (G, N)
    rdr_matrix = np.divide(
        T,
        rdr_denom,
        out=np.full_like(rdr_denom, fill_value=np.nan, dtype=np.float32),
        where=rdr_denom > 0,
    )
    rdr_matrix[rdr_matrix == 0] = np.nan
    rdr_matrix = rdr_matrix.T
    rdr_matrix[~np.isnan(rdr_matrix)] = np.log2(rdr_matrix[~np.isnan(rdr_matrix)])
    rdr_matrix = rdr_matrix[:, rdr_masks]
    perc_nan_rdr = np.round(np.sum(np.isnan(rdr_matrix)) / rdr_matrix.size, 3)
    print(f"perc.nan.rdr={perc_nan_rdr:.3%}")

    # normalize
    if norm:
        baf_matrix[~np.isnan(baf_matrix)] -= 0.5
        baf_z = stats.zscore(baf_matrix, axis=0, nan_policy="omit")
        baf_z = np.where(np.isnan(baf_z), 0.0, baf_z)
        rdr_z = stats.zscore(rdr_matrix, axis=0, nan_policy="omit")
        rdr_z = np.where(np.isnan(rdr_z), 0.0, rdr_z)
    else:
        baf_z = np.where(np.isnan(baf_matrix), 0.5, baf_matrix)
        rdr_z = np.where(np.isnan(rdr_matrix), 0.0, rdr_matrix)
    features = np.concatenate([baf_z, rdr_z], axis=1)
    return features
