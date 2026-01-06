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


def prepare_rdr_baf_features(sx_data: SX_Data, base_props: np.ndarray, norm=True):
    # allele data
    Y = sx_data.Y
    D = sx_data.D
    baf_matrix = np.divide(
        Y, D, out=np.full_like(D, fill_value=np.nan, dtype=np.float32), where=D > 0
    )
    baf_matrix = baf_matrix.T
    baf_masks = sx_data.MASK["IMBALANCED"]
    baf_matrix = baf_matrix[:, baf_masks]
    perc_nan_baf = np.round(np.sum(np.isnan(baf_matrix)) / baf_matrix.size, 3)
    print(f"perc.nan.baf={perc_nan_baf:.3%}")

    # rdr data
    X = sx_data.X
    T = sx_data.T
    rdr_masks = sx_data.MASK["ANEUPLOID"]
    rdr_denom = base_props[:, None] @ T[None, :]  # (G, N)
    rdr_matrix = np.divide(
        X,
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
