import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import cluster

from copytyping.inference.model_utils import compute_baseline_proportions
from copytyping.sx_data.sx_data import SX_Data
from copytyping.utils import is_tumor_label


def prepare_rdr_baf_features(sx_data: SX_Data, base_props: np.ndarray, norm=True):
    """Build cell x feature matrix from BAF + log2RDR at informative bins."""
    Y = sx_data.Y
    D = sx_data.D
    baf_matrix = np.divide(
        Y, D, out=np.full_like(D, fill_value=np.nan, dtype=np.float32), where=D > 0
    )
    baf_matrix = baf_matrix.T
    baf_masks = sx_data.MASK["IMBALANCED"]
    baf_matrix = baf_matrix[:, baf_masks]

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


def kmeans_copytyping(
    data_sources: dict, barcodes: pd.DataFrame, ref_label: str, K: int
):
    """K-means clustering on BAF+RDR features. Baseline from ref-label normals."""
    is_normal = ~barcodes[ref_label].apply(is_tumor_label).to_numpy()
    n_normal = int(is_normal.sum())
    if n_normal == 0:
        raise ValueError(f"kmeans requires ref normals but found 0 in {ref_label}")
    logging.info(f"kmeans: {n_normal} ref normals for baseline")

    features = []
    for data_type, sx in data_sources.items():
        base_props = compute_baseline_proportions(sx.X, sx.T, is_normal)
        my_features = prepare_rdr_baf_features(sx, base_props, norm=False)
        features.append(my_features)
    data_matrix = np.concatenate(features, axis=1)

    kmeans = cluster.KMeans(n_clusters=K, init="k-means++")
    kmeans.fit(data_matrix)
    return kmeans.labels_


def cluster_label_major_vote(
    anns: pd.DataFrame,
    cluster_labels: np.ndarray,
    cell_label="raw_label",
    ref_label="cell_type",
):
    """Assign cluster labels to normal or tumor based on majority vote vs ref_label."""
    anns[f"{cell_label}-raw"] = cluster_labels.astype(str)
    anns[cell_label] = "NA"
    tumor_id = 1
    for lab, anns_lab in anns.groupby(by=cluster_labels, sort=False):
        n_tumor = sum(is_tumor_label(x) for x in anns_lab[ref_label])
        n_normal = len(anns_lab) - n_tumor
        logging.info(f"cluster {lab}: normal={n_normal}, tumor={n_tumor}")
        if n_normal >= n_tumor:
            anns.loc[anns_lab.index, cell_label] = "normal"
        else:
            anns.loc[anns_lab.index, cell_label] = f"tumor{tumor_id}"
            tumor_id += 1

    clone_props = {
        lab: np.mean(anns[cell_label].to_numpy() == lab)
        for lab in sorted(anns[cell_label].unique())
    }
    return anns, clone_props
