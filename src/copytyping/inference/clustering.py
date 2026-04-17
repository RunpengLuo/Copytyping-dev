import numpy as np
import pandas as pd
from scipy import stats

import scanpy as sc
from scanpy import AnnData
from sklearn import cluster

from copytyping.utils import *
from copytyping.inference.cell_model import Cell_Model
from copytyping.sx_data.sx_data import SX_Data


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


def kmeans_copytyping(sc_model: Cell_Model, params: dict):
    features = []
    for data_type in sc_model.data_types:
        my_features = prepare_rdr_baf_features(
            sc_model.data_sources[data_type], params[f"{data_type}-lambda"], norm=False
        )
        features.append(my_features)
    data_matrix = np.concatenate(features, axis=1)

    ncenters = sc_model.num_clones
    kmeans = cluster.KMeans(n_clusters=ncenters, init="k-means++")
    kmeans.fit(data_matrix)
    return kmeans.labels_


def leiden_copytyping(
    sc_model: Cell_Model, params: dict, random_state=42, resolution=1
):
    features = []
    for data_type in sc_model.data_types:
        my_features = prepare_rdr_baf_features(
            sc_model.data_sources[data_type], params[f"{data_type}-lambda"], norm=True
        )
        features.append(my_features)
    data_matrix = np.concatenate(features, axis=1)

    adata = AnnData(X=data_matrix)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, metric="euclidean")
    sc.tl.leiden(adata, resolution=resolution, random_state=random_state)
    # sc.tl.umap(adata)
    return adata.obs["leiden"].to_numpy()


def ward_copytyping(
    sc_model: Cell_Model,
    params: dict,
    label="ward",
):
    features = []
    for data_type in sc_model.data_types:
        my_features = prepare_rdr_baf_features(
            sc_model.data_sources[data_type], params[f"{data_type}-lambda"], norm=False
        )
        features.append(my_features)
    data_matrix = np.concatenate(features, axis=1)

    ncenters = sc_model.num_clones

    hier = cluster.AgglomerativeClustering(
        n_clusters=ncenters,
        linkage="ward",
    )
    cluster_labels = hier.fit_predict(X=data_matrix)
    return cluster_labels


def cluster_label_major_vote(
    anns: pd.DataFrame,
    cluster_labels: np.ndarray,
    cell_label="raw_label",
    cell_type="cell_type",
):
    """
    assign cluster labels to normal or tumor based on majority vote wrt cell types.
    """
    anns[f"{cell_label}-raw"] = cluster_labels
    anns[f"{cell_label}-raw"] = anns[f"{cell_label}-raw"].astype("str")

    anns[cell_label] = "NA"
    tumor_id = 1
    for lab, anns_lab in anns.groupby(by=cluster_labels, sort=False):
        n_normal = np.sum(anns_lab[cell_type] != "Tumor_cell")
        n_tumor = len(anns_lab) - n_normal
        print(lab, n_normal, n_tumor)
        if n_normal > n_tumor:
            anns.loc[anns_lab.index, cell_label] = "normal"
        elif n_normal < n_tumor:
            anns.loc[anns_lab.index, cell_label] = f"tumor{tumor_id}"
            tumor_id += 1

    clone_props = {
        lab: np.mean(anns[cell_label].to_numpy() == lab)
        for lab in sorted(anns[cell_label].unique())
    }
    return anns, clone_props
