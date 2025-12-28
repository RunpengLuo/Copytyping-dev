import os
import sys

import numpy as np
import pandas as pd

import scanpy as sc
from scanpy import AnnData
from sklearn import cluster, mixture

from copytyping.utils import *
from copytyping.inference.cell_model import Cell_Model
from copytyping.sx_data.sx_data import SX_Data
from copytyping.inference.inference_utils import prepare_rdr_baf_features


def kmeans_copytyping(sc_model: Cell_Model, params: dict):
    features = []
    for data_type in sc_model.data_types:
        my_features = prepare_rdr_baf_features(
            sc_model.data_sources[data_type], params[f"{data_type}-lambda"], norm=False
        )
        features.append(my_features)
    data_matrix = np.concatenate(features, axis=1)

    anns = sc_model.barcodes.copy(deep=True)
    ncenters = sc_model.num_clones
    kmeans = cluster.KMeans(n_clusters=ncenters, init="k-means++")
    kmeans.fit(data_matrix)
    return kmeans.labels_


def leiden_copytyping(sc_model: Cell_Model, params: dict, random_state=42, resolution=1):
    features = []
    for data_type in sc_model.data_types:
        my_features = prepare_rdr_baf_features(
            sc_model.data_sources[data_type], params[f"{data_type}-lambda"], norm=True
        )
        features.append(my_features)
    data_matrix = np.concatenate(features, axis=1)

    anns = sc_model.barcodes.copy(deep=True)
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

    anns = sc_model.barcodes.copy(deep=True)
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
