import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import cluster

from copytyping.inference.count_data import Count_Data
from copytyping.inference.model_utils import compute_baseline_proportions
from copytyping.utils import is_tumor_label


def prepare_rdr_baf_features(
    count_data: Count_Data, base_props: np.ndarray, norm: bool = True
):
    """Build cell x feature matrix from BAF + log2RDR at informative bins."""
    count_B = np.asarray(count_data.count_B)
    count_N = np.asarray(count_data.count_C)
    baf_matrix = np.divide(
        count_B,
        count_N,
        out=np.full_like(count_N, fill_value=np.nan, dtype=np.float32),
        where=count_N > 0,
    )
    baf_matrix = baf_matrix.T
    baf_masks = count_data.allele_mask["IMBALANCED"]
    baf_matrix = baf_matrix[:, baf_masks]

    count_X = np.asarray(count_data.count_X)
    count_T = count_X.sum(axis=0)
    rdr_masks = count_data.total_mask["ANEUPLOID"]
    rdr_denom = base_props[:, None] @ count_T[None, :]  # (G, N)
    rdr_matrix = np.divide(
        count_X,
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
    data_sources: dict[str, Count_Data],
    barcodes: pd.DataFrame,
    ref_label: str,
    K: int,
    label: str,
):
    """K-means clustering on BAF+RDR features. Returns (anns, clone_props).

    If ref_label is in barcodes, clusters get majority-voted normal/tumor names
    via cluster_label_major_vote; otherwise raw "clusterN" names are kept.
    """
    is_normal = ~barcodes[ref_label].apply(is_tumor_label).to_numpy()
    n_normal = int(is_normal.sum())
    if n_normal == 0:
        raise ValueError(f"kmeans requires ref normals but found 0 in {ref_label}")
    logging.info(f"kmeans: {n_normal} ref normals for baseline")

    features = []
    for count_data in data_sources.values():
        count_X = np.asarray(count_data.count_X)
        base_props = compute_baseline_proportions(
            count_X, count_X.sum(axis=0), is_normal
        )
        features.append(prepare_rdr_baf_features(count_data, base_props, norm=False))
    data_matrix = np.concatenate(features, axis=1)

    kmeans = cluster.KMeans(n_clusters=K, init="k-means++")
    kmeans.fit(data_matrix)
    raw_labels = kmeans.labels_

    anns = barcodes.copy(deep=True)
    if ref_label in barcodes.columns:
        return cluster_label_major_vote(
            anns, raw_labels, cell_label=label, ref_label=ref_label
        )
    anns[label] = ["cluster" + str(x) for x in raw_labels]
    clone_props = {
        lab: np.mean(anns[label].to_numpy() == lab)
        for lab in sorted(anns[label].unique())
    }
    return anns, clone_props


def cluster_label_major_vote(
    anns: pd.DataFrame,
    cluster_labels: np.ndarray,
    cell_label: str = "raw_label",
    ref_label: str = "cell_type",
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
