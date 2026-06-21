import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData


##################################################
# cell-type annotation
##################################################


def annotate_adata_celltype(
    adata: "AnnData",
    cell_type_df: pd.DataFrame,
    ref_label: str,
    assay_type: str,
) -> None:
    """Add cell_type annotations to adata.obs from cell_type_df."""
    ct_map = cell_type_df.set_index("BARCODE")[ref_label]
    if ref_label in adata.obs.columns:
        logging.warning(
            f"overwriting existing '{ref_label}' column "
            f"in {assay_type} h5ad obs with cell_type_df"
        )
    adata.obs[ref_label] = (
        adata.obs_names.to_series().map(ct_map).fillna("Unknown").values
    )


##################################################
# LOH BAF
##################################################


def compute_loh_baf(
    ballele_counts: np.ndarray,
    total_allele_counts: np.ndarray,
    cn_A: np.ndarray,
    cn_B: np.ndarray,
    clones: list[str],
) -> tuple[np.ndarray, list[tuple[str, list[str]]]]:
    """Per-spot aggregated BAF over clone-specific LOH clusters.

    Args:
        ballele_counts: (G, N) cluster-level B-allele counts.
        total_allele_counts: (G, N) cluster-level total-allele counts (A + B).
        cn_A/cn_B: (G, K) per-clone copy numbers.
        clones: clone names, length K.

    Returns (baf_array, loh_info) where:
        baf_array: float (N, K_tumor) — per-spot BAF aggregated over LOH clusters of each tumor clone.
            NaN if no allele coverage or no LOH clusters for that clone.
        loh_info: list of (clone_name, list of "cluster <tab> clone states") per clone with LOH.
    """
    num_clones = len(clones)
    num_cells = ballele_counts.shape[1]
    K_tumor = num_clones - 1
    baf = np.full((num_cells, K_tumor), np.nan)
    loh_info = []

    for ki in range(K_tumor):
        k = ki + 1  # skip normal
        clone = clones[k]
        loh_mask = (cn_B[:, k] == 0) & (cn_A[:, k] > 0)
        if loh_mask.sum() == 0:
            continue

        entries = []
        for gi in np.where(loh_mask)[0]:
            cn_parts = [
                f"{clones[j]}={cn_A[gi, j]}|{cn_B[gi, j]}" for j in range(num_clones)
            ]
            entries.append(f"cluster{gi}\t{', '.join(cn_parts)}")
        loh_info.append((clone, entries))

        Y_loh = ballele_counts[loh_mask].sum(axis=0).astype(float)
        D_loh = total_allele_counts[loh_mask].sum(axis=0).astype(float)
        valid = D_loh > 0
        baf[valid, ki] = Y_loh[valid] / D_loh[valid]

        logging.info(f"LOH clusters for {clone} ({int(loh_mask.sum())} clusters):")
        for entry in entries:
            logging.info(f"  {entry}")

    return baf, loh_info
