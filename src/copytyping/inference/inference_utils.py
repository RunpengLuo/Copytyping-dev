import logging

import pandas as pd

from copytyping.utils import NA_CELLTYPE


def merge_celltype_into_barcodes(barcodes_df, cell_type_df, ref_label, data_type):
    """Merge cell_type annotations into barcodes DataFrame.

    Drops the ref_label column if all labels are uninformative (NA_CELLTYPE).
    """
    barcodes_df = pd.merge(
        left=barcodes_df,
        right=cell_type_df[["BARCODE", ref_label]],
        on="BARCODE",
        how="left",
        validate="1:1",
        sort=False,
    )
    barcodes_df[ref_label] = barcodes_df[ref_label].fillna("Unknown").astype(str)
    if barcodes_df[ref_label].isin(NA_CELLTYPE).all():
        logging.warning(
            f"all {data_type} barcodes have "
            f"uninformative {ref_label} labels "
            f"(all in NA_CELLTYPE={NA_CELLTYPE})"
        )
        barcodes_df = barcodes_df.drop(columns=[ref_label])
    return barcodes_df


def annotate_adata_celltype(adata, cell_type_df, ref_label, data_type):
    """Add cell_type annotations to adata.obs from cell_type_df."""
    ct_map = cell_type_df.set_index("BARCODE")[ref_label]
    if ref_label in adata.obs.columns:
        logging.warning(
            f"overwriting existing '{ref_label}' column "
            f"in {data_type} h5ad obs with cell_type_df"
        )
    adata.obs[ref_label] = (
        adata.obs_names.to_series().map(ct_map).fillna("Unknown").values
    )
