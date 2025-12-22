import os
import sys

import pandas as pd
import numpy as np
import scanpy as sc
from scanpy import AnnData

# from matplotlib.colors import TwoSlopeNorm
# import matplotlib.colors as mcolors
# import seaborn as sns
# from scipy.cluster.hierarchy import linkage, leaves_list
# from matplotlib.collections import LineCollection
# from scipy import sparse, stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from copytyping.sx_data.sx_data import SX_Data
from copytyping.io_utils import *


def plot_umap_copynumber(
    sample: str,
    data_type: str,
    features: np.ndarray,
    anns: pd.DataFrame,
    lab_type="cell_label",
    cell_type="cell_type",
    figsize=(18, 6),
    filename=None,
    **kwargs,
):
    # build anndata
    adata = AnnData(X=features)
    adata.obs[lab_type] = anns[lab_type].to_numpy()
    adata.obs[cell_type] = anns[cell_type].to_numpy()
    if "max_posterior" in anns:
        adata.obs["max_posterior"] = anns["max_posterior"].tolist()
    else:
        adata.obs["max_posterior"] = 1.0

    sc.pp.pca(adata)
    sc.pp.neighbors(adata, metric="euclidean")
    sc.tl.umap(adata)

    with PdfPages(filename) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        sc.pl.umap(
            adata, color=cell_type, title=f"label={cell_type}", show=False, ax=axes[0]
        )
        sc.pl.umap(
            adata, color=lab_type, title=f"label={lab_type}", show=False, ax=axes[1]
        )
        sc.pl.umap(
            adata,
            color="max_posterior",
            title=f"colored by copytyping posterior probability",
            show=False,
            cmap="viridis",
            ax=axes[2],
        )
        plt.suptitle(f"{sample} {data_type} UMAP")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        pdf.close()
    return


# plot UMAP
def plot_umap_total_expression(
    sx_data: SX_Data,
    anns: pd.DataFrame,
    sample: str,
    data_type: str,
    mask_cnp=True,
    mask_id="CNP",
    lab_type="cell_label",
    figsize=(20, 10),
    filename=None,
    **kwargs,
):
    pass
