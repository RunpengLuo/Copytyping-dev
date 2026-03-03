import os
import sys
import shutil

import numpy as np
import pandas as pd

import scanpy as sc
import squidpy as sq

from copytyping.utils import *

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from copytyping.sx_data.sx_data import SX_Data
from copytyping.io_utils import load_visium_path_annotation
from copytyping.inference.validation import evaluate_malignant_accuracy


def set_normal_blue_unique(adata, col, blue="#1f77b4"):
    # make categorical
    adata.obs[col] = adata.obs[col].astype("category")
    cats = list(adata.obs[col].cat.categories)

    # base palette, then drop the blue from it
    base = sns.color_palette("tab10", n_colors=max(len(cats) + 1, 10)).as_hex()
    others = [c for c in base if c.lower() != blue.lower()]

    # assign colors: normal -> blue, others -> from filtered palette
    colors = []
    j = 0
    for c in cats:
        if c == "normal":
            colors.append(blue)
        else:
            colors.append(others[j])
            j += 1

    adata.uns[f"{col}_colors"] = colors


def plot_visium_HE(
    sample: str,
    anns: pd.DataFrame,
    adata: sc.AnnData,
    out_dir: str,
    spot_label="spot_label",
    path_label="Microregion_annotation",
    figsize=(20, 13),
    dpi=150,
    alpha=0.50,
    size=1.5,
    title_info="",
    fig_type="svg",
    trans=True,
    library_id=None,
):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype']  = 42
    plt.rcParams['svg.fonttype'] = 'none'

    n_clones = sum(1 for c in anns.columns if str.startswith(c, "clone")) + 1
    clones = ["normal"] + [f"clone{i}" for i in range(1, n_clones)]

    # plot raw HE image
    ax = sq.pl.spatial_scatter(adata, color=None, library_id=library_id, return_ax=True)
    ax.figure.savefig(
        os.path.join(out_dir, f"{sample}.HE.raw.{fig_type}"), dpi=dpi, bbox_inches="tight",
        transparent = trans
    )
    plt.close(ax.figure)

    adata.obs[spot_label] = anns[spot_label].astype("category")
    adata.obs[path_label] = anns[path_label].astype("category")
    adata.obsm["clone_proportions"] = anns[["tumor"] + clones].to_numpy()
    adata.uns["clone_proportions_cols"] = ["tumor"] + clones
    for i, clone in enumerate(adata.uns["clone_proportions_cols"]):
        adata.obs[clone] = adata.obsm["clone_proportions"][:, i]

    set_normal_blue_unique(adata, spot_label)
    set_normal_blue_unique(adata, path_label)

    ax = sq.pl.spatial_scatter(
        adata,
        color=spot_label,
        size=size,
        library_id=library_id,
        title=f"Copytyping assignments\n{title_info}",
        alpha=alpha,
        return_ax=True,
    )
    ax.figure.savefig(
        os.path.join(out_dir, f"{sample}.{spot_label}.trans.{fig_type}"),
        dpi=dpi,
        bbox_inches="tight",
        transparent = trans
    )
    plt.close(ax.figure)

    ax = sq.pl.spatial_scatter(
        adata,
        color=spot_label,
        size=size,
        library_id=library_id,
        title=f"Copytyping assignments\n{title_info}",
        return_ax=True,
    )
    ax.figure.savefig(
        os.path.join(out_dir, f"{sample}.{spot_label}.{fig_type}"),
        dpi=dpi,
        bbox_inches="tight",
        transparent = trans
    )
    plt.close(ax.figure)

    ax = sq.pl.spatial_scatter(
        adata,
        color=path_label,
        size=size,
        library_id=library_id,
        title=path_label,
        return_ax=True,
    )
    ax.figure.savefig(
        os.path.join(out_dir, f"{sample}.{path_label}.{fig_type}"),
        dpi=dpi,
        bbox_inches="tight",
        transparent = trans
    )
    plt.close(ax.figure)

    # plot spot proportions
    for clone in ["tumor"] + clones:
        ax = sq.pl.spatial_scatter(
            adata,
            color=clone,
            size=size,
            library_id=library_id,
            cmap="magma_r",
            vmin=0,
            vmax=1,
            title=f"coptyping inferred spot proportion - {clone}",
            return_ax=True,
        )

        ax.figure.savefig(
            f"{out_dir}/{sample}.spot_proportion.{clone}.{fig_type}",
            dpi=dpi,
            bbox_inches="tight",
            transparent = trans
        )
        plt.close(ax.figure)
    return


if __name__ == "__main__":
    _, sample, visium_file, ann_file, validate_file, out_dir = sys.argv

    print(f"plot visium for {sample}")
    os.makedirs(out_dir, exist_ok=True)
    adata: sc.AnnData = sc.read_h5ad(visium_file)
    print(adata)

    anns = load_visium_path_annotation(
        validate_file,
        raw_label="Microregion_annotation",
        label="path_label",
        anns=pd.read_table(ann_file, sep="\t"),
    )
    metric, metric_str = evaluate_malignant_accuracy(
        anns, cell_label="spot_label", cell_type="path_label", tumor_type="tumor"
    )

    plot_visium_HE(sample, anns, adata, out_dir, title_info=metric_str)
    print("done")
