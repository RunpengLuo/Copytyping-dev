import logging

import pandas as pd
import numpy as np

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage, leaves_list
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from copytyping.sx_data.sx_data import SX_Data
from copytyping.utils import read_whitelist_segments
from copytyping.plot.plot_copynumber import (
    BLACK,
    plot_cnv_legend,
    plot_cnv_profile,
)
from copytyping.plot.plot_common import build_wl_coords

from copytyping.inference.model_utils import empirical_baf_gn, empirical_rdr_gn

# Silence fontTools subsetter chatter
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
logging.getLogger("fontTools.ttLib").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.ERROR)


def plot_heatmap(
    ax: plt.Axes,
    cell_labels: np.ndarray,
    cnv_blocks: pd.DataFrame,
    X_mat: np.ndarray,
    wl_segments: pd.DataFrame,
    height=1,
    plot_chrname=True,
    title=None,
    ylabel=None,
    cmap=None,
    norm=None,
):
    (N, G) = X_mat.shape
    assert len(cnv_blocks) == G, "unmatched data"
    assert len(cell_labels) == N, "unmatched data"

    wl = build_wl_coords(cnv_blocks, wl_segments)
    x_edges = wl["x_edges"]
    col_bin_ids = wl["col_bin_ids"]
    ch_coords = wl["ch_coords"]
    seg_coords = wl["seg_coords"]
    ch_offset = wl["chr_end"]
    chs = cnv_blocks["#CHR"].unique()

    # -------- build extended matrix with NaN gaps --------
    n_cols = len(col_bin_ids)
    C_ext = np.full((N, n_cols), np.nan, dtype=float)
    for j, bid in enumerate(col_bin_ids):
        if bid >= 0:
            C_ext[:, j] = X_mat[:, bid]

    # 1D edges for pcolormesh
    x_edges = np.asarray(x_edges, dtype=float)
    y_edges = np.linspace(0.0, height, N + 1)
    C = np.ma.masked_invalid(C_ext)
    ax.pcolormesh(
        x_edges, y_edges, C, cmap=cmap, norm=norm, shading="flat", rasterized=True
    )
    for ch_ofs in ch_coords:
        ax.vlines(
            ch_ofs,
            ymin=0,
            ymax=1,
            transform=ax.get_xaxis_transform(),
            linewidth=1,
            colors=BLACK,
        )

    # centromere dots
    for seg_ofs in seg_coords:
        ax.vlines(
            seg_ofs,
            ymin=0,
            ymax=1,
            transform=ax.get_xaxis_transform(),
            linewidth=1,
            colors=BLACK,
            linestyles="dashed",
        )

    ax.set_xlim(0, ch_offset)
    ax.set_xlabel("")
    if plot_chrname:
        ax.set_xticks(
            [
                ch_coords[i] + (ch_coords[i + 1] - ch_coords[i]) // 2
                for i in range(len(ch_coords) - 1)
            ]
        )
        ax.set_xticklabels(chs, rotation=60, fontsize=8)
        ax.tick_params(
            axis="x", labeltop=True, labelbottom=False, top=False, bottom=False
        )
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])

    # detect contiguous label blocks for y-axis tick placement
    boundaries = np.r_[True, cell_labels[1:] != cell_labels[:-1]]
    starts = np.flatnonzero(boundaries)
    ends = np.r_[starts[1:], N]
    labels_block = cell_labels[starts]

    yticks = []
    yticklabels = []
    for start, end, label in zip(starts, ends, labels_block):
        y0 = y_edges[start]
        y1 = y_edges[end]

        proportion = round(100 * np.sum(cell_labels == label) / len(cell_labels), 1)

        rect = Rectangle(
            (0, y0),
            ch_offset,
            y1 - y0,
            fill=False,
            edgecolor="black",
            linewidth=1.0,
        )
        ax.add_patch(rect)

        yticks.append(0.5 * (y0 + y1))
        yticklabels.append(f"{label} ({proportion}%)")

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.tick_params(axis="y", length=0)

    if ylabel is not None:
        ax.set_ylabel(ylabel, rotation=0, ha="right", va="center")
    if title is not None:
        ax.set_title(title)
    return x_edges, y_edges, C


def cluster_per_group(
    mask: np.ndarray,
    X: np.ndarray,
    cell_labels: np.ndarray,
    uniq_labels: list,
):
    # cluster within groups
    order_indices = []
    for cat in uniq_labels:
        cell_mask = cell_labels == cat
        if np.sum(cell_mask) == 0:
            continue
        X_group = X[cell_mask][:, mask]
        X_group = np.nan_to_num(X_group, nan=0.5)
        if X_group.shape[0] > 2:
            Z = linkage(X_group, method="ward", metric="euclidean")
            leaf_order = leaves_list(Z)
            order_indices.extend(np.where(cell_mask)[0][leaf_order])
        else:
            order_indices.extend(np.where(cell_mask)[0])
    return X[order_indices]


def prepare_rdr(
    sx_data: SX_Data,
    cell_labels=None,
    uniq_labels=None,
    base_props=None,
    agg_size=1,
    log2=True,
    cluster_by_val=False,
):
    X_agg_list, Tn_agg_list, cell_labels_agg = [], [], []
    for lab in uniq_labels:
        idx = np.where(cell_labels == lab)[0]
        n_cells = len(idx)
        n_groups = int(np.ceil(n_cells / agg_size))
        for g in range(n_groups):
            sub_idx = idx[g * agg_size : (g + 1) * agg_size]
            if len(sub_idx) == 0:
                continue
            # sum counts per bin
            X_sum = sx_data.X[:, sub_idx].sum(axis=1)
            Tn_sum = sx_data.T[sub_idx].sum()
            X_agg_list.append(X_sum)
            Tn_agg_list.append(Tn_sum)
            cell_labels_agg.append(lab)
    X = np.column_stack(X_agg_list)  # (n_bins, new_cells)
    T = np.array(Tn_agg_list, dtype=np.int32)
    cell_labels = np.array(cell_labels_agg)

    rdr_matrix = empirical_rdr_gn(X, T, base_props, log2=log2)
    rdr_matrix = rdr_matrix.T

    # group cells by labels, clustering.
    if cluster_by_val:
        aneuploid_mask = sx_data.MASK["ANEUPLOID"]
        rdr_matrix = cluster_per_group(
            aneuploid_mask, rdr_matrix, cell_labels, uniq_labels
        )
    return rdr_matrix, cell_labels


def prepare_pi_gk(
    sx_data: SX_Data,
    cell_labels=None,
    uniq_labels=None,
    base_props=None,
    agg_size=1,
    cluster_by_val=False,
):
    X_agg_list, Tn_agg_list, cell_labels_agg = [], [], []
    for lab in uniq_labels:
        idx = np.where(cell_labels == lab)[0]
        n_cells = len(idx)
        n_groups = int(np.ceil(n_cells / agg_size))
        for g in range(n_groups):
            sub_idx = idx[g * agg_size : (g + 1) * agg_size]
            if len(sub_idx) == 0:
                continue
            # sum counts per bin
            X_sum = sx_data.X[:, sub_idx].sum(axis=1)
            Tn_sum = sx_data.T[sub_idx].sum()
            X_agg_list.append(X_sum)
            Tn_agg_list.append(Tn_sum)
            cell_labels_agg.append(lab)
    X = np.column_stack(X_agg_list)  # (n_bins, new_cells)
    T = np.array(Tn_agg_list, dtype=np.int32)
    cell_labels = np.array(cell_labels_agg)
    pi_gk_matrix = X / T[None, :]
    pi_gk_matrix[pi_gk_matrix == 0] = np.nan
    pi_gk_matrix = pi_gk_matrix.T
    # group cells by labels, clustering.
    if cluster_by_val:
        aneuploid_mask = sx_data.MASK["ANEUPLOID"]
        pi_gk_matrix = cluster_per_group(
            aneuploid_mask, pi_gk_matrix, cell_labels, uniq_labels
        )
    return pi_gk_matrix, cell_labels


def prepare_baf(
    sx_data: SX_Data,
    cell_labels=None,
    uniq_labels=None,
    agg_size=1,
    cluster_by_val=False,
):
    Y_agg_list, D_agg_list, cell_labels_agg = [], [], []
    for lab in uniq_labels:
        idx = np.where(cell_labels == lab)[0]
        n_cells = len(idx)
        n_groups = int(np.ceil(n_cells / agg_size))
        for g in range(n_groups):
            sub_idx = idx[g * agg_size : (g + 1) * agg_size]
            if len(sub_idx) == 0:
                continue
            # sum counts per bin
            Y_sum = sx_data.Y[:, sub_idx].sum(axis=1)
            D_sum = sx_data.D[:, sub_idx].sum(axis=1)
            Y_agg_list.append(Y_sum)
            D_agg_list.append(D_sum)
            cell_labels_agg.append(lab)
    Y = np.column_stack(Y_agg_list)  # (n_bins, new_cells)
    D = np.column_stack(D_agg_list)
    cell_labels = np.array(cell_labels_agg)

    baf_matrix = empirical_baf_gn(Y, D)
    baf_matrix = baf_matrix.T

    # group cells by labels, clustering.
    if cluster_by_val:
        imbalanced_mask = sx_data.MASK["IMBALANCED"]
        baf_matrix = cluster_per_group(
            imbalanced_mask, baf_matrix, cell_labels, uniq_labels
        )
    return baf_matrix, cell_labels


def plot_cnv_heatmap(
    sample: str,
    data_type: str,
    haplo_blocks: pd.DataFrame,
    sx_data: SX_Data,
    anns: pd.DataFrame,
    region_bed: str,
    proportions=None,
    val="BAF",
    base_props=None,
    agg_size=5,
    lab_type="cell_label",
    figsize=(20, 13),
    hratios=[10, 1, 2],
    filename=None,
    dpi=300,
    transparent=False,
    title_info="",
):
    assert val in ["BAF", "RDR", "log2RDR", "COUNT", "pi_gk"]
    wl_fragments = read_whitelist_segments(region_bed)
    logging.debug(f"plot CNV heatmap val={val}")

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    if anns is None:
        cell_labels = np.full(sx_data.N, fill_value="unknown")
    else:
        cell_labels = anns[lab_type].to_numpy()
    assert len(cell_labels) == sx_data.N

    # order cells by labels, aggregate same label cells
    uniq_labels = np.unique(cell_labels)
    if lab_type == "copytyping-label":
        # pcolormesh y=0 is bottom, so reverse desired top-to-bottom order
        desired = ["NA", "normal"] + [f"clone{c}" for c in range(1, sx_data.K)]
        uniq_labels = [
            lab for lab in reversed(desired) if lab in np.unique(cell_labels)
        ]

    # order props by unique labels, since data also get ordered in prepare step
    # then aggregate to match agg_size grouping
    if proportions is not None:
        ord_props = []
        for lab in uniq_labels:
            idx = np.where(cell_labels == lab)[0]
            lab_props = proportions[idx]
            if agg_size > 1:
                agg = [
                    np.mean(lab_props[g : g + agg_size])
                    for g in range(0, len(lab_props), agg_size)
                ]
                ord_props.extend(agg)
            else:
                ord_props.extend(lab_props)
        proportions = np.array(ord_props)

    cluster_by_val = False
    data_info = sx_data.cnv_blocks
    if val == "BAF":
        data_matrix, cell_labels = prepare_baf(
            sx_data, cell_labels, uniq_labels, agg_size, cluster_by_val
        )
        boundaries = np.linspace(0, 1, 11)  # [0.0, 0.1, ..., 1.0]
        colors = [
            "#1f77b4",
            "#3b8bc6",
            "#67a9cf",
            "#90c4d6",
            "#b8d6da",
            "#d9d9d9",
            "#fddbc7",
            "#f4a582",
            "#d6604d",
            "#b2182b",
        ]
        cticks = [0.0, 0.25, 0.5, 0.75, 1.0]

        cmap = mcolors.ListedColormap(colors, name="baf_disc")
        cmap.set_bad("white")
        norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
    elif val in ["RDR", "log2RDR"]:
        data_matrix, cell_labels = prepare_rdr(
            sx_data,
            cell_labels,
            uniq_labels,
            base_props,
            agg_size,
            val == "log2RDR",
            cluster_by_val=cluster_by_val,
        )
        cmap = "coolwarm"
        if val == "log2RDR":
            norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
            cticks = [-1.0, -0.50, 0.0, 0.50, 1.0]
        else:
            norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2)
            cticks = [0.0, 0.5, 1.0, 1.50, 2.0]
    elif val == "pi_gk":
        data_matrix, cell_labels = prepare_pi_gk(
            sx_data, cell_labels, uniq_labels, base_props, agg_size, cluster_by_val
        )

        colors = [
            "#1f77b4",
            "#3b8bc6",
            "#67a9cf",
            "#90c4d6",
            "#b8d6da",
            "#d9d9d9",
            "#fddbc7",
            "#f4a582",
            "#d6604d",
            "#b2182b",
        ]
        cticks = [0.0, 0.25, 0.5, 0.75, 1.0]

        cmap = mcolors.LinearSegmentedColormap.from_list("pi_cont", colors, N=256)
        norm = mcolors.Normalize(vmin=data_matrix.min(), vmax=data_matrix.max())

    if proportions is not None:
        logging.debug("plot tumor proportions")
        N = data_matrix.shape[0]
        assert len(proportions) == N
        change_pts = np.flatnonzero(cell_labels[1:] != cell_labels[:-1]) + 1
        boundaries = np.r_[0, change_pts, N]

        idx = np.arange(N)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            # DESC proportion
            order = np.argsort(proportions[start:end])[::-1]
            idx[start:end] = idx[start:end][order]

        data_matrix = data_matrix[idx]
        cell_labels = cell_labels[idx]
        proportions = proportions[idx]

    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=figsize, gridspec_kw={"height_ratios": hratios}
    )
    fig.subplots_adjust(top=0.99, right=0.95, hspace=0.03)

    x_edges, y_edges, _ = plot_heatmap(
        axes[0],
        cell_labels,
        data_info,
        data_matrix,
        wl_fragments,
        height=10,
        cmap=cmap,
        norm=norm,
    )

    plot_cnv_profile(axes[1], haplo_blocks, wl_fragments, plot_chrname=False)
    plot_cnv_legend(axes[2])

    title = f"{sample} {data_type} {val} Heatmap"
    if agg_size > 1:
        title += f" (pseudobulk-{agg_size} cell for visualization)"
    if title_info != "":
        title += f"\n{title_info}"
    fig.suptitle(title, y=0.99, fontsize=14, fontweight="bold")

    fig.tight_layout(rect=[0.0, 0.0, 0.95, 0.99])

    extra_pad = 0.0
    if proportions is not None:
        fig.canvas.draw()
        bbox = axes[0].get_position()
        cbar_width = 0.01  # in figure coords
        pad = extra_pad + 0.005  # gap between heatmap and colorbar
        extra_pad = pad + cbar_width
        ax_vec = fig.add_axes(
            [
                bbox.x1 + pad,
                bbox.y0,
                cbar_width,
                bbox.height,
            ]
        )
        x = np.array([0, 1])
        C = proportions[:, None]
        purity_cmap = "magma_r"
        norm_vec = mcolors.Normalize(vmin=0.0, vmax=1.0)
        ax_vec.pcolormesh(
            x,
            y_edges,
            C,
            cmap=purity_cmap,
            norm=norm_vec,
            shading="flat",
            rasterized=True,
        )
        ax_vec.set_xticks([0.5])
        ax_vec.set_xticklabels(["purity"], rotation=0)
        ax_vec.tick_params(
            axis="x", labeltop=True, labelbottom=False, top=False, bottom=False
        )
        ax_vec.set_yticks([])
        ax_vec.set_ylim(axes[0].get_ylim())

    fig.canvas.draw()  # ensure axis positions are finalised before adding colorbar
    bbox = axes[0].get_position()

    cbar_width = 0.01  # in figure coords
    pad = extra_pad + 0.01  # gap between heatmap and colorbar

    cax = fig.add_axes(
        [
            bbox.x1 + pad,
            bbox.y0,
            cbar_width,
            bbox.height / 5,
        ]
    )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks(cticks)

    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight", transparent=transparent)
        plt.close()
        return
    plt.show()
    return
