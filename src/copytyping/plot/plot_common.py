import os
import sys

import pandas as pd
import numpy as np
# import scanpy as sc
# from scanpy import AnnData

# from matplotlib.colors import TwoSlopeNorm
# import matplotlib.colors as mcolors
import seaborn as sns

# from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib.collections import LineCollection
from scipy import sparse, stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from collections import OrderedDict

from copytyping.utils import get_chr_sizes
from copytyping.io_utils import *
from copytyping.sx_data.sx_data import SX_Data


##################################################
# plot SNP depth vs BAF
def plot_snps_DP_BAF(
    snp_info: pd.DataFrame,
    baf_vals: np.ndarray,
    allele_counts: np.ndarray,
    out_file: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    sns.histplot(x=baf_vals, ax=axes[0], binrange=[0, 1], bins=20)
    sns.histplot(x=allele_counts, ax=axes[1], bins=20)
    sns.scatterplot(x=allele_counts, y=baf_vals, ax=axes[2])
    plt.tight_layout()
    fig.savefig(out_file, dpi=100)
    return


# plot phased SNPs statistics
def plot_snps_per_chrom(
    snp_info: pd.DataFrame,
    haplo_blocks: pd.DataFrame,
    genome_file: str,
    out_dir: str,
    out_prefix: str,
    lab_raw="PS",
    lab_corr="HB",
    s=4,
):
    print("plot 1D per-SNP B-allele frequency")
    chrom_sizes = get_chr_sizes(genome_file)

    colors = ["#1f77b4", "#ff7f0e"]  # blue / orange

    if lab_raw in snp_info:
        codes_raw = snp_info[lab_raw].astype("category").cat.codes % 2
        color_raw = codes_raw.map({0: colors[0], 1: colors[1]}).to_numpy()

    if lab_corr in snp_info:
        codes_corr = snp_info[lab_corr].astype("category").cat.codes % 2
        color_corr = codes_corr.map({0: colors[0], 1: colors[1]}).to_numpy()

    snps_chs = snp_info.groupby("#CHR", sort=False)
    for chrom in snp_info["#CHR"].unique():
        chr_end = chrom_sizes[chrom]
        out_file = os.path.join(out_dir, f"{out_prefix}.{chrom}.png")
        snps_ch = snps_chs.get_group(chrom)
        print(f"plot {chrom} with #SNP={len(snps_ch)}")
        fig, axes = plt.subplots(3, 1, figsize=(40, 6), sharex=True)
        fig.suptitle(f"{chrom}", fontsize=12)

        axes[0].scatter(
            snps_ch["POS"],
            snps_ch["BAF_RAW"],
            s=s,
            color=color_raw[snps_ch.index],
            alpha=0.6,
            rasterized=True,
        )
        axes[1].scatter(
            snps_ch["POS"],
            snps_ch["BAF_CORR"],
            s=s,
            color=color_corr[snps_ch.index],
            alpha=0.6,
            rasterized=True,
        )

        # TODO add segment BAF hlines
        [
            axes[i].hlines(
                y=0.5,
                xmin=0,
                xmax=chr_end,
                colors="grey",
                linestyle=":",
                linewidth=1,
            )
            for i in [0, 1]
        ]
        # BAF lines
        exp_baf_lines = []
        for _, row in haplo_blocks.loc[haplo_blocks["#CHR"] == chrom].iterrows():
            exp_baf_lines.append([(row["START"], row["BAF"]), (row["END"], row["BAF"])])
        bl_colors = [(0, 0, 0, 1)] * len(exp_baf_lines)
        [
            axes[i].add_collection(
                LineCollection(exp_baf_lines, linewidth=2, colors=bl_colors)
            )
            for i in [0, 1]
        ]

        axes[2].scatter(
            snps_ch["POS"],
            snps_ch["DP"],
            s=s,
            color=color_raw[snps_ch.index],
            alpha=0.6,
            rasterized=True,
        )

        axes[0].set_ylim(0, 1)
        axes[1].set_ylim(0, 1)
        for ax in axes:
            ax.set_xlim(0, chr_end)
            ax.grid(alpha=0.2)

        axes[0].set_ylabel("BAF_RAW")
        axes[1].set_ylabel("BAF_CORR")
        axes[2].set_ylabel("DEPTH")
        fig.supxlabel("Position (bp)")
        plt.tight_layout()

        fig.savefig(out_file, dpi=150)
        plt.close(fig)
        fig.clear()
    return


##################################################
def plot_library_sizes(
    sx_data: SX_Data, sample: str, data_type: str, out_file: str, celltypes=None
):
    T = sx_data.T
    mean = np.mean(T)
    med = np.median(T)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.histplot(x=T, hue=celltypes, bins=50, ax=ax)
    ax.set_title(f"{sample} - {data_type} - library size\nmean={mean:.3f},median={med}")
    fig.savefig(out_file, dpi=150)
    return


##################################################
# def cluster_per_group(
#     adata: AnnData,
#     cluster_chroms=None,
#     groupby="cell_label"
# ):
#     # cluster within groups
#     if cluster_chroms is not None:
#         chrom_mask = adata.var["#CHR"].isin(cluster_chroms)
#     else:
#         chrom_mask = np.ones(adata.n_vars, dtype=bool)
#     order_indices = []
#     groups = adata.obs[groupby]
#     for cat in groups.unique():
#         cell_mask = groups == cat
#         if np.sum(cell_mask) == 0:
#             continue
#         X_group = adata.X[cell_mask][:, chrom_mask]
#         X_group = np.nan_to_num(X_group, nan=0.5)
#         if X_group.shape[0] > 2:
#             Z = linkage(X_group, method="ward", metric="euclidean")
#             leaf_order = leaves_list(Z)
#             order_indices.extend(np.where(cell_mask)[0][leaf_order])
#         else:
#             order_indices.extend(np.where(cell_mask)[0])
#     return adata[order_indices, :].copy()

##################################################
# plot 1D cell/spot by genome bin heatmap for one modality
# def plot_baf_1d(
#     sx_data: SX_Data,
#     anns: pd.DataFrame,
#     sample: str,
#     data_type: str,
#     mask_cnp=True,
#     mask_id="CNP",
#     lab_type="cell_label",
#     figsize=(20, 10),
#     filename=None,
#     agg_size=5,
#     skip_cluster_per_group=False,
#     **kwargs
# ):
#     bin_info = sx_data.bin_info
#     if mask_cnp:
#         bin_info = bin_info.loc[sx_data.MASK[mask_id], :]
#     print(f"plot 1D BAF heatmap, #bins={len(bin_info)}")

#     print(bin_info.head())
#     # BAF data
#     Y = sx_data.Y
#     D = sx_data.D

#     if anns is None:
#         cell_labels = np.full(Y.shape[1], fill_value="unknown")
#     else:
#         cell_labels = anns[lab_type].to_numpy()
#     assert len(cell_labels) == Y.shape[1]

#     # aggregate subset cells
#     if agg_size > 1:
#         Y_agg_list, D_agg_list, cell_labels_agg = [], [], []
#         for lab in np.unique(cell_labels):
#             idx = np.where(cell_labels == lab)[0]
#             n_cells = len(idx)
#             n_groups = int(np.ceil(n_cells / agg_size))
#             for g in range(n_groups):
#                 sub_idx = idx[g*agg_size:(g+1)*agg_size]
#                 if len(sub_idx) == 0:
#                     continue
#                 # sum counts per bin
#                 Y_sum = Y[:, sub_idx].sum(axis=1)
#                 D_sum = D[:, sub_idx].sum(axis=1)
#                 Y_agg_list.append(Y_sum)
#                 D_agg_list.append(D_sum)
#                 cell_labels_agg.append(lab)
#         Y = np.column_stack(Y_agg_list)  # (n_bins, new_cells)
#         D = np.column_stack(D_agg_list)
#         cell_labels = np.array(cell_labels_agg)

#     baf_matrix = np.divide(
#         Y, D, out=np.full_like(D, fill_value=np.nan, dtype=np.float32), where=D > 0
#     )
#     if mask_cnp:
#         baf_matrix = baf_matrix[sx_data.MASK[mask_id]]

#     baf_matrix = baf_matrix.T

#     # build anndata
#     adata = AnnData(X=baf_matrix)
#     adata.obs[lab_type] = cell_labels
#     adata.var[['#CHR','START','END']] = bin_info[['#CHR','START','END']].values
#     if not skip_cluster_per_group:
#         adata_sorted = cluster_per_group(adata, cluster_chroms=None, groupby=lab_type)
#     else:
#         adata_sorted = adata
#     chroms = adata_sorted.var["#CHR"].to_numpy()
#     chr_change_idx = np.where(chroms[1:] != chroms[:-1])[0] + 1
#     chr_pos = [0] + chr_change_idx.tolist()
#     var_group_labels = list(chroms[chr_pos])
#     var_group_positions = [
#         (chr_pos[i], chr_pos[i + 1] if i + 1 < len(chr_pos) else len(bin_info))
#         for i in range(len(chr_pos))
#     ]


#     cmap = mcolors.LinearSegmentedColormap.from_list(
#         "baf_map",
#         # [(0.0, "blue"), (0.5, "green"), (1.0, "red")]
#         [(0.0, "#1f77b4"), (0.5, "#bfbfbf"), (1.0, "#d62728")]
#     )
#     cmap.set_bad(color="white")  # NaNs pure white
#     norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

#     ax_dict = sc.pl.heatmap(
#         adata,
#         var_names=adata.var_names,
#         groupby=lab_type,
#         figsize=figsize,
#         cmap=cmap,
#         norm=norm,
#         show_gene_labels=False,
#         var_group_positions=var_group_positions,
#         var_group_labels=var_group_labels,
#         dendrogram=False,
#         show=False,
#         **kwargs,
#     )

#     ax_dict["heatmap_ax"].vlines(chr_pos[1:], lw=0.6, ymin=0, ymax=adata.shape[0], color="black")
#     ax_dict["heatmap_ax"].set_title(f"{sample} {data_type} BAF Heatmap", y=1.10)
#     if not filename is None:
#         sc.pl._utils.savefig(filename, dpi=300)
#         plt.close()
#         return
#     plt.show()

# def plot_rdr_1d(
#     sx_data: SX_Data,
#     anns: pd.DataFrame,
#     sample: str,
#     data_type: str,
#     base_props: np.ndarray,
#     mask_cnp=True,
#     mask_id="CNP",
#     lab_type="cell_label",
#     figsize=(20, 10),
#     filename=None,
#     agg_size=5,
#     verbose=1,
#     **kwargs
# ):
#     bin_info = sx_data.bin_info
#     cnp_mask = base_props > 0
#     if mask_cnp:
#         cnp_mask &= sx_data.MASK[mask_id]
#     bin_info = bin_info.loc[cnp_mask, :]

#     T = sx_data.T # (G, N)
#     T = sx_data.T # (N, )

#     cell_labels = anns[lab_type].to_numpy()
#     assert len(cell_labels) == T.shape[1]

#     if agg_size > 1:
#         T_agg_list, Tn_agg_list, cell_labels_agg = [], [], []
#         for lab in np.unique(cell_labels):
#             idx = np.where(cell_labels == lab)[0]
#             n_cells = len(idx)
#             n_groups = int(np.ceil(n_cells / agg_size))
#             for g in range(n_groups):
#                 sub_idx = idx[g*agg_size:(g+1)*agg_size]
#                 if len(sub_idx) == 0:
#                     continue
#                 # sum counts per bin
#                 T_sum = T[:, sub_idx].sum(axis=1)
#                 Tn_sum = T[sub_idx].sum()
#                 T_agg_list.append(T_sum)
#                 Tn_agg_list.append(Tn_sum)
#                 cell_labels_agg.append(lab)
#         T = np.column_stack(T_agg_list)  # (n_bins, new_cells)
#         T = np.array(Tn_agg_list, dtype=np.int32)
#         cell_labels = np.array(cell_labels_agg)

#     rdr_matrix = (T / (base_props[:, None] @ T[None, :])).T
#     rdr_matrix = rdr_matrix[:, cnp_mask]

#     if verbose:
#         print(f"before log2 transform median={np.median(rdr_matrix)} max={np.max(rdr_matrix)}")
#     rdr_matrix = np.log2(np.clip(rdr_matrix, a_min=1e-6, a_max=np.inf))
#     if verbose:
#         print(f"after log2 transform median={np.median(rdr_matrix)} max={np.max(rdr_matrix)}")

#     # build anndata
#     adata = AnnData(X=rdr_matrix)
#     adata.obs[lab_type] = cell_labels
#     adata.var[['#CHR','START','END']] = bin_info[['#CHR','START','END']].values
#     adata_sorted = cluster_per_group(adata, cluster_chroms=None, groupby=lab_type)

#     chroms = adata_sorted.var["#CHR"].to_numpy()
#     chr_change_idx = np.where(chroms[1:] != chroms[:-1])[0] + 1
#     chr_pos = [0] + chr_change_idx.tolist()
#     var_group_labels = list(chroms[chr_pos])
#     var_group_positions = [
#         (chr_pos[i], chr_pos[i + 1] if i + 1 < len(chr_pos) else len(bin_info))
#         for i in range(len(chr_pos))
#     ]

#     norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

#     ax_dict = sc.pl.heatmap(
#         adata,
#         var_names=adata.var_names,
#         groupby=lab_type,
#         figsize=figsize,
#         cmap="coolwarm",
#         norm=norm,
#         show_gene_labels=False,
#         var_group_positions=var_group_positions,
#         var_group_labels=var_group_labels,
#         dendrogram=False,
#         show=False,
#         **kwargs,
#     )

#     ax_dict["heatmap_ax"].vlines(chr_pos[1:], lw=0.6, ymin=0, ymax=adata.shape[0], color="black")
#     ax_dict["heatmap_ax"].set_title(f"{sample} {data_type} log2-RDR Heatmap", y=1.10)
#     if not filename is None:
#         sc.pl._utils.savefig(filename, dpi=300)
#         plt.close()
#         return
#     plt.show()


##################################################
# plot 1d clone-aggregated BAF
def build_ch_boundary(
    region_df: pd.DataFrame, chrs: list, chr_sizes: dict, chr_shift=int(10e6)
):
    # get 1d plot chromosome offsets, global information
    chr_offsets = OrderedDict()
    for i, ch in enumerate(chrs):
        if i == 0:
            chr_offsets[ch] = chr_shift
        else:
            prev_ch = chrs[i - 1]
            offset = chr_offsets[prev_ch] + chr_sizes[prev_ch]
            chr_offsets[ch] = offset
    chr_end = chr_offsets[chrs[-1]] + chr_sizes[chrs[-1]] + chr_shift
    chr_bounds = list(chr_offsets.values()) + [
        chr_offsets[chrs[-1]] + chr_sizes[chrs[-1]]
    ]
    xlab_chrs = chrs  # ignore first dummy variable
    xtick_chrs = []
    for i in range(len(chrs)):
        l = chr_offsets[chrs[i]]
        if i < len(chrs) - 2:
            r = chr_offsets[chrs[i + 1]]
        else:
            r = chr_end
        xtick_chrs.append((l + r) / 2)

    # infer chromosome-gaps from SEG file
    # all samples should share same gaps
    dummy_sample = region_df["SAMPLE"].unique()[0]
    chr_gaps = OrderedDict()
    for ch in chrs:
        chr_regions = region_df[
            (region_df["#CHR"] == ch) & (region_df["SAMPLE"] == dummy_sample)
        ][["START", "END"]].to_numpy()
        chr_regions_shift = chr_regions + chr_offsets[ch]
        chr_gaps[ch] = []
        if chr_regions[0, 0] > 0:
            chr_gaps[ch].append([chr_offsets[ch], chr_regions_shift[0, 0]])
        for i in range(len(chr_regions_shift) - 1):
            _, curr_t = chr_regions_shift[i,]
            next_s, next_t = chr_regions_shift[i + 1,]
            if curr_t < next_s:
                chr_gaps[ch].append([curr_t, next_s])
        if next_t - chr_offsets[ch] < chr_sizes[ch]:
            chr_gaps[ch].append([next_t, chr_offsets[ch] + chr_sizes[ch]])
    return (
        chr_offsets,
        chr_bounds,
        chr_gaps,
        chr_end,
        xlab_chrs,
        xtick_chrs,
    )


def plot_rdr_baf_1d_aggregated(
    sx_data: SX_Data,
    anns: pd.DataFrame,
    base_props: np.ndarray,
    sample: str,
    data_type: str,
    genome_file: str,
    mask_cnp=True,
    mask_id="CNP",
    lab_type="cell_label",
    figsize=(20, 4),
    filename=None,
    **kwargs,
):
    """
    For each decision
        1) for each bin, aggregate b-counts and t-counts
        2) compute per-bin aggregated BAF
    Plot 1d scatter along the chromosomes
        1) BAF
        2) chromosome boundary
        3) expected BAF
    """
    print("plot 1D scatter aggregted BAF")
    chrom_sizes = get_chr_sizes(genome_file)
    bin_info = sx_data.bin_info
    # bin_info = sx_data.bin_info
    # feat_mask = base_props > 0
    if mask_cnp:
        bin_info = bin_info.loc[sx_data.MASK[mask_id], :]
        # feat_mask &= sx_data.MASK[mask_id]
    # bin_info = bin_info.loc[mask_id, :]

    exp_bafs = None
    exp_rdrs = None  # TODO
    if lab_type == "cell_label":
        exp_bafs = sx_data.BAF
        if mask_cnp:
            exp_bafs = exp_bafs[sx_data.MASK[mask_id], :]

    bin_info = bin_info.copy(deep=True)

    # BAF data
    Y = sx_data.Y
    D = sx_data.D
    if mask_cnp:
        Y = Y[sx_data.MASK[mask_id]]
        D = D[sx_data.MASK[mask_id]]

    cell_labels = anns[lab_type].tolist()
    uniq_cell_labels = anns[lab_type].unique()
    assert (len(bin_info), len(cell_labels)) == Y.shape

    ################
    bin_info["SAMPLE"] = sample
    chrs = bin_info["#CHR"].unique().tolist()
    ret = build_ch_boundary(bin_info, chrs, chrom_sizes, chr_shift=int(10e6))
    (
        chr_offsets,
        chr_bounds,
        chr_gaps,
        chr_end,
        xlab_chrs,
        xtick_chrs,
    ) = ret

    positions = bin_info.apply(
        func=lambda r: chr_offsets[r["#CHR"]] + (r.START + r.END) // 2, axis=1
    ).to_numpy()
    abs_starts = bin_info.apply(func=lambda r: chr_offsets[r["#CHR"]] + r.START, axis=1)
    abs_ends = bin_info.apply(func=lambda r: chr_offsets[r["#CHR"]] + r.END, axis=1)
    ################
    # prepare platte and markers
    markersize = float(max(20, 4 - np.floor(len(bin_info) / 500)))
    # markersize_centroid = 10
    # marker_bd_width = 0.8
    sns.set_style("whitegrid")
    palette = sns.color_palette("husl")
    if len(bin_info) > 8:
        palette = sns.color_palette("husl", n_colors=len(uniq_cell_labels))
    else:
        palette = sns.color_palette("Set2", n_colors=len(uniq_cell_labels))
    sns.set_palette(palette)

    ################
    pdf_fd = PdfPages(filename)
    # compute aggregate BAF
    uniq_cell_labels = anns[lab_type].unique()
    for i, cell_label in enumerate(uniq_cell_labels):
        barcode_idxs = anns[anns[lab_type] == cell_label].index.to_numpy()
        num_bcs = len(barcode_idxs)
        agg_bcounts = np.sum(Y[:, barcode_idxs], axis=1)
        agg_tcounts = np.sum(D[:, barcode_idxs], axis=1)
        agg_bafs = agg_bcounts[agg_tcounts > 0] / agg_tcounts[agg_tcounts > 0]
        _positions = positions[agg_tcounts > 0]
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        ax.scatter(
            _positions,
            agg_bafs,
            s=markersize,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.8,
            color=palette[i],
            marker="o",  # ensure filled circle
        )
        ax.vlines(
            list(chr_offsets.values()),
            ymin=0,
            ymax=1,
            transform=ax.get_xaxis_transform(),
            linewidth=0.5,
            colors="k",
        )
        # add BAF 0.5 line
        ax.hlines(
            y=0.5,
            xmin=0,
            xmax=chr_end,
            colors="grey",
            linestyle=":",
            linewidth=1,
        )
        if not exp_bafs is None and cell_label != "NA":
            clone_baf = list(exp_bafs[:, sx_data.clones.index(cell_label)])
            exp_baf_lines = [
                [(s, baf), (t, baf)]
                for (s, t, baf) in zip(abs_starts, abs_ends, clone_baf)
            ]
            bl_colors = [(0, 0, 0, 1)] * len(clone_baf)
            ax.add_collection(
                LineCollection(exp_baf_lines, linewidth=2, colors=bl_colors)
            )
        ax.grid(False)
        plt.setp(ax, xlim=(0, chr_end), xticks=xtick_chrs, xlabel="")
        ax.set_xticklabels(xlab_chrs, rotation=60, fontsize=8)
        ax.set_ylabel("aggregated BAF")
        ax.set_title(
            f"{sample} {data_type} Aggregated B-allele Frequency Plot\n{cell_label} #{num_bcs}"
        )
        fig.tight_layout()
        pdf_fd.savefig(fig, dpi=150)
        fig.clear()
        plt.close()
    pdf_fd.close()
    return


##################################################
# plot parameters
def plot_baseline_proportions(params: dict, out_file: str, data_type: str):
    base_props = params[f"{data_type}-lambda"]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    ax.hist(x=base_props, bins=50)
    title = f"{data_type} baseline proportions mean={base_props.mean():.3f} std={base_props.std():.3f}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    return


def plot_dispersions(params: dict, out_file: str, data_type: str, name="tau"):
    dispersions = params[f"{data_type}-{name}"]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    ax.hist(x=dispersions, bins=50)
    title = f"{data_type} dispersion-{name} mean={dispersions.mean():.3f} std={dispersions.std():.3f}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    return


def plot_posteriors(
    anns: pd.DataFrame,
    out_file: str,
    lab_type="cell_label",
):
    fig, ax = plt.subplots(1, 1)
    sns.histplot(
        data=anns,
        x="normal",
        hue=lab_type,
        multiple="stack",
        ax=ax,
        binrange=[0, 1],
        bins=10,
    )

    ax.set_title(f"normal posterior")
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)


def plot_params(
    params: dict, out_file: str, data_type: str, names=["tau", "lambda", "inv_phi"]
):
    with PdfPages(out_file) as pdf:
        for name in names:
            param = params[f"{data_type}-{name}"]
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            ax.hist(x=params[f"{data_type}-{name}"], bins=50)
            title = f"{data_type} {name}\nmean={param.mean():.3f} std={param.std():.3f}"
            ax.set_title(title)
            fig.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()
        pdf.close()
    return


##################################################
def plot_cross_heatmap(
    assign_df: pd.DataFrame,
    sample: str,
    outfile: str,
    acol="final_type",
    bcol="Decision",
):
    """
    Plot heatmap to cross-check assignments and other method's result
    """
    avals = assign_df[acol].unique().tolist()
    bvals = assign_df[bcol].unique().tolist()
    num_avals = len(avals)
    num_bvals = len(bvals)
    data = pd.pivot_table(
        assign_df, index=acol, columns=bcol, aggfunc="size", fill_value=0
    ).astype(int)
    data = data.astype(int)
    print(data)

    fig, axes = plt.subplots(
        num_avals,
        1,
        figsize=(6, 6),
        gridspec_kw={"height_ratios": [1] * num_avals},
        # constrained_layout=True
    )

    if num_avals == 1:
        axes = [axes]
    for i, aval in enumerate(avals):
        row = np.array(data.loc[aval, bvals].tolist()).reshape(1, num_bvals)
        ax = axes[i]
        im = ax.imshow(
            row, aspect="auto", cmap="RdYlGn", vmin=np.min(row), vmax=np.max(row)
        )
        for j in range(num_bvals):
            ax.text(
                j,
                0,
                row[0, j],
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=24,
            )
        if i != num_avals - 1:
            ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(aval, rotation=45, labelpad=50)
        cbar = plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    axes[0].set_title(
        f"Cell Assignment Heatmap {sample}", fontweight="bold", fontsize=18
    )
    axes[-1].set_xticks(np.arange(num_bvals), labels=bvals)
    axes[-1].tick_params("x", length=0, width=0, gridOn=False, left=False, right=False)
    plt.tight_layout()

    plt.savefig(outfile, dpi=300)
    plt.close()
    return
