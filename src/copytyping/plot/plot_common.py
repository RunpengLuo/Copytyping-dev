import os
import sys
import logging

import pandas as pd
import numpy as np
# import scanpy as sc
# from scanpy import AnnData

# from matplotlib.colors import TwoSlopeNorm
# import matplotlib.colors as mcolors
import seaborn as sns
from collections import OrderedDict
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
    logging.info("plot 1D per-SNP B-allele frequency")
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
        logging.info(f"plot {chrom} with #SNP={len(snps_ch)}")
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
        for i in [0, 1]:
            axes[i].hlines(
                y=0.5,
                xmin=0,
                xmax=chr_end,
                colors="grey",
                linestyle=":",
                linewidth=1,
            )
        # BAF lines
        exp_baf_lines = []
        for _, row in haplo_blocks.loc[haplo_blocks["#CHR"] == chrom].iterrows():
            exp_baf_lines.append([(row["START"], row["BAF"]), (row["END"], row["BAF"])])
        bl_colors = [(0, 0, 0, 1)] * len(exp_baf_lines)
        for i in [0, 1]:
            axes[i].add_collection(
                LineCollection(exp_baf_lines, linewidth=2, colors=bl_colors)
            )

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
def build_ch_boundary(
    region_df: pd.DataFrame, chrs: list, chr_sizes: dict, chr_shift=10_000_000
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
    logging.info("plot 1D scatter aggregted BAF")
    chrom_sizes = get_chr_sizes(genome_file)
    cnv_blocks = sx_data.cnv_blocks
    if mask_cnp:
        cnv_blocks = cnv_blocks.loc[sx_data.MASK[mask_id], :]

    exp_bafs = None
    exp_rdrs = None  # TODO
    if lab_type == "cell_label":
        exp_bafs = sx_data.BAF
        if mask_cnp:
            exp_bafs = exp_bafs[sx_data.MASK[mask_id], :]

    cnv_blocks = cnv_blocks.copy(deep=True)

    # BAF data
    Y = sx_data.Y
    D = sx_data.D
    if mask_cnp:
        Y = Y[sx_data.MASK[mask_id]]
        D = D[sx_data.MASK[mask_id]]

    cell_labels = anns[lab_type].tolist()
    uniq_cell_labels = anns[lab_type].unique()
    assert (len(cnv_blocks), len(cell_labels)) == Y.shape

    ################
    cnv_blocks["SAMPLE"] = sample
    chrs = cnv_blocks["#CHR"].unique().tolist()
    ret = build_ch_boundary(cnv_blocks, chrs, chrom_sizes, chr_shift=int(10e6))
    (
        chr_offsets,
        chr_bounds,
        chr_gaps,
        chr_end,
        xlab_chrs,
        xtick_chrs,
    ) = ret

    positions = cnv_blocks.apply(
        func=lambda r: chr_offsets[r["#CHR"]] + (r.START + r.END) // 2, axis=1
    ).to_numpy()
    abs_starts = cnv_blocks.apply(func=lambda r: chr_offsets[r["#CHR"]] + r.START, axis=1)
    abs_ends = cnv_blocks.apply(func=lambda r: chr_offsets[r["#CHR"]] + r.END, axis=1)
    ################
    # prepare platte and markers
    markersize = float(max(20, 4 - np.floor(len(cnv_blocks) / 500)))
    # markersize_centroid = 10
    # marker_bd_width = 0.8
    sns.set_style("whitegrid")
    palette = sns.color_palette("husl")
    if len(cnv_blocks) > 8:
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

    ax.set_title("normal posterior")
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)


def plot_params(
    params: dict, out_file: str, data_type: str, names=["tau", "lambda", "inv_phi"]
):
    with PdfPages(out_file) as pdf:
        for name in names:
            param = params[f"{data_type}-{name}"]
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            ax.hist(x=param, bins=50)
            title = f"{data_type} {name}\nmean={param.mean():.3f} std={param.std():.3f}"
            ax.set_title(title)
            fig.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close()
    return

def plot_loss(
    losses: list,
    out_loss_file: str,
    val_type="log-likelihood",
    dpi=100
):
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("iterations")
    ax.set_ylabel(val_type)
    fig.tight_layout()
    fig.savefig(out_loss_file, dpi=dpi)
    plt.close(fig)


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
    logging.info(data)

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
