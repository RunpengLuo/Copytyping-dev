import os
import logging

import pandas as pd
import numpy as np
import seaborn as sns
from collections import OrderedDict
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from copytyping.utils import get_chr_sizes
from copytyping.io_utils import *
from copytyping.sx_data.sx_data import SX_Data


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
    xlab_chrs = chrs
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
    figsize=(20, 8),
    filename=None,
    **kwargs,
):
    """
    Per-clone RDR + BAF scatter plot along the genome.
    Each clone = one PDF page with 2 rows: RDR (top) and BAF (bottom).

    Observed RDR = x_{g,n} / (T_n * lambda_g)  (CalicoST S2)
    Observed BAF = y_{g,n} / D_{g,n}
    """
    logging.info("plot 1D scatter aggregated RDR+BAF")
    chrom_sizes = get_chr_sizes(genome_file)
    cnv_blocks = sx_data.cnv_blocks
    if mask_cnp:
        cnv_blocks = cnv_blocks.loc[sx_data.MASK[mask_id], :]

    exp_bafs = sx_data.BAF
    if mask_cnp:
        exp_bafs = exp_bafs[sx_data.MASK[mask_id], :]

    cnv_blocks = cnv_blocks.copy(deep=True)

    # count data
    X = sx_data.X
    Y = sx_data.Y
    D = sx_data.D
    T = sx_data.T
    if mask_cnp:
        mask = sx_data.MASK[mask_id]
        X = X[mask]
        Y = Y[mask]
        D = D[mask]

    # Mask baseline proportions to the same bins as the count data.
    masked_base_props = base_props
    if masked_base_props is not None and mask_cnp:
        masked_base_props = masked_base_props[sx_data.MASK[mask_id]]

    cell_labels = anns[lab_type].tolist()
    uniq_cell_labels = anns[lab_type].unique()
    assert Y.shape[0] == len(cnv_blocks), (
        f"bin count mismatch: Y has {Y.shape[0]} rows but cnv_blocks has {len(cnv_blocks)}"
    )
    assert Y.shape[1] == len(cell_labels), (
        f"cell count mismatch: Y has {Y.shape[1]} columns but annotations has {len(cell_labels)}"
    )

    # genome coordinates
    cnv_blocks["SAMPLE"] = sample
    chrs = cnv_blocks["#CHR"].unique().tolist()
    ret = build_ch_boundary(cnv_blocks, chrs, chrom_sizes, chr_shift=int(10e6))
    chr_offsets, chr_bounds, chr_gaps, chr_end, xlab_chrs, xtick_chrs = ret

    positions = cnv_blocks.apply(
        func=lambda r: chr_offsets[r["#CHR"]] + (r.START + r.END) // 2, axis=1
    ).to_numpy()
    abs_starts = cnv_blocks.apply(
        func=lambda r: chr_offsets[r["#CHR"]] + r.START, axis=1
    )
    abs_ends = cnv_blocks.apply(func=lambda r: chr_offsets[r["#CHR"]] + r.END, axis=1)

    # Clamp to 20 for any reasonably sized dataset (formula goes negative above ~2k bins).
    markersize = float(max(20, 4 - np.floor(len(cnv_blocks) / 500)))

    # Single PDF, one page per clone.
    pdf_pages = PdfPages(filename)

    for i, cell_label in enumerate(uniq_cell_labels):
        barcode_idxs = anns[anns[lab_type] == cell_label].index.to_numpy()
        num_bcs = len(barcode_idxs)
        color = sns.color_palette("husl", n_colors=len(uniq_cell_labels))[i]

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)

        # ── RDR panel (top) ──
        ax_rdr = axes[0]
        if masked_base_props is not None:
            # observed RDR = sum(X[:,spots]) / (sum(T[spots]) * lambda_g)
            agg_x = np.sum(X[:, barcode_idxs], axis=1).astype(np.float64)
            agg_T = np.sum(T[barcode_idxs]).astype(np.float64)
            rdr_valid = masked_base_props > 0
            obs_rdr = np.full(len(agg_x), np.nan)
            obs_rdr[rdr_valid] = agg_x[rdr_valid] / (
                agg_T * masked_base_props[rdr_valid]
            )
            pos_rdr = positions[rdr_valid & np.isfinite(obs_rdr)]
            val_rdr = obs_rdr[rdr_valid & np.isfinite(obs_rdr)]
            ax_rdr.scatter(
                pos_rdr,
                val_rdr,
                s=markersize,
                edgecolors="black",
                linewidths=0.5,
                alpha=0.8,
                color=color,
            )
            # expected RDR line at 1.0 (normal baseline)
            ax_rdr.hlines(
                y=1.0,
                xmin=0,
                xmax=chr_end,
                colors="grey",
                linestyle=":",
                linewidth=1,
            )
            # expected RDR per segment for this clone: C[g,k] / C[g,0]
            if cell_label != "NA":
                try:
                    clone_idx = sx_data.clones.index(cell_label)
                    C_normal = np.maximum(sx_data.C[:, 0], 1)
                    if mask_cnp:
                        C_normal = C_normal[sx_data.MASK[mask_id]]
                        clone_C = sx_data.C[sx_data.MASK[mask_id], clone_idx]
                    else:
                        clone_C = sx_data.C[:, clone_idx]
                    exp_rdr = clone_C / C_normal
                    exp_rdr_lines = [
                        [(s, r), (t, r)]
                        for s, t, r in zip(abs_starts, abs_ends, exp_rdr)
                    ]
                    ax_rdr.add_collection(
                        LineCollection(
                            exp_rdr_lines,
                            linewidth=2,
                            colors=[(0, 0, 0, 1)] * len(exp_rdr),
                        )
                    )
                except ValueError:
                    pass
            ax_rdr.set_ylim([-0.1, min(max(val_rdr.max() * 1.1, 2.0), 6.0)])
        else:
            ax_rdr.text(
                0.5,
                0.5,
                "RDR not available (no baseline)",
                transform=ax_rdr.transAxes,
                ha="center",
                va="center",
            )
        ax_rdr.vlines(
            list(chr_offsets.values()),
            ymin=0,
            ymax=1,
            transform=ax_rdr.get_xaxis_transform(),
            linewidth=0.5,
            colors="k",
        )
        ax_rdr.set_ylabel(f"{cell_label}\nRDR")
        ax_rdr.set_title(f"{sample} {data_type} — {cell_label} (n={num_bcs})")
        ax_rdr.grid(False)

        # ── BAF panel (bottom) ──
        ax_baf = axes[1]
        agg_bcounts = np.sum(Y[:, barcode_idxs], axis=1)
        agg_tcounts = np.sum(D[:, barcode_idxs], axis=1)
        baf_valid = agg_tcounts > 0
        agg_bafs = agg_bcounts[baf_valid] / agg_tcounts[baf_valid]
        pos_baf = positions[baf_valid]
        ax_baf.scatter(
            pos_baf,
            agg_bafs,
            s=markersize,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.8,
            color=color,
        )
        ax_baf.vlines(
            list(chr_offsets.values()),
            ymin=0,
            ymax=1,
            transform=ax_baf.get_xaxis_transform(),
            linewidth=0.5,
            colors="k",
        )
        ax_baf.hlines(
            y=0.5,
            xmin=0,
            xmax=chr_end,
            colors="grey",
            linestyle=":",
            linewidth=1,
        )
        # expected BAF lines
        if exp_bafs is not None and cell_label != "NA":
            try:
                clone_idx = sx_data.clones.index(cell_label)
                clone_baf = exp_bafs[:, clone_idx]
                exp_baf_lines = [
                    [(s, baf), (t, baf)]
                    for s, t, baf in zip(abs_starts, abs_ends, clone_baf)
                ]
                ax_baf.add_collection(
                    LineCollection(
                        exp_baf_lines,
                        linewidth=2,
                        colors=[(0, 0, 0, 1)] * len(clone_baf),
                    )
                )
            except ValueError:
                pass
        ax_baf.set_ylim([-0.05, 1.05])
        ax_baf.set_ylabel(f"{cell_label}\nphased AF")
        ax_baf.grid(False)
        plt.setp(ax_baf, xlim=(0, chr_end), xticks=xtick_chrs)
        ax_baf.set_xticklabels(xlab_chrs, rotation=60, fontsize=8)

        fig.tight_layout()
        pdf_pages.savefig(fig, dpi=150)
        plt.close(fig)

    pdf_pages.close()
    return


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
    x_col = "max_posterior"
    title = "max posterior"
    sns.histplot(
        data=anns,
        x=x_col,
        hue=lab_type,
        multiple="stack",
        ax=ax,
        binrange=[0, 1],
        bins=20,
    )

    ax.set_title(title)
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


def plot_loss(losses: list, out_loss_file: str, val_type="log-likelihood", dpi=100):
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("iterations")
    ax.set_ylabel(val_type)
    fig.tight_layout()
    fig.savefig(out_loss_file, dpi=dpi)
    plt.close(fig)


def plot_cross_heatmap(
    assign_df: pd.DataFrame,
    sample: str,
    outfile: str,
    acol="final_type",
    bcol="Decision",
):
    """
    Plot heatmap to cross-check assignments vs reference cell types.

    Rows = predicted labels (acol), columns = reference cell types (bcol).
    Color encodes row-normalized fraction so each predicted label's distribution
    is directly comparable. Raw counts are annotated in each cell.
    """
    # --- build pivot table (counts) ---
    data = pd.pivot_table(
        assign_df, index=acol, columns=bcol, aggfunc="size", fill_value=0
    ).astype(int)
    logging.info(data)

    # sort rows: normal first, then clones, NA last
    row_order = (
        [r for r in ["normal"] if r in data.index]
        + sorted([r for r in data.index if r.startswith("clone")])
        + [r for r in data.index if r not in ["normal"] and not r.startswith("clone")]
    )
    # sort columns: put tumor-like labels first
    tumor_first = [c for c in data.columns if c.lower() in {"tumor", "tumor_cell"}]
    other_cols = [c for c in data.columns if c not in tumor_first]
    col_order = tumor_first + other_cols

    data = data.loc[row_order, col_order]
    counts = data.to_numpy(dtype=float)

    # row-normalise → fraction of each predicted label
    row_sums = counts.sum(axis=1, keepdims=True)
    fractions = np.divide(
        counts, row_sums, where=row_sums > 0, out=np.zeros_like(counts)
    )

    num_rows, num_cols = counts.shape
    cell_w = max(0.7, 8.0 / num_cols)
    cell_h = 1.2
    fig_w = max(8, cell_w * num_cols + 2)
    fig_h = max(3, cell_h * num_rows + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(fractions, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    # annotate with raw counts; auto-size font to fit cells
    font_size = max(6, min(11, int(280 / num_cols)))
    for i in range(num_rows):
        for j in range(num_cols):
            n = int(counts[i, j])
            frac = fractions[i, j]
            text_color = "white" if frac < 0.35 or frac > 0.75 else "black"
            ax.text(
                j,
                i,
                str(n),
                ha="center",
                va="center",
                color=text_color,
                fontsize=font_size,
                fontweight="bold",
            )

    ax.set_xticks(np.arange(num_cols))
    ax.set_xticklabels(col_order, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(num_rows))
    ax.set_yticklabels(row_order, fontsize=10)
    ax.tick_params(length=0)
    ax.set_xlabel(bcol, fontsize=10)
    ax.set_ylabel(acol, fontsize=10)
    ax.set_title(
        f"Cell Assignment Heatmap  {sample}", fontweight="bold", fontsize=13, pad=10
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Fraction within predicted label", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    return
