import os

import pandas as pd
import numpy as np
import seaborn as sns
from collections import OrderedDict
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from copytyping.plot.plot_cnp import get_cn_colors, plot_cnv_legend, plot_cnv_profile
from copytyping.utils import NA_CELLTYPE, get_chr_sizes, is_tumor_label
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
        left = chr_offsets[chrs[i]]
        if i < len(chrs) - 2:
            right = chr_offsets[chrs[i + 1]]
        else:
            right = chr_end
        xtick_chrs.append((left + right) / 2)

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


def _merge_exp_lines(abs_starts, abs_ends, exp_vals, chrs):
    """Merge adjacent bins with the same expected value into one line segment."""
    lines = []
    chr_arr = np.asarray(chrs)
    n = len(exp_vals)
    i = 0
    while i < n:
        j = i + 1
        while j < n and exp_vals[j] == exp_vals[i] and chr_arr[j] == chr_arr[i]:
            j += 1
        lines.append([(abs_starts[i], exp_vals[i]), (abs_ends[j - 1], exp_vals[i])])
        i = j
    return lines


def build_wl_coords(cnv_blocks, wl_segments):
    """Map bins to the wl_segments coordinate system (same as plot_cnv_profile).

    Returns a dict with:
        positions, abs_starts, abs_ends — per-bin arrays (length G)
        x_edges, col_bin_ids — for pcolormesh grid (heatmap)
        ch_coords, seg_coords — chromosome / centromere boundary offsets
        chr_vlines, chr_end, xlab_chrs, xtick_chrs — axis decoration
    """
    cnv_blocks = cnv_blocks.reset_index(drop=True)
    chs = cnv_blocks["#CHR"].unique()
    wl_chs = wl_segments.groupby("#CHR", sort=False)
    bins_chs = cnv_blocks.groupby("#CHR", sort=False, observed=True)

    G = len(cnv_blocks)
    positions = np.full(G, np.nan)
    abs_starts = np.full(G, np.nan)
    abs_ends = np.full(G, np.nan)

    x_edges = [0.0]
    col_bin_ids = []

    ch_offset = 0.0
    ch_coords = []
    seg_coords = []

    for ch in chs:
        ch_coords.append(ch_offset)
        wl_ch = wl_chs.get_group(ch)
        bins_ch = bins_chs.get_group(ch)

        for si in range(len(wl_ch)):
            wl_row = wl_ch.iloc[si]
            wl_s, wl_e = wl_row["START"], wl_row["END"]
            seg_start = ch_offset
            seg_end = ch_offset + (wl_e - wl_s)

            in_seg = bins_ch[(bins_ch["START"] < wl_e) & (bins_ch["END"] > wl_s)]

            if in_seg.empty:
                if seg_end > x_edges[-1]:
                    col_bin_ids.append(-1)
                    x_edges.append(seg_end)
                ch_offset = seg_end
                if (si < len(wl_ch) - 1) or (si == 0 and wl_s > 0):
                    seg_coords.append(ch_offset)
                continue

            bin_starts = (
                np.maximum(in_seg["START"], wl_s) - wl_s + ch_offset
            ).to_numpy(float)
            bin_ends = (np.minimum(in_seg["END"], wl_e) - wl_s + ch_offset).to_numpy(
                float
            )
            bin_ids = in_seg.index.to_numpy()

            for idx, bs, be in zip(bin_ids, bin_starts, bin_ends):
                abs_starts[idx] = bs
                abs_ends[idx] = be
                positions[idx] = (bs + be) / 2

            ch_offset = seg_end
            if (si < len(wl_ch) - 1) or (si == 0 and wl_s != 0):
                seg_coords.append(ch_offset)

            cur = seg_start
            if seg_start > x_edges[-1]:
                col_bin_ids.append(-1)
                x_edges.append(seg_start)
                cur = seg_start

            for s, e, bid in zip(bin_starts, bin_ends, bin_ids):
                if s > cur:
                    col_bin_ids.append(-1)
                    x_edges.append(s)
                    cur = s
                if e > cur:
                    col_bin_ids.append(bid)
                    x_edges.append(e)
                    cur = e

            if cur < seg_end:
                col_bin_ids.append(-1)
                x_edges.append(seg_end)
    ch_coords.append(ch_offset)

    chr_end = ch_offset
    xlab_chrs = list(chs)
    xtick_chrs = [(ch_coords[i] + ch_coords[i + 1]) / 2 for i in range(len(chs))]
    chr_vlines = ch_coords[:-1]

    return {
        "positions": positions,
        "abs_starts": abs_starts,
        "abs_ends": abs_ends,
        "x_edges": np.asarray(x_edges, dtype=float),
        "col_bin_ids": col_bin_ids,
        "ch_coords": ch_coords,
        "seg_coords": seg_coords,
        "chr_vlines": chr_vlines,
        "chr_end": chr_end,
        "xlab_chrs": xlab_chrs,
        "xtick_chrs": xtick_chrs,
    }


def plot_rdr_baf_1d_pseudobulk(
    sx_data: SX_Data,
    anns: pd.DataFrame,
    base_props: np.ndarray,
    sample: str,
    data_type: str,
    genome_file: str,
    haplo_blocks: pd.DataFrame = None,
    wl_segments: pd.DataFrame = None,
    mask_cnp=True,
    mask_id="CNP",
    lab_type="cell_label",
    figsize=(20, 4),
    filename=None,
    log2=True,
    markersize=20,
    **kwargs,
):
    """
    Per-clone log2RDR + BAF scatter plot along the genome.
    Each clone = one PDF page with 2 rows: log2RDR (top) and BAF (bottom).

    Observed RDR = x_{g,n} / (T_n * lambda_g)
    Observed BAF = y_{g,n} / D_{g,n}
    """
    chrom_sizes = get_chr_sizes(genome_file)
    cnv_blocks = sx_data.cnv_blocks
    if mask_cnp:
        cnv_blocks = cnv_blocks.loc[sx_data.MASK[mask_id], :]

    exp_bafs = getattr(sx_data, "BAF", None)
    if exp_bafs is not None and mask_cnp:
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

    # genome coordinates — use wl_segments system when CNP row will be shown,
    # so scatter and plot_cnv_profile share the same x-axis range.
    has_cnp = haplo_blocks is not None and wl_segments is not None
    if has_cnp:
        wl = build_wl_coords(cnv_blocks, wl_segments)
        positions = wl["positions"]
        abs_starts = wl["abs_starts"]
        abs_ends = wl["abs_ends"]
        chr_vlines = wl["chr_vlines"]
        chr_end = wl["chr_end"]
        xlab_chrs = wl["xlab_chrs"]
        xtick_chrs = wl["xtick_chrs"]
    else:
        cnv_blocks["SAMPLE"] = sample
        chrs = cnv_blocks["#CHR"].unique().tolist()
        ret = build_ch_boundary(cnv_blocks, chrs, chrom_sizes, chr_shift=int(10e6))
        chr_offsets, chr_bounds, chr_gaps, chr_end, xlab_chrs, xtick_chrs = ret
        chr_vlines = list(chr_offsets.values())

        positions = cnv_blocks.apply(
            func=lambda r: chr_offsets[r["#CHR"]] + (r.START + r.END) // 2, axis=1
        ).to_numpy()
        abs_starts = cnv_blocks.apply(
            func=lambda r: chr_offsets[r["#CHR"]] + r.START, axis=1
        ).to_numpy()
        abs_ends = cnv_blocks.apply(
            func=lambda r: chr_offsets[r["#CHR"]] + r.END, axis=1
        ).to_numpy()

    linecolor = (0, 0, 0, 1)

    # Order labels: normal, clone1, clone2, ... (skip NA)
    ordered_labels = [x for x in ["normal"] if x in uniq_cell_labels] + sorted(
        [x for x in uniq_cell_labels if x.startswith("clone")]
    )
    rdr_label = "log2RDR" if log2 else "RDR"
    default_color = "grey"

    state_style, _ = get_cn_colors()

    # ── Build single-page figure ──
    # 2 rows (RDR + BAF) per clone, then CNP profile + CNP legend at bottom.
    # Use nested GridSpec: tight hspace within each clone pair, larger gap between.
    n_clones = len(ordered_labels)
    # outer grid: one slot per clone + one for bottom rows
    row_h = figsize[1] / 2
    fig_h = row_h * n_clones * 2 + (2 if has_cnp else 1)
    fig = plt.figure(figsize=(figsize[0], fig_h))

    outer_ratios = [2] * n_clones + [0.5 + 0.3 if has_cnp else 0.3]
    outer = GridSpec(
        n_clones + 1,
        1,
        figure=fig,
        height_ratios=outer_ratios,
        hspace=0.15,
        top=0.97,
    )

    axes = []
    for ci in range(n_clones):
        inner = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[ci], height_ratios=[1, 1], hspace=0.08
        )
        axes.append(fig.add_subplot(inner[0]))
        axes.append(fig.add_subplot(inner[1]))

    if has_cnp:
        inner_bot = GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=outer[n_clones],
            height_ratios=[0.5, 0.3],
            hspace=0.15,
        )
        axes.append(fig.add_subplot(inner_bot[0]))
        axes.append(fig.add_subplot(inner_bot[1]))
    else:
        axes.append(fig.add_subplot(outer[n_clones]))

    for ci, cell_label in enumerate(ordered_labels):
        ax_rdr = axes[ci * 2]
        ax_baf = axes[ci * 2 + 1]
        ax_rdr.set_zorder(2)
        ax_baf.set_zorder(2)
        ax_rdr.set_facecolor("white")
        ax_baf.set_facecolor("white")
        ax_rdr.sharex(ax_baf)

        barcode_idxs = anns[anns[lab_type] == cell_label].index.to_numpy()
        num_bcs = len(barcode_idxs)

        # Determine per-bin colors from (A,B) copy-number state
        bin_colors = [default_color] * len(cnv_blocks)
        clone_C_full = None
        if (
            cell_label != "NA"
            and hasattr(sx_data, "clones")
            and cell_label in sx_data.clones
        ):
            clone_idx = sx_data.clones.index(cell_label)
            C_normal_full = np.maximum(sx_data.C[:, 0], 1).astype(np.float64)
            clone_C_full = sx_data.C[:, clone_idx].astype(np.float64)
            clone_A = sx_data.A[:, clone_idx]
            clone_B = sx_data.B[:, clone_idx]
            if mask_cnp:
                C_normal_full = C_normal_full[sx_data.MASK[mask_id]]
                clone_C_full = clone_C_full[sx_data.MASK[mask_id]]
                clone_A = clone_A[sx_data.MASK[mask_id]]
                clone_B = clone_B[sx_data.MASK[mask_id]]
            bin_colors = [
                state_style.get((int(a), int(b)), state_style["default"])
                for a, b in zip(clone_A, clone_B)
            ]

        # ── RDR panel ──
        if masked_base_props is not None:
            agg_x = np.sum(X[:, barcode_idxs], axis=1).astype(np.float64)
            agg_T = np.sum(T[barcode_idxs]).astype(np.float64)
            rdr_valid = masked_base_props > 0
            obs_rdr = np.full(len(agg_x), np.nan)
            obs_rdr[rdr_valid] = agg_x[rdr_valid] / (
                agg_T * masked_base_props[rdr_valid]
            )
            if log2:
                log2_mask = rdr_valid & (obs_rdr > 0)
                obs_rdr[log2_mask] = np.log2(obs_rdr[log2_mask])
                obs_rdr[rdr_valid & ~log2_mask] = np.nan
            valid = rdr_valid & np.isfinite(obs_rdr)
            pos_rdr = positions[valid]
            val_rdr = obs_rdr[valid]
            rdr_colors = [bin_colors[j] for j in np.where(valid)[0]]
            ax_rdr.scatter(
                pos_rdr,
                val_rdr,
                s=markersize,
                c=rdr_colors,
                edgecolors="black",
                linewidths=0.1,
            )
            for coll in ax_rdr.collections:
                coll.set_rasterized(True)
            ax_rdr.vlines(
                chr_vlines,
                ymin=0,
                ymax=1,
                transform=ax_rdr.get_xaxis_transform(),
                linewidth=0.5,
                colors="k",
            )
            exp_vals = None
            if clone_C_full is not None:
                C_normal = np.maximum(sx_data.C[:, 0], 1).astype(np.float64)
                if mask_cnp:
                    C_normal = C_normal[sx_data.MASK[mask_id]]
                exp_vals = clone_C_full / C_normal
                if log2:
                    exp_vals = np.log2(np.maximum(exp_vals, 1e-6))
                ax_rdr.add_collection(
                    LineCollection(
                        _merge_exp_lines(
                            abs_starts, abs_ends, exp_vals, cnv_blocks["#CHR"]
                        ),
                        linewidth=1.5,
                        colors=[linecolor],
                    )
                )
            if log2:
                y_lo = min(val_rdr.min() * 1.1, -1.0) if len(val_rdr) else -1.0
                y_hi = max(val_rdr.max() * 1.1, 1.0) if len(val_rdr) else 1.0
                if exp_vals is not None:
                    y_lo = min(y_lo, float(exp_vals.min()) - 0.1)
                    y_hi = max(y_hi, float(exp_vals.max()) + 0.1)
                ax_rdr.set_ylim([y_lo, y_hi])
            else:
                exp_max = float(exp_vals.max()) if exp_vals is not None else 1.0
                ax_rdr.set_ylim(
                    [-0.1, min(max(val_rdr.max() * 1.1, exp_max * 1.1, 2.0), 6.0)]
                )
        ax_rdr.set_ylabel(rdr_label, fontsize=8)
        umi_label = "atac-fragment" if "atac" in data_type else "umi"
        total_gene = int(np.sum(X[:, barcode_idxs]))
        total_snp = int(np.sum(D[:, barcode_idxs]))
        ax_rdr.set_title(
            f"{cell_label} (n={num_bcs},"
            f" gene-{umi_label}={total_gene:,},"
            f" snp-{umi_label}={total_snp:,})",
            fontsize=9,
            fontweight="bold",
            loc="left",
        )
        plt.setp(ax_rdr, xlim=(0, chr_end), xticks=xtick_chrs)
        ax_rdr.set_xticklabels([])
        ax_rdr.tick_params(axis="x", bottom=False, top=False, labeltop=False)
        ax_rdr.grid(False)

        # ── BAF panel ──
        agg_bcounts = np.sum(Y[:, barcode_idxs], axis=1)
        agg_tcounts = np.sum(D[:, barcode_idxs], axis=1)
        baf_valid = agg_tcounts > 0
        agg_bafs = agg_bcounts[baf_valid] / agg_tcounts[baf_valid]
        pos_baf = positions[baf_valid]
        baf_colors = [bin_colors[j] for j in np.where(baf_valid)[0]]
        ax_baf.scatter(
            pos_baf,
            agg_bafs,
            s=markersize,
            c=baf_colors,
            edgecolors="black",
            linewidths=0.1,
        )
        for coll in ax_baf.collections:
            coll.set_rasterized(True)
        ax_baf.vlines(
            chr_vlines,
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
        if exp_bafs is not None and cell_label != "NA" and hasattr(sx_data, "clones"):
            clone_idx = sx_data.clones.index(cell_label)
            clone_baf = exp_bafs[:, clone_idx]
            ax_baf.add_collection(
                LineCollection(
                    _merge_exp_lines(
                        abs_starts, abs_ends, clone_baf, cnv_blocks["#CHR"]
                    ),
                    linewidth=1.5,
                    colors=[linecolor],
                )
            )
        ax_baf.set_ylim([-0.05, 1.05])
        ax_baf.set_ylabel("BAF", fontsize=8)
        plt.setp(ax_baf, xlim=(0, chr_end), xticks=xtick_chrs)
        ax_baf.tick_params(axis="x", top=False, labeltop=False)
        if ci == n_clones - 1:
            ax_baf.set_xticklabels(xlab_chrs, rotation=60, fontsize=8)
        else:
            ax_baf.set_xticklabels([])
            ax_baf.tick_params(axis="x", bottom=False)
        ax_baf.grid(False)

    # ── Bottom rows: CNP profile + legend ──
    if has_cnp:
        plot_cnv_profile(axes[-2], haplo_blocks, wl_segments, plot_chrname=False)
    plot_cnv_legend(axes[-1])

    fig.suptitle(
        f"sample={sample}  data_type={data_type}",
        fontsize=10,
    )
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
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


def _is_na_label(label):
    """Check if label is uninformative (NA/Unknown, case-insensitive)."""
    return label.lower() in {x.lower() for x in NA_CELLTYPE}


def plot_crosstab(
    assign_df: pd.DataFrame,
    sample: str,
    outfile: str,
    acol="copytyping-label",
    bcol="cell_type",
):
    """
    Plot cross-tabulation heatmap: rows = GT labels (bcol), cols = predicted (acol).

    Cells are white by default. Red highlights misassignments:
    - GT non-tumor assigned to clone*
    - GT tumor assigned to normal
    GT labels in NA_CELLTYPE (Unknown, NA) are never marked red.
    """
    ct = pd.crosstab(assign_df[bcol], assign_df[acol])

    gt_tumor = sorted([r for r in ct.index if is_tumor_label(r)])
    gt_normal = sorted([r for r in ct.index if not is_tumor_label(r)])
    row_order = gt_tumor + gt_normal

    pred_normal = [c for c in ct.columns if c == "normal"]
    pred_clone = sorted([c for c in ct.columns if c.startswith("clone")])
    pred_other = sorted([c for c in ct.columns if c not in pred_normal + pred_clone])
    col_order = pred_normal + pred_clone + pred_other

    row_order = [r for r in row_order if r in ct.index]
    col_order = [c for c in col_order if c in ct.columns]
    ct = ct.loc[row_order, col_order]
    counts = ct.to_numpy(dtype=float)
    num_rows, num_cols = counts.shape

    gt_tumor_mask = np.array([is_tumor_label(r) for r in row_order])
    pred_tumor_mask = np.array([c.startswith("clone") for c in col_order])
    tp = counts[np.ix_(gt_tumor_mask, pred_tumor_mask)].sum()
    fp = counts[np.ix_(~gt_tumor_mask, pred_tumor_mask)].sum()
    fn = counts[np.ix_(gt_tumor_mask, ~pred_tumor_mask)].sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    error = np.zeros((num_rows, num_cols), dtype=bool)
    for i, gt_lab in enumerate(row_order):
        if _is_na_label(gt_lab):
            continue
        gt_is_tumor = is_tumor_label(gt_lab)
        for j, pred_lab in enumerate(col_order):
            pred_is_tumor = pred_lab.startswith("clone")
            pred_is_normal = pred_lab == "normal"
            if gt_is_tumor and pred_is_normal:
                error[i, j] = True
            if not gt_is_tumor and pred_is_tumor:
                error[i, j] = True

    # Red for any error cell with count > 0; white otherwise
    rgba = np.ones((num_rows, num_cols, 4))
    for i in range(num_rows):
        for j in range(num_cols):
            if error[i, j] and counts[i, j] > 0:
                rgba[i, j] = [1.0, 0.85, 0.85, 1.0]

    cell_w = max(0.5, min(0.7, 5.0 / num_cols))
    cell_h = max(0.35, min(0.5, 4.0 / num_rows))
    fig_w = max(4, cell_w * num_cols + 2.5)
    fig_h = max(2.5, cell_h * num_rows + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(rgba, aspect="auto")

    font_size = max(5, min(8, int(160 / max(num_cols, num_rows))))
    for i in range(num_rows):
        for j in range(num_cols):
            n = int(counts[i, j])
            color = "grey" if n == 0 else "black"
            ax.text(
                j,
                i,
                str(n),
                ha="center",
                va="center",
                color=color,
                fontsize=font_size,
            )

    # Tick labels with counts and percentages
    total_n = counts.sum()
    col_sums = counts.sum(axis=0)
    row_sums_1d = counts.sum(axis=1)
    col_labels = [
        f"{c}\n({int(col_sums[j])}, {col_sums[j] / total_n * 100:.1f}%)"
        for j, c in enumerate(col_order)
    ]
    row_labels = [
        f"{r}\n({int(row_sums_1d[i])}, {row_sums_1d[i] / total_n * 100:.1f}%)"
        for i, r in enumerate(row_order)
    ]
    ax.set_xticks(np.arange(num_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(num_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.tick_params(length=0)
    ax.set_xlabel(f"predicted ({acol})", fontsize=10)
    ax.set_ylabel(f"reference ({bcol})", fontsize=10)
    ax.set_title(
        f"{sample}\nprec={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}",
        fontsize=11,
    )

    # Grid lines between cells
    for i in range(num_rows + 1):
        ax.axhline(i - 0.5, color="grey", linewidth=0.5)
    for j in range(num_cols + 1):
        ax.axvline(j - 0.5, color="grey", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
