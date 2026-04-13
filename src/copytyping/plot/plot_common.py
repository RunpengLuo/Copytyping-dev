import os

import pandas as pd
import numpy as np
import seaborn as sns
from collections import OrderedDict
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from copytyping.utils import get_chr_sizes, is_tumor_label
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
    ).to_numpy()
    abs_ends = cnv_blocks.apply(
        func=lambda r: chr_offsets[r["#CHR"]] + r.END, axis=1
    ).to_numpy()

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
            ax_rdr.vlines(
                list(chr_offsets.values()),
                ymin=0,
                ymax=1,
                transform=ax_rdr.get_xaxis_transform(),
                linewidth=0.5,
                colors="k",
            )
            ax_rdr.hlines(
                y=1.0,
                xmin=0,
                xmax=chr_end,
                colors="grey",
                linestyle=":",
                linewidth=1,
            )
            # expected RDR lines
            exp_rdr_max = 1.0
            if cell_label != "NA":
                clone_idx = sx_data.clones.index(cell_label)
                C_normal = np.maximum(sx_data.C[:, 0], 1).astype(np.float64)
                clone_C = sx_data.C[:, clone_idx].astype(np.float64)
                if mask_cnp:
                    C_normal = C_normal[sx_data.MASK[mask_id]]
                    clone_C = clone_C[sx_data.MASK[mask_id]]
                exp_rdr = clone_C / C_normal
                exp_rdr_lines = [
                    [(s, r), (t, r)] for s, t, r in zip(abs_starts, abs_ends, exp_rdr)
                ]
                ax_rdr.add_collection(
                    LineCollection(
                        exp_rdr_lines,
                        linewidth=2,
                        colors=[(0, 0, 0, 1)] * len(exp_rdr),
                    )
                )
                exp_rdr_max = float(exp_rdr.max())
            y_top = min(max(val_rdr.max() * 1.1, exp_rdr_max * 1.1, 2.0), 6.0)
            ax_rdr.set_ylim([-0.1, y_top])
        else:
            ax_rdr.text(
                0.5,
                0.5,
                "RDR not available (no baseline)",
                transform=ax_rdr.transAxes,
                ha="center",
                va="center",
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
    Plot cross-tabulation heatmap: rows = GT labels (bcol), cols = predicted (acol).

    Cells are white by default. Red highlights misassignments:
    - GT non-tumor assigned to clone*/NA
    - GT tumor assigned to normal/NA
    Raw counts annotated in each cell.
    """
    ct = pd.crosstab(assign_df[bcol], assign_df[acol])

    # Sort rows: tumor GT first, then non-tumor
    gt_tumor = sorted([r for r in ct.index if is_tumor_label(r)])
    gt_normal = sorted([r for r in ct.index if not is_tumor_label(r)])
    row_order = gt_tumor + gt_normal

    # Sort cols: normal, clone*, NA/other
    pred_normal = [c for c in ct.columns if c == "normal"]
    pred_clone = sorted([c for c in ct.columns if c.startswith("clone")])
    pred_other = sorted([c for c in ct.columns if c not in pred_normal + pred_clone])
    col_order = pred_normal + pred_clone + pred_other

    row_order = [r for r in row_order if r in ct.index]
    col_order = [c for c in col_order if c in ct.columns]
    ct = ct.loc[row_order, col_order]
    counts = ct.to_numpy(dtype=float)
    num_rows, num_cols = counts.shape

    # Compute precision / recall / f1 for the title
    gt_tumor_mask = np.array([is_tumor_label(r) for r in row_order])
    pred_tumor_mask = np.array([c.startswith("clone") for c in col_order])
    tp = counts[np.ix_(gt_tumor_mask, pred_tumor_mask)].sum()
    fp = counts[np.ix_(~gt_tumor_mask, pred_tumor_mask)].sum()
    fn = counts[np.ix_(gt_tumor_mask, ~pred_tumor_mask)].sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Build error mask: red for misassignments
    error = np.zeros((num_rows, num_cols), dtype=bool)
    for i, gt_lab in enumerate(row_order):
        gt_is_tumor = is_tumor_label(gt_lab)
        for j, pred_lab in enumerate(col_order):
            pred_is_tumor = pred_lab.startswith("clone")
            pred_is_normal = pred_lab == "normal"
            pred_is_na = pred_lab == "NA"
            if gt_is_tumor and (pred_is_normal or pred_is_na):
                error[i, j] = True
            if not gt_is_tumor and (pred_is_tumor or pred_is_na):
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
        f"{r}  ({int(row_sums_1d[i])}, {row_sums_1d[i] / total_n * 100:.1f}%)"
        for i, r in enumerate(row_order)
    ]
    ax.set_xticks(np.arange(num_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(num_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.tick_params(length=0)
    ax.set_xlabel(f"predicted ({acol})", fontsize=10)
    ax.set_ylabel(f"GT ({bcol})", fontsize=10)
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
