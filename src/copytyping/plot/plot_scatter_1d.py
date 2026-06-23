import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from copytyping.plot.plot_copynumber import (
    cnp_has_mirror,
    plot_ascn_legend,
    plot_ascn_profile,
    get_cn_colors,
    plot_cnv_legend,
    plot_cnv_profile,
)
from copytyping.plot.plot_common import build_wl_coords
from copytyping.utils import get_chr_sizes, read_whitelist_segments


##################################################
# genome-axis helpers
##################################################


def _build_ch_boundary(
    region_df: pd.DataFrame,
    chrs: list,
    chr_sizes: dict,
    chr_shift: int = 10_000_000,
):
    chr_offsets = OrderedDict()
    for i, ch in enumerate(chrs):
        if i == 0:
            chr_offsets[ch] = chr_shift
        else:
            prev_ch = chrs[i - 1]
            offset = chr_offsets[prev_ch] + chr_sizes[prev_ch]
            chr_offsets[ch] = offset
    chr_end = chr_offsets[chrs[-1]] + chr_sizes[chrs[-1]] + chr_shift
    xlab_chrs = chrs
    xtick_chrs = []
    for i in range(len(chrs)):
        left = chr_offsets[chrs[i]]
        if i < len(chrs) - 2:
            right = chr_offsets[chrs[i + 1]]
        else:
            right = chr_end
        xtick_chrs.append((left + right) / 2)

    chr_gaps = OrderedDict()
    dummy_sample = region_df["SAMPLE"].unique()[0]
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
    return (chr_offsets, chr_gaps, chr_end, xlab_chrs, xtick_chrs)


def _merge_exp_lines(
    abs_starts: np.ndarray,
    abs_ends: np.ndarray,
    exp_vals: np.ndarray,
    chrs: np.ndarray,
):
    """Merge adjacent bins with the same expected value into one line segment.

    Skips bins with NaN positions (outside whitelist regions) or NaN expected
    values (e.g. where the expectation is undefined / not applicable).
    """
    exp_vals = np.asarray(exp_vals, dtype=float)
    valid = np.isfinite(abs_starts) & np.isfinite(abs_ends) & np.isfinite(exp_vals)
    abs_starts = np.asarray(abs_starts)[valid]
    abs_ends = np.asarray(abs_ends)[valid]
    exp_vals = exp_vals[valid]
    chr_arr = np.asarray(chrs)[valid]

    lines = []
    n = len(exp_vals)
    i = 0
    while i < n:
        j = i + 1
        while j < n and exp_vals[j] == exp_vals[i] and chr_arr[j] == chr_arr[i]:
            j += 1
        lines.append([(abs_starts[i], exp_vals[i]), (abs_ends[j - 1], exp_vals[i])])
        i = j
    return lines


##################################################
# 1d scatter plots
##################################################


def plot_scatter_1d_pseudobulk(
    ax: Axes,
    positions: np.ndarray,
    obs_values: np.ndarray,
    chr_vlines: list,
    chr_end: float,
    xtick_chrs: list,
    xlab_chrs: list | None = None,
    exp_lines: list | None = None,
    colors: np.ndarray | None = None,
    ylabel: str = "value",
    ylim: tuple | None = None,
    markersize: int = 20,
    title: str | None = None,
    show_xticklabels: bool = True,
):
    """Scatter plot of per-bin pseudobulk values along the genome.

    Args:
        ax: matplotlib Axes to plot on.
        positions: (G,) array of genomic positions (absolute coordinates).
        obs_values: (G,) array of observed values (e.g. log2RDR or BAF).
        chr_vlines: list of x positions for chromosome boundary lines.
        chr_end: rightmost x coordinate.
        xtick_chrs: list of x positions for chromosome tick marks.
        xlab_chrs: list of chromosome labels. If None, xticklabels are hidden.
        exp_lines: list of [(x_start, y), (x_end, y)] line segments for
            expected values, or None.
        colors: per-bin colors (length G), or None for default.
        ylabel: y-axis label string.
        ylim: (ymin, ymax) tuple, or None for auto.
        markersize: scatter marker size.
        title: axes title string, or None.
        show_xticklabels: whether to show chromosome labels on x-axis.

    Returns:
        ax
    """
    valid = np.isfinite(obs_values) & np.isfinite(positions)
    c = colors if colors is not None else "steelblue"
    if isinstance(c, list):
        c = [c[j] for j in np.where(valid)[0]]
    ax.scatter(
        positions[valid],
        obs_values[valid],
        s=markersize,
        c=c,
        edgecolors="black",
        linewidths=0.1,
    )
    for coll in ax.collections:
        coll.set_rasterized(True)
    ax.vlines(
        chr_vlines,
        ymin=0,
        ymax=1,
        transform=ax.get_xaxis_transform(),
        linewidth=0.5,
        colors="k",
    )
    if exp_lines is not None:
        ax.add_collection(
            LineCollection(exp_lines, linewidth=1.5, colors=[(0, 0, 0, 1)])
        )
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    if title is not None:
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.set_xlim(0, chr_end)
    ax.set_xticks(xtick_chrs)
    if show_xticklabels and xlab_chrs is not None:
        ax.set_xticklabels(xlab_chrs, rotation=60, fontsize=11, fontweight="bold")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", bottom=False)
    ax.grid(False)
    return ax


def plot_rdr_baf_1d_pseudobulk(
    read_counts: np.ndarray,
    ballele_counts: np.ndarray,
    total_allele_counts: np.ndarray,
    cn_A: np.ndarray,
    cn_B: np.ndarray,
    cn_C: np.ndarray,
    cn_BAF: np.ndarray,
    clones: list[str],
    cnprofile: pd.DataFrame,
    anns: pd.DataFrame,
    base_props: np.ndarray,
    sample: str,
    assay_type: str,
    genome_file: str,
    region_bed: str,
    haplo_blocks: pd.DataFrame | None = None,
    lab_type: str = "cell_label",
    is_inferred: bool = True,
    figsize: tuple = (20, 4),
    filename: str | None = None,
    pdf_pages: PdfPages | None = None,
    log2: bool = True,
    rdr_ylim: tuple = (-5, 5),
    markersize: int = 20,
    ascn_profile: bool = False,
    **kwargs,
):
    """Per-clone log2RDR + BAF scatter plot along the genome, single page.

    Observed RDR = read_count_{g,n} / (library_size_n * lambda_g)
    Observed BAF = ballele_{g,n} / total_allele_{g,n}

    Args:
        read_counts: (G, N) read depth / feature counts.
        ballele_counts: (G, N) B-allele counts.
        total_allele_counts: (G, N) total-allele counts (A + B).
        cn_A/cn_B/cn_C: (G, K) per-clone copy numbers.
        cn_BAF: (G, K) per-clone expected B-allele frequency.
        clones: clone names, length K.
        cnprofile: per-row genomic coordinates (#CHR/START/END).
        haplo_blocks: CN profile drawn as the bottom strip (or None).
    """
    chrom_sizes = get_chr_sizes(genome_file)
    cnprofile = cnprofile[["#CHR", "START", "END"]].copy(deep=True)
    total_cells = len(anns)

    library_size = read_counts.sum(axis=0).astype(np.float64)

    cell_labels = anns[lab_type].tolist()
    uniq_cell_labels = anns[lab_type].unique()
    assert ballele_counts.shape[0] == len(cnprofile)
    assert ballele_counts.shape[1] == len(cell_labels)

    # genome coordinates
    wl_segments = read_whitelist_segments(region_bed)
    has_cnp = haplo_blocks is not None
    if has_cnp:
        wl = build_wl_coords(cnprofile, wl_segments)
        positions = wl["positions"]
        abs_starts = wl["abs_starts"]
        abs_ends = wl["abs_ends"]
        chr_vlines = wl["chr_vlines"]
        chr_end = wl["chr_end"]
        xlab_chrs = wl["xlab_chrs"]
        xtick_chrs = wl["xtick_chrs"]
    else:
        cnprofile["SAMPLE"] = sample
        chrs = cnprofile["#CHR"].unique().tolist()
        ret = _build_ch_boundary(cnprofile, chrs, chrom_sizes, chr_shift=int(10e6))
        chr_offsets, chr_gaps, chr_end, xlab_chrs, xtick_chrs = ret
        chr_vlines = list(chr_offsets.values())

        positions = cnprofile.apply(
            func=lambda r: chr_offsets[r["#CHR"]] + (r.START + r.END) // 2, axis=1
        ).to_numpy()
        abs_starts = cnprofile.apply(
            func=lambda r: chr_offsets[r["#CHR"]] + r.START, axis=1
        ).to_numpy()
        abs_ends = cnprofile.apply(
            func=lambda r: chr_offsets[r["#CHR"]] + r.END, axis=1
        ).to_numpy()

    if is_inferred:
        # Order: normal, clone1, clone2, ..., then other non-NA labels
        ordered_labels = (
            [x for x in ["normal"] if x in uniq_cell_labels]
            + sorted([x for x in uniq_cell_labels if x.startswith("clone")])
            + sorted(
                [
                    x
                    for x in uniq_cell_labels
                    if x != "normal" and not x.startswith("clone") and x != "NA"
                ]
            )
        )
    else:
        # External label: keep original order, skip NA
        ordered_labels = [x for x in uniq_cell_labels if x != "NA"]
    rdr_label = "log2RDR" if log2 else "RDR"
    default_color = "grey"

    state_style, _ = get_cn_colors()

    # ── Build single-page figure ──
    n_clones = len(ordered_labels)
    row_h = figsize[1] / 2
    fig_h = row_h * n_clones * 2 + (2 if has_cnp else 1)
    fig = plt.figure(figsize=(figsize[0], fig_h))

    outer_ratios = [2] * n_clones + [0.5 + 0.3 if has_cnp else 0.3]
    outer = GridSpec(
        n_clones + 1,
        1,
        figure=fig,
        height_ratios=outer_ratios,
        hspace=0.35,
        top=0.97,
    )

    axes = []
    for ci in range(n_clones):
        inner = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[ci], height_ratios=[1, 1], hspace=0.25
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

        barcode_idxs = anns[anns[lab_type] == cell_label].index.to_numpy()
        num_bcs = len(barcode_idxs)

        # per-bin colors from (A,B) copy-number state
        bin_colors = [default_color] * len(cnprofile)
        clone_C_full = None
        if (
            is_inferred
            and cell_label != "NA"
            and clones is not None
            and cell_label in clones
        ):
            clone_idx = clones.index(cell_label)
            clone_C_full = cn_C[:, clone_idx].astype(np.float64)
            clone_A = cn_A[:, clone_idx]
            clone_B = cn_B[:, clone_idx]
            bin_colors = [
                state_style.get((int(a), int(b)), state_style["default"])
                for a, b in zip(clone_A, clone_B)
            ]

        # ── RDR panel ──
        rdr_exp_lines = None
        rdr_ylim_eff = rdr_ylim
        if base_props is not None:
            agg_x = np.sum(read_counts[:, barcode_idxs], axis=1).astype(np.float64)
            agg_T = np.sum(library_size[barcode_idxs]).astype(np.float64)
            rdr_valid = base_props > 0
            obs_rdr = np.full(len(agg_x), np.nan)
            obs_rdr[rdr_valid] = agg_x[rdr_valid] / (agg_T * base_props[rdr_valid])
            if log2:
                log2_mask = rdr_valid & (obs_rdr > 0)
                obs_rdr[log2_mask] = np.log2(obs_rdr[log2_mask])
                obs_rdr[rdr_valid & ~log2_mask] = np.nan
            exp_vals = None
            if clone_C_full is not None:
                denom = float(np.sum(base_props * clone_C_full))
                if denom > 0:
                    exp_vals = clone_C_full / denom
                    if log2:
                        exp_vals = np.log2(np.maximum(exp_vals, 1e-6))
            if exp_vals is not None:
                rdr_exp_lines = _merge_exp_lines(
                    abs_starts, abs_ends, exp_vals, cnprofile["#CHR"]
                )
            if log2:
                candidates = [obs_rdr[np.isfinite(obs_rdr)]]
                if exp_vals is not None:
                    candidates.append(np.asarray(exp_vals))
                all_vals = np.concatenate(
                    [c[np.isfinite(c)] for c in candidates if len(c) > 0]
                )
                if all_vals.size > 0 and all_vals.min() >= -2 and all_vals.max() <= 2:
                    rdr_ylim_eff = (-2, 2)
            else:
                exp_max = float(exp_vals.max()) if exp_vals is not None else 1.0
                valid_rdr = obs_rdr[np.isfinite(obs_rdr)]
                rdr_ylim_eff = (
                    (
                        -0.1,
                        min(max(valid_rdr.max() * 1.1, exp_max * 1.1, 2.0), 6.0),
                    )
                    if valid_rdr.size > 0
                    else (-0.1, 2.0)
                )
        else:
            obs_rdr = np.full(len(positions), np.nan)

        feat_label = {"atac": "fragment", "gex": "umi"}.get(assay_type, "count")
        total_counts = int(np.sum(read_counts[:, barcode_idxs]))
        snp_counts = int(np.sum(total_allele_counts[:, barcode_idxs]))
        prop = round(100 * num_bcs / total_cells, 1) if total_cells > 0 else 0.0
        rdr_title = (
            f"{cell_label} (n={num_bcs}, prop={prop}%,"
            f" {assay_type}-{feat_label}={total_counts:,},"
            f" snp-{feat_label}={snp_counts:,})"
        )
        plot_scatter_1d_pseudobulk(
            ax_rdr,
            positions,
            obs_rdr,
            chr_vlines,
            chr_end,
            xtick_chrs,
            xlab_chrs,
            exp_lines=rdr_exp_lines,
            colors=bin_colors,
            ylabel=rdr_label,
            ylim=rdr_ylim_eff,
            markersize=markersize,
            title=rdr_title,
            show_xticklabels=False,
        )

        # ── BAF panel ──
        agg_bcounts = np.sum(ballele_counts[:, barcode_idxs], axis=1).astype(np.float64)
        agg_tcounts = np.sum(total_allele_counts[:, barcode_idxs], axis=1).astype(
            np.float64
        )
        obs_baf = np.divide(
            agg_bcounts,
            agg_tcounts,
            out=np.full_like(agg_bcounts, np.nan),
            where=agg_tcounts > 0,
        )
        baf_exp_lines = None
        if (
            is_inferred
            and cn_BAF is not None
            and clones is not None
            and cell_label in clones
        ):
            clone_idx = clones.index(cell_label)
            clone_baf = cn_BAF[:, clone_idx].copy()
            baf_exp_lines = _merge_exp_lines(
                abs_starts, abs_ends, clone_baf, cnprofile["#CHR"]
            )
        plot_scatter_1d_pseudobulk(
            ax_baf,
            positions,
            obs_baf,
            chr_vlines,
            chr_end,
            xtick_chrs,
            xlab_chrs,
            exp_lines=baf_exp_lines,
            colors=bin_colors,
            ylabel="BAF",
            ylim=(-0.05, 1.05),
            markersize=markersize,
            show_xticklabels=True,
        )

    # ── Bottom rows: CNP profile + legend ──
    if has_cnp:
        ax_cnp = axes[-2]
        if ascn_profile:
            plot_ascn_profile(ax_cnp, haplo_blocks, wl_segments, plot_chrname=False)
        else:
            plot_cnv_profile(ax_cnp, haplo_blocks, wl_segments, plot_chrname=False)
        ax_cnp.set_xlim(0, chr_end)
    if ascn_profile:
        plot_ascn_legend(axes[-1])
    else:
        plot_cnv_legend(axes[-1], has_mirror=has_cnp and cnp_has_mirror(haplo_blocks))

    title = f"sample={sample}  platform={kwargs.get('platform', '')}  assay_type={assay_type}"
    if kwargs.get("subtitle"):
        title += f"\n{kwargs['subtitle']}"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    if pdf_pages is not None:
        pdf_pages.savefig(fig, dpi=150, bbox_inches="tight")
    elif filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
