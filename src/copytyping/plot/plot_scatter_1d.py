import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from copytyping.plot.plot_copynumber import (
    get_cn_colors,
    plot_cnv_legend,
    plot_cnv_profile,
)
from copytyping.plot.plot_common import build_wl_coords
from copytyping.sx_data.sx_data import SX_Data
from copytyping.utils import get_chr_sizes


def _build_ch_boundary(
    region_df: pd.DataFrame, chrs: list, chr_sizes: dict, chr_shift=10_000_000
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


def _merge_exp_lines(abs_starts, abs_ends, exp_vals, chrs):
    """Merge adjacent bins with the same expected value into one line segment.

    Skips bins with NaN positions (outside whitelist regions).
    """
    valid = np.isfinite(abs_starts) & np.isfinite(abs_ends)
    abs_starts = np.asarray(abs_starts)[valid]
    abs_ends = np.asarray(abs_ends)[valid]
    exp_vals = np.asarray(exp_vals)[valid]
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


def plot_rdr_baf_1d_pseudobulk(
    sx_data: SX_Data,
    anns: pd.DataFrame,
    base_props: np.ndarray,
    sample: str,
    data_type: str,
    genome_file: str,
    haplo_blocks: pd.DataFrame = None,
    wl_segments: pd.DataFrame = None,
    resolution: str = "seg",
    mask_cnp=True,
    mask_id="CNP",
    lab_type="cell_label",
    is_inferred=True,
    figsize=(20, 4),
    filename=None,
    log2=True,
    markersize=20,
    **kwargs,
):
    """Per-clone log2RDR + BAF scatter plot along the genome, single page.

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

    X = sx_data.X
    Y = sx_data.Y
    D = sx_data.D
    T = sx_data.T
    if mask_cnp:
        mask = sx_data.MASK[mask_id]
        X = X[mask]
        Y = Y[mask]
        D = D[mask]

    masked_base_props = base_props
    if masked_base_props is not None and mask_cnp:
        masked_base_props = masked_base_props[sx_data.MASK[mask_id]]

    cell_labels = anns[lab_type].tolist()
    uniq_cell_labels = anns[lab_type].unique()
    assert Y.shape[0] == len(cnv_blocks)
    assert Y.shape[1] == len(cell_labels)

    # genome coordinates
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
        ret = _build_ch_boundary(cnv_blocks, chrs, chrom_sizes, chr_shift=int(10e6))
        chr_offsets, chr_gaps, chr_end, xlab_chrs, xtick_chrs = ret
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
        hspace=0.20,
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

        barcode_idxs = anns[anns[lab_type] == cell_label].index.to_numpy()
        num_bcs = len(barcode_idxs)

        # per-bin colors from (A,B) copy-number state
        bin_colors = [default_color] * len(cnv_blocks)
        clone_C_full = None
        if (
            is_inferred
            and cell_label != "NA"
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
        feat_label = {"atac": "fragment", "gex": "umi"}.get(data_type, "count")
        total_counts = int(np.sum(X[:, barcode_idxs]))
        snp_counts = int(np.sum(D[:, barcode_idxs]))
        ax_rdr.set_title(
            f"{cell_label} (n={num_bcs},"
            f" {data_type}-{feat_label}={total_counts:,},"
            f" snp-{feat_label}={snp_counts:,})",
            fontsize=9,
            fontweight="bold",
            loc="left",
        )
        plt.setp(ax_rdr, xlim=(0, chr_end), xticks=xtick_chrs)
        ax_rdr.set_xticklabels([])
        ax_rdr.tick_params(axis="x", bottom=False)
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
        if (
            is_inferred
            and exp_bafs is not None
            and hasattr(sx_data, "clones")
            and cell_label in sx_data.clones
        ):
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
        if ci == n_clones - 1:
            ax_baf.set_xticklabels(xlab_chrs, rotation=60, fontsize=8)
        else:
            ax_baf.set_xticklabels([])
            ax_baf.tick_params(axis="x", bottom=False)
        ax_baf.grid(False)

    # ── Bottom rows: CNP profile + legend ──
    if has_cnp:
        ax_cnp = axes[-2]
        plot_cnv_profile(ax_cnp, haplo_blocks, wl_segments, plot_chrname=False)
        ax_cnp.set_xlim(0, chr_end)
    plot_cnv_legend(axes[-1])

    title = (
        f"sample={sample}  platform={kwargs.get('platform', '')}  data_type={data_type}"
    )
    if kwargs.get("subtitle"):
        title += f"\n{kwargs['subtitle']}"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
