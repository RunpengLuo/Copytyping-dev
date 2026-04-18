import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from copytyping.utils import NA_CELLTYPE, is_tumor_label


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
    metric: dict,
    acol="copytyping-label",
    bcol="cell_type",
):
    """Plot cross-tabulation heatmap: rows = GT labels (bcol), cols = predicted (acol)."""
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

    prec = metric.get("precision", np.nan)
    rec = metric.get("recall", np.nan)
    f1 = metric.get("f1", np.nan)

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
            ax.text(j, i, str(n), ha="center", va="center", color=color, fontsize=font_size)

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
    ax.set_title(f"{sample}\nprec={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}", fontsize=11)

    for i in range(num_rows + 1):
        ax.axhline(i - 0.5, color="grey", linewidth=0.5)
    for j in range(num_cols + 1):
        ax.axvline(j - 0.5, color="grey", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics_barplot(
    summary: pd.DataFrame,
    outfile: str,
    metrics=("precision", "recall", "f1"),
    sample_col="SAMPLE",
    dtypes_col="DATA_TYPES",
    group_col="cancer_type",
    dpi=200,
):
    """Bar plot of prec/recall/f1 per sample, one page per cancer_type group."""
    colors = {"precision": "#1f77b4", "recall": "#ff7f0e", "f1": "#2ca02c"}
    n_metrics = len(metrics)

    valid = summary.dropna(subset=list(metrics), how="all").copy()
    if valid.empty:
        return

    has_groups = group_col in valid.columns
    groups = sorted(valid[group_col].unique()) if has_groups else [None]

    with PdfPages(outfile) as pdf:
        for grp in groups:
            df = valid[valid[group_col] == grp] if grp is not None else valid
            if df.empty:
                continue

            sample_df = (
                df.sort_values("f1", ascending=False)
                .drop_duplicates(subset=[sample_col], keep="first")
                .sort_values(sample_col)
            )
            samples = sample_df[sample_col].tolist()
            if dtypes_col in sample_df.columns:
                tick_labels = [
                    f"{s}\n({dt})" for s, dt in zip(samples, sample_df[dtypes_col])
                ]
            else:
                tick_labels = samples
            n = len(samples)
            x = np.arange(n)
            bar_w = 1.0 / (n_metrics + 0.5)

            fig_w = max(6, n * 0.8 + 2)
            fig, ax = plt.subplots(figsize=(fig_w, 4))

            for mi, m in enumerate(metrics):
                vals = sample_df[m].astype(float).to_numpy()
                bars = ax.bar(
                    x + mi * bar_w, vals, width=bar_w,
                    color=colors.get(m, f"C{mi}"), label=m,
                )
                for bar, v in zip(bars, vals):
                    if np.isfinite(v):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2, v + 0.01,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=6,
                        )

            ax.set_xticks(x + bar_w * (n_metrics - 1) / 2)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Score")
            ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
            title = grp if grp else "all samples"
            ax.set_title(title, fontsize=11, fontweight="bold")
            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)
