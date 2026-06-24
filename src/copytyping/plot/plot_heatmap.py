import logging

import pandas as pd
import numpy as np

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle

from copytyping.utils import read_whitelist_segments
from copytyping.inference.model_utils import empirical_baf_gn, empirical_rdr_gn
from copytyping.plot.plot_copynumber import (
    cnp_has_mirror,
    plot_ascn_legend,
    plot_ascn_profile,
    plot_cnv_legend,
    plot_cnv_profile,
)
from copytyping.plot.plot_common import (
    BAF_COLORS,
    BLACK,
    PURITY_CMAP,
    build_label_color_maps,
    build_wl_coords,
    make_baf_cmap,
)

# pretty display names for label columns (strip titles + legend titles)
_LABEL_DISPLAY = {"copytyping_label": "Copy-typing"}


def _display_name(name: str):
    return _LABEL_DISPLAY.get(name, name)


# Silence fontTools subsetter chatter
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
logging.getLogger("fontTools.ttLib").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.ERROR)


def plot_heatmap(
    ax: plt.Axes,
    cell_labels: np.ndarray,
    cnprofile: pd.DataFrame,
    X_mat: np.ndarray,
    wl_segments: pd.DataFrame,
    height: int = 1,
    plot_chrname: bool = True,
    title: str | None = None,
    ylabel: str | None = None,
    cmap: mcolors.Colormap | None = None,
    norm: mcolors.Normalize | None = None,
    show_block_labels: bool = True,
):
    (N, G) = X_mat.shape
    assert len(cnprofile) == G, "unmatched data"
    assert len(cell_labels) == N, "unmatched data"

    wl = build_wl_coords(cnprofile, wl_segments)
    x_edges = wl["x_edges"]
    col_bin_ids = wl["col_bin_ids"]
    ch_coords = wl["ch_coords"]
    seg_coords = wl["seg_coords"]
    ch_offset = wl["chr_end"]
    chs = cnprofile["#CHR"].unique()

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
        ax.set_xticklabels(chs, rotation=60, fontsize=11, fontweight="bold")
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

    if show_block_labels:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=11, fontweight="bold")
    else:
        # labels live in the left color strips / right legends instead
        ax.set_yticks([])
    ax.tick_params(axis="y", length=0)

    if ylabel is not None:
        ax.set_ylabel(ylabel, rotation=0, ha="right", va="center")
    if title is not None:
        ax.set_title(title)
    return x_edges, y_edges, C


def _row_layout(
    cell_labels: np.ndarray,
    uniq_labels: list,
    agg_size: int,
    secondary: list[np.ndarray] | None = None,
):
    """Per-output-row original cell indices, grouping agg_size cells within each label.

    Rows are emitted in uniq_labels order (bottom-to-top in the heatmap). Within each
    primary-label block, cells are first sub-sorted by ``secondary`` (successive label
    arrays, e.g. cell_type) so same-secondary cells cluster before agg chunking.
    """
    groups = []
    for lab in uniq_labels:
        idx = np.where(cell_labels == lab)[0]
        if secondary:
            # lexsort: last key is primary, so reverse to make secondary[0] primary
            codes = [np.unique(s, return_inverse=True)[1][idx] for s in secondary]
            idx = idx[np.lexsort(tuple(reversed(codes)))]
        for g in range(0, len(idx), agg_size):
            sub = idx[g : g + agg_size]
            if len(sub) > 0:
                groups.append(sub)
    return groups


def _aggregate_columns(mat: np.ndarray, row_groups: list[np.ndarray]):
    """Sum mat[:, group] over each row group -> (n_bins, n_rows)."""
    return np.column_stack([mat[:, g].sum(axis=1) for g in row_groups])


def _mode(arr: np.ndarray):
    """Most frequent value in arr (ties broken by sort order)."""
    vals, counts = np.unique(arr, return_counts=True)
    return vals[np.argmax(counts)]


def prepare_rdr(
    read_counts: np.ndarray,
    row_groups: list[np.ndarray],
    base_props: np.ndarray,
    log2: bool = True,
):
    library_size = read_counts.sum(axis=0)
    count_X = _aggregate_columns(read_counts, row_groups)
    count_T = np.array([library_size[g].sum() for g in row_groups], dtype=np.int64)
    return empirical_rdr_gn(count_X, count_T, base_props, log2=log2).T


def prepare_pi_gk(read_counts: np.ndarray, row_groups: list[np.ndarray]):
    library_size = read_counts.sum(axis=0)
    X = _aggregate_columns(read_counts, row_groups)
    T = np.array([library_size[g].sum() for g in row_groups], dtype=np.int64)
    pi_gk_matrix = X / T[None, :]
    pi_gk_matrix[pi_gk_matrix == 0] = np.nan
    return pi_gk_matrix.T


def prepare_baf(
    ballele_counts: np.ndarray,
    total_allele_counts: np.ndarray,
    row_groups: list[np.ndarray],
):
    count_B = _aggregate_columns(ballele_counts, row_groups)
    count_N = _aggregate_columns(total_allele_counts, row_groups)
    return empirical_baf_gn(count_B, count_N).T


def plot_label_strips(
    fig: plt.Figure,
    base_ax: plt.Axes,
    y_edges: np.ndarray,
    row_label_map: dict[str, np.ndarray],
    color_maps: dict[str, dict[str, str]],
    strip_width: float = 0.012,
    gap: float = 0.004,
):
    """Draw vertical categorical color strips to the LEFT of base_ax, one per label.

    row_label_map maps label name -> (n_rows,) values in bottom-to-top row order.
    color_maps maps label name -> {value: color}. Returns
    [(name, {value: color}, {value: fraction})], where fraction is the share of
    rows holding that value (used to annotate legend entries).
    """
    fig.canvas.draw()
    bbox = base_ax.get_position()
    x_cursor = bbox.x0 - gap
    legends_info = []
    for name, values in row_label_map.items():
        values = np.array([str(v) for v in values])
        n_rows = max(len(values), 1)
        prop_dict = {v: int((values == v).sum()) / n_rows for v in color_maps[name]}
        color_dict = color_maps[name]
        order = list(color_dict)
        codes = np.array([order.index(v) for v in values], dtype=float)[:, None]
        x_cursor -= strip_width
        ax = fig.add_axes([x_cursor, bbox.y0, strip_width, bbox.height])
        strip_cmap = mcolors.ListedColormap([color_dict[v] for v in order])
        strip_norm = mcolors.BoundaryNorm(np.arange(len(order) + 1) - 0.5, strip_cmap.N)
        ax.pcolormesh(
            np.array([0.0, 1.0]),
            y_edges,
            codes,
            cmap=strip_cmap,
            norm=strip_norm,
            shading="flat",
            rasterized=True,
        )
        ax.set_xticks([0.5])
        # vertical, bold strip title (thin strips → vertical avoids overlap)
        ax.set_xticklabels(
            [_display_name(name)], rotation=90, fontsize=11, fontweight="bold"
        )
        ax.tick_params(
            axis="x", labeltop=True, labelbottom=False, top=False, bottom=False
        )
        ax.set_yticks([])
        ax.set_ylim(base_ax.get_ylim())
        x_cursor -= gap
        legends_info.append((name, color_dict, prop_dict))
    return legends_info


def draw_label_legends(
    fig: plt.Figure,
    base_ax: plt.Axes,
    legends_info: list[tuple[str, dict[str, str], dict[str, float]]],
    x0: float,
    entry_h: float = 0.038,
    gap: float = 0.06,
):
    """Stack one borderless categorical legend per label, top-aligned, at figure-x x0.

    Legend titles are bold; entries are large for readability. Each entry is
    annotated with the value's share of rows, e.g. ``clone1 (10.12%)``.
    """
    fig.canvas.draw()
    bbox = base_ax.get_position()
    y_top = bbox.y1
    for name, color_dict, prop_dict in legends_info:
        handles = [
            Patch(facecolor=col, label=f"{v} ({prop_dict.get(v, 0.0) * 100:.2f}%)")
            for v, col in color_dict.items()
        ]
        leg = fig.legend(
            handles=handles,
            title=_display_name(name),
            loc="upper left",
            bbox_to_anchor=(x0, y_top),
            frameon=False,
            fontsize=13,
            title_fontsize=15,
        )
        leg.get_title().set_fontweight("bold")
        fig.add_artist(leg)
        y_top -= entry_h * (len(handles) + 1) + gap


def plot_cnv_heatmap(
    sample: str,
    assay_type: str,
    haplo_blocks: pd.DataFrame,
    read_counts: np.ndarray,
    ballele_counts: np.ndarray,
    total_allele_counts: np.ndarray,
    cnprofile: pd.DataFrame,
    num_clones: int,
    anns: pd.DataFrame,
    region_bed: str,
    proportions: np.ndarray | None = None,
    val: str = "BAF",
    base_props: np.ndarray | None = None,
    agg_size: int = 5,
    label_cols: list | None = None,
    primary_label: str | None = None,
    figsize: tuple = (20, 13),
    hratios: list = [10, 2, 2],
    filename: str | None = None,
    pdf_pages: PdfPages | None = None,
    dpi: int = 300,
    transparent: bool = False,
    rep_id: str = "",
    ascn_profile: bool = False,
    color_maps: dict | None = None,
):
    assert val in ["BAF", "RDR", "log2RDR", "COUNT", "pi_gk"]
    wl_fragments = read_whitelist_segments(region_bed)
    logging.debug(f"plot CNV heatmap val={val}")

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    if label_cols is None:
        label_cols = []
    if primary_label is None:
        primary_label = label_cols[0] if label_cols else None

    num_cells = read_counts.shape[1]
    if primary_label is None or anns is None:
        primary_labels = np.full(num_cells, fill_value="unknown")
    else:
        primary_labels = anns[primary_label].to_numpy()
    assert len(primary_labels) == num_cells

    # order cells (bottom-to-top) by the primary label, then aggregate within label
    uniq_labels = list(np.unique(primary_labels))
    if primary_label and primary_label.startswith("copytyping_label"):
        # pcolormesh y=0 is bottom, so reverse desired top-to-bottom order
        desired = ["NA", "normal"] + [f"clone{c}" for c in range(1, num_clones)]
        present = set(primary_labels)
        uniq_labels = [lab for lab in reversed(desired) if lab in present]

    # within each primary block, sub-sort cells by the other label columns
    # (e.g. cell_type) so same-secondary cells cluster before agg chunking
    secondary_vals = (
        [
            anns[c].to_numpy()
            for c in label_cols
            if c != primary_label and c in anns.columns
        ]
        if anns is not None
        else None
    )
    row_groups = _row_layout(
        primary_labels, uniq_labels, agg_size, secondary=secondary_vals
    )
    row_primary = np.array([primary_labels[g[0]] for g in row_groups])

    data_info = cnprofile
    if val == "BAF":
        data_matrix = prepare_baf(ballele_counts, total_allele_counts, row_groups)
        cmap, norm = make_baf_cmap()
        cticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    elif val in ["RDR", "log2RDR"]:
        data_matrix = prepare_rdr(read_counts, row_groups, base_props, val == "log2RDR")
        cmap = "coolwarm"
        if val == "log2RDR":
            norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
            cticks = [-1.0, -0.50, 0.0, 0.50, 1.0]
        else:
            norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2)
            cticks = [0.0, 0.5, 1.0, 1.50, 2.0]
    elif val == "pi_gk":
        data_matrix = prepare_pi_gk(read_counts, row_groups)
        cticks = [0.0, 0.25, 0.5, 0.75, 1.0]
        cmap = mcolors.LinearSegmentedColormap.from_list("pi_cont", BAF_COLORS, N=256)
        norm = mcolors.Normalize(
            vmin=np.nanmin(data_matrix), vmax=np.nanmax(data_matrix)
        )

    # per-row purity (mean over each aggregated group)
    row_props = None
    if proportions is not None:
        row_props = np.array([np.mean(proportions[g]) for g in row_groups])

    # within each primary-label block, order rows by DESC purity
    if row_props is not None:
        logging.debug("order rows by tumor proportions")
        n_rows = len(row_groups)
        change_pts = np.flatnonzero(row_primary[1:] != row_primary[:-1]) + 1
        block_bounds = np.r_[0, change_pts, n_rows]
        perm = np.arange(n_rows)
        for start, end in zip(block_bounds[:-1], block_bounds[1:]):
            order = np.argsort(row_props[start:end])[::-1]
            perm[start:end] = perm[start:end][order]
        row_groups = [row_groups[i] for i in perm]
        row_primary = row_primary[perm]
        data_matrix = data_matrix[perm]
        row_props = row_props[perm]

    # per-row value of every requested label column (mode within each group)
    row_label_map = {}
    if anns is not None:
        for col in label_cols:
            if col not in anns.columns:
                continue
            col_vals = anns[col].to_numpy()
            row_label_map[col] = np.array([_mode(col_vals[g]) for g in row_groups])
    # use the shared palette if provided (consistent across all plots), else
    # build one from this heatmap's own (aggregated) label values
    if color_maps is None:
        color_maps = build_label_color_maps(row_label_map, primary_label)

    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=figsize, gridspec_kw={"height_ratios": hratios}
    )
    fig.subplots_adjust(top=0.99, right=0.95, hspace=0.03)

    x_edges, y_edges, _ = plot_heatmap(
        axes[0],
        row_primary,
        data_info,
        data_matrix,
        wl_fragments,
        height=10,
        cmap=cmap,
        norm=norm,
        show_block_labels=False,
    )

    if ascn_profile:
        plot_ascn_profile(axes[1], haplo_blocks, wl_fragments, plot_chrname=False)
        plot_ascn_legend(axes[2])
    else:
        plot_cnv_profile(axes[1], haplo_blocks, wl_fragments, plot_chrname=False)
        plot_cnv_legend(axes[2], has_mirror=cnp_has_mirror(haplo_blocks))

    title = f"{sample} {rep_id} {assay_type} {val} Heatmap".replace("  ", " ")
    if agg_size > 1:
        title += f"\n(pseudobulk {agg_size} cell for visualization)"
    fig.suptitle(title, y=0.99, fontsize=14, fontweight="bold")

    fig.tight_layout(rect=[0.0, 0.0, 0.95, 0.99])

    # left: one categorical color strip per label column (clone label, cell type, ...)
    legends_info = plot_label_strips(fig, axes[0], y_edges, row_label_map, color_maps)

    extra_pad = 0.0
    if row_props is not None:
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
        C = row_props[:, None]
        purity_cmap = PURITY_CMAP
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

    # right: one categorical legend per label column (value -> color)
    if legends_info:
        legend_x0 = bbox.x1 + extra_pad + 0.05
        draw_label_legends(fig, axes[0], legends_info, x0=legend_x0)

    if pdf_pages is not None:
        pdf_pages.savefig(fig, dpi=dpi, bbox_inches="tight", transparent=transparent)
        plt.close(fig)
        return
    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight", transparent=transparent)
        plt.close()
        return
    plt.show()
    return
