import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from copytyping.utils import INVALID_LABELS, NA_CELLTYPE


# ── Unified label color palette (shared by heatmap + visium) ──
# normal-like -> gray, invalid/NA -> dark gray, tumor clones -> qualitative palette
NORMAL_COLOR = "lightgray"
NA_COLOR = "darkgray"
BLACK = (0, 0, 0, 1)
_TUMOR_COLORS = [mcolors.to_hex(c) for c in plt.get_cmap("tab10").colors]

# Sequential purity / posterior overlay; shared by heatmap + visium.
PURITY_CMAP = "magma_r"

# Diverging BAF palette (blue = A-skewed, gray = balanced, red = B-skewed),
# 10 discrete bins. Shared by heatmap (BAF/pi_gk) and visium LOH BAF.
BAF_COLORS = [
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

# Qualitative palettes for non-clone label sets (cell_type, path annotation, ...).
# The clone/primary label always uses the tab10 clone scheme (build_label_colors);
# each additional label set gets its own palette here, indexed by slot.
_CATEGORICAL_PALETTES = ["Set2", "Dark2", "Set3", "tab20b"]


def _is_normal_like(label: str):
    """True for the diploid/normal reference label (e.g. 'normal', 'Normal_cell')."""
    return str(label).lower().startswith("normal")


def _label_color_index(label: str):
    """Stable color index for a label: clone1->0, clone2->1, ...; others by hash."""
    m = re.match(r"clone(\d+)", str(label))
    if m:
        return int(m.group(1)) - 1
    return hash(str(label)) % len(_TUMOR_COLORS)


def build_label_colors(categories: list, clone_indexed: bool = True):
    """Clone-label colors: INVALID->NA_COLOR, 'normal'->NORMAL_COLOR, cloneN->tab10[N-1].

    Consistent regardless of subset/order.
    """
    colors = []
    tumor_i = 0
    for c in categories:
        if c in INVALID_LABELS:
            colors.append(NA_COLOR)
        elif c == "normal":
            colors.append(NORMAL_COLOR)
        elif clone_indexed:
            colors.append(_TUMOR_COLORS[_label_color_index(c) % len(_TUMOR_COLORS)])
        else:
            colors.append(_TUMOR_COLORS[tumor_i % len(_TUMOR_COLORS)])
            tumor_i += 1
    return colors


def build_categorical_colors(categories: list, palette: str = "Set2"):
    """Arbitrary label set: normal-like -> gray, invalid/NA -> dark gray, rest from
    `palette` (a qualitative cmap). Used for non-clone strips (e.g. cell_type)."""
    base = [mcolors.to_hex(c) for c in plt.get_cmap(palette).colors]
    colors = []
    i = 0
    for c in categories:
        cs = str(c)
        if cs in INVALID_LABELS or cs in NA_CELLTYPE:
            colors.append(NA_COLOR)
        elif _is_normal_like(cs):
            colors.append(NORMAL_COLOR)
        else:
            colors.append(base[i % len(base)])
            i += 1
    return colors


def make_baf_cmap():
    """Discrete diverging BAF colormap + [0,1] BoundaryNorm (10 bins, NaN -> white)."""
    cmap = mcolors.ListedColormap(BAF_COLORS, name="baf_disc")
    cmap.set_bad("white")
    norm = mcolors.BoundaryNorm(np.linspace(0, 1, 11), cmap.N, clip=True)
    return cmap, norm


def label_colors_for(categories: list, is_primary: bool, palette_index: int = 0):
    """Colors for one label column. The primary (clone) label uses the tab10 clone
    scheme; any other label set uses its own qualitative palette (by palette_index).
    Shared by heatmap strips and visium so coloring stays consistent across plots."""
    if is_primary:
        return build_label_colors(categories, clone_indexed=True)
    palette = _CATEGORICAL_PALETTES[palette_index % len(_CATEGORICAL_PALETTES)]
    return build_categorical_colors(categories, palette=palette)


def build_label_color_maps(
    row_label_map: dict[str, np.ndarray], primary_label: str | None
):
    """Per-label {value: color} maps. The primary (clone) label uses the tab10 clone
    scheme; each other label set gets its own distinct qualitative palette. Normal-like
    values are gray in every scheme (consistent with visium)."""
    color_maps = {}
    other_i = 0
    for name, values in row_label_map.items():
        cats = sorted({str(v) for v in values})
        is_primary = name == primary_label
        cols = label_colors_for(cats, is_primary=is_primary, palette_index=other_i)
        if not is_primary:
            other_i += 1
        color_maps[name] = dict(zip(cats, cols))
    return color_maps


def build_wl_coords(cnprofile: pd.DataFrame, wl_segments: pd.DataFrame):
    """Map bins to the wl_segments coordinate system (same as plot_cnv_profile).

    Returns a dict with:
        positions, abs_starts, abs_ends — per-bin arrays (length G)
        x_edges, col_bin_ids — for pcolormesh grid (heatmap)
        ch_coords, seg_coords — chromosome / centromere boundary offsets
        chr_vlines, chr_end, xlab_chrs, xtick_chrs — axis decoration
    """
    cnprofile = cnprofile.reset_index(drop=True)
    chs = cnprofile["#CHR"].unique()
    wl_chs = wl_segments.groupby("#CHR", sort=False)
    bins_chs = cnprofile.groupby("#CHR", sort=False, observed=True)

    G = len(cnprofile)
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


def plot_loss(
    losses: list, out_loss_file: str, val_type: str = "log-likelihood", dpi: int = 100
):
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("iterations")
    ax.set_ylabel(val_type)
    fig.tight_layout()
    fig.savefig(out_loss_file, dpi=dpi)
    plt.close(fig)
