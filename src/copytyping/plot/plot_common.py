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
_TUMOR_COLORS = [mcolors.to_hex(c) for c in plt.get_cmap("tab10").colors]


def _is_normal_like(label: str) -> bool:
    """True for the diploid/normal reference label (e.g. 'normal', 'Normal_cell')."""
    return str(label).lower().startswith("normal")


def _label_color_index(label: str) -> int:
    """Stable color index for a label: clone1->0, clone2->1, ...; others by hash."""
    m = re.match(r"clone(\d+)", str(label))
    if m:
        return int(m.group(1)) - 1
    return hash(str(label)) % len(_TUMOR_COLORS)


def build_label_colors(categories: list, clone_indexed: bool = True) -> list[str]:
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


def build_categorical_colors(categories: list, palette: str = "Set2") -> list[str]:
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


def build_wl_coords(cnprofile: pd.DataFrame, wl_segments: pd.DataFrame) -> dict:
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
