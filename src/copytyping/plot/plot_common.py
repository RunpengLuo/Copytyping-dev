import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

from copytyping.utils import INVALID_LABELS, NA_CELLTYPE, is_tumor_label


class FigureSaver:
    """Multi-figure writer that is a drop-in for ``PdfPages``.

    With ``img_type='pdf'`` it writes a single multi-page PDF at
    ``{out_base}.pdf``. With ``img_type in {'png','svg'}`` each page is written
    as its own file ``{out_base}.p{i}.{img_type}``. ``savefig`` mirrors the
    ``PdfPages.savefig`` signature (extra kwargs are ignored — dpi/transparent
    come from the saver) so callers need no changes. Use as a context manager.
    """

    def __init__(
        self,
        out_base: str,
        img_type: str = "pdf",
        dpi: int = 300,
        transparent: bool = False,
    ):
        self.out_base = out_base
        self.img_type = img_type
        self.dpi = dpi
        self.transparent = transparent
        self._page = 0
        self._pdf = PdfPages(f"{out_base}.pdf") if img_type == "pdf" else None

    def savefig(self, fig: plt.Figure, *args, **kwargs):
        if self._pdf is not None:
            self._pdf.savefig(
                fig, dpi=self.dpi, bbox_inches="tight", transparent=self.transparent
            )
        else:
            fig.savefig(
                f"{self.out_base}.p{self._page}.{self.img_type}",
                dpi=self.dpi,
                bbox_inches="tight",
                transparent=self.transparent,
            )
        self._page += 1

    def close(self):
        if self._pdf is not None:
            self._pdf.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ── Unified label color palette (shared by heatmap + visium) ──
# normal-like -> gray, invalid/NA -> dark gray, tumor clones -> qualitative palette
NORMAL_COLOR = "lightgray"
NA_COLOR = "darkgray"
BLACK = (0, 0, 0, 1)
_TUMOR_COLORS = [mcolors.to_hex(c) for c in plt.get_cmap("tab10").colors]
_TUMOR_COLORS_20 = [mcolors.to_hex(c) for c in plt.get_cmap("tab20").colors]

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


def make_baf_cmap():
    """Discrete diverging BAF colormap + [0,1] BoundaryNorm (10 bins, NaN -> white)."""
    cmap = mcolors.ListedColormap(BAF_COLORS, name="baf_disc")
    cmap.set_bad("white")
    norm = mcolors.BoundaryNorm(np.linspace(0, 1, 11), cmap.N, clip=True)
    return cmap, norm


def _clone_order_key(c: str):
    """Sort key for the clone label: normal first, then clone1, clone2, ...,
    then any other label, with invalid/NA last."""
    if c == "normal":
        return (0, 0, "")
    m = re.match(r"clone(\d+)$", c)
    if m:
        return (1, int(m.group(1)), "")
    if c in INVALID_LABELS:
        return (3, 0, c)
    return (2, 0, c)


def _is_colored_label(c: str):
    """A label that consumes a palette slot (not gray): not normal-like, not NA."""
    return c not in INVALID_LABELS and c not in NA_CELLTYPE and not _is_normal_like(c)


def build_label_color_maps(
    row_label_map: dict[str, np.ndarray], primary_label: str | None
):
    """Per-label {value: color} maps drawn from ONE shared palette.

    Colors are assigned to distinct label *values* (a global set union), so a value
    that appears in several label sets — e.g. ``normal`` in both copytyping and a
    path annotation, or shared clones — gets the SAME color everywhere. The primary
    (clone) label is visited first, ordered normal -> clone1 -> clone2 -> ..., so
    clones take the leading palette slots; remaining sets (sorted alphabetically)
    contribute any new values. Normal-like values are gray and invalid/NA are dark
    gray (these consume no palette slot). Palette is tab10, or tab20 when more than
    10 distinct colored values are needed."""
    # primary first (clone order), then the rest (alphabetical)
    names = ([primary_label] if primary_label in row_label_map else []) + [
        n for n in row_label_map if n != primary_label
    ]

    def cats_for(name: str):
        uniq = {str(v) for v in row_label_map[name]}
        return (
            sorted(uniq, key=_clone_order_key)
            if name == primary_label
            else sorted(uniq)
        )

    # global value -> color: first encounter (in visit order) fixes the color
    ordered_values = []
    seen = set()
    for name in names:
        for c in cats_for(name):
            if _is_colored_label(c) and c not in seen:
                seen.add(c)
                ordered_values.append(c)
    palette = (
        _TUMOR_COLORS if len(ordered_values) <= len(_TUMOR_COLORS) else _TUMOR_COLORS_20
    )
    value_color = {c: palette[i % len(palette)] for i, c in enumerate(ordered_values)}

    color_maps = {}
    for name in names:
        cmap = {}
        for c in cats_for(name):
            if c in INVALID_LABELS or c in NA_CELLTYPE:
                cmap[c] = NA_COLOR
            elif _is_normal_like(c):
                cmap[c] = NORMAL_COLOR
            else:
                cmap[c] = value_color[c]
        color_maps[name] = cmap
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


def plot_tumor_post_hist(
    anns: pd.DataFrame,
    tumor_post: str,
    ref_label: str | None = None,
    title: str | None = None,
    bins: int = 41,
    pdf_pages: "FigureSaver | PdfPages | None" = None,
    out_file: str | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> None:
    """Tumor-posterior histogram, plus reference-set contamination curves.

    Panel 1: histogram of ``tumor_post``; stacked by reference cell type when
    ``ref_label`` is available (else a single histogram). Diagnoses contamination
    of the normal/reference set used for the RDR baseline: tumor-type cells
    sitting at low posterior are tumor cells misassigned to normal.

    Panels 2-3 (only with ``ref_label``) condition on the MAP-normal pool
    (``tumor_post < 0.5``, the reference candidates) and report the tumor false
    negatives (true tumor cells kept in the reference set, FN count + FN/#tumor):
    - panel 2: vs an absolute ``tumor_post`` cutoff swept 0..1 step 0.1 (pool
      cells with ``tumor_post < cutoff`` kept);
    - panel 3: vs the top-x% of the pool by P(normal)=1-tumor_post.
    Same cells per set size, different knob -> guides a dataset-general rule.
    Saved to ``pdf_pages`` else ``out_file``.
    """
    has_ref = ref_label is not None and ref_label in anns.columns
    cols = [tumor_post] + ([ref_label] if has_ref else [])
    df = anns[cols].copy()
    df[tumor_post] = pd.to_numeric(df[tumor_post], errors="coerce")
    df = df[np.isfinite(df[tumor_post].to_numpy())]
    if has_ref:
        df = df[~df[ref_label].isin(NA_CELLTYPE)]
    if df.empty:
        return
    post = df[tumor_post].to_numpy()
    edges = np.linspace(0.0, 1.0, bins)

    n_panels = 3 if has_ref else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3.4 * n_panels), squeeze=False)
    ax_hist = axes[0, 0]
    if has_ref:
        # normal-like cell types first, tumor-like last (the diagnostic group)
        groups = sorted(
            df[ref_label].unique(), key=lambda g: (is_tumor_label(str(g)), str(g))
        )
        color_map = build_label_color_maps({ref_label: df[ref_label].to_numpy()}, None)[
            ref_label
        ]
        data = [df.loc[df[ref_label] == g, tumor_post].to_numpy() for g in groups]
        colors = [color_map[g] for g in groups]
        labels = [f"{g} (n={len(d)})" for g, d in zip(groups, data)]
        ax_hist.hist(data, bins=edges, stacked=True, color=colors, label=labels)
        ax_hist.legend(fontsize=7, ncol=2, title=f"ref {ref_label}", title_fontsize=8)
    else:
        ax_hist.hist(post, bins=edges, color="steelblue")
    ax_hist.set_ylabel("cell count")
    ax_hist.set_xlabel(f"tumor posterior ({tumor_post})")
    ax_hist.axvline(0.5, ls="--", color="k", lw=1.0, alpha=0.7)
    ax_hist.set_xlim(0.0, 1.0)

    if has_ref:
        is_tum = np.array([is_tumor_label(str(g)) for g in df[ref_label]])
        n_tum = int(is_tum.sum())
        map_normal = post < 0.5  # reference candidates (MAP = normal)

        def _draw_fn(ax, xs, fn, xlabel, xticks):
            rate = fn / n_tum if n_tum > 0 else np.zeros_like(fn, dtype=float)
            ax.plot(xs, fn, "-o", color="purple")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("tumor_cell FN count", color="purple")
            ax.tick_params(axis="y", labelcolor="purple")
            ax.set_xticks(xticks)
            ax_r = ax.twinx()
            ax_r.plot(xs, rate, "-s", color="darkorange")
            ax_r.set_ylabel("FN / #tumor_cell", color="darkorange")
            ax_r.tick_params(axis="y", labelcolor="darkorange")
            ax_r.set_ylim(0.0, max(0.02, float(rate.max()) * 1.1))

        # panel 2: absolute tumor_post cutoff within the MAP-normal pool
        cutoffs = np.round(np.arange(0.0, 1.0001, 0.1), 1)
        fn_cut = np.array(
            [int((is_tum & map_normal & (post < c)).sum()) for c in cutoffs]
        )
        _draw_fn(
            axes[1, 0],
            cutoffs,
            fn_cut,
            "tumor_post cutoff (MAP-normal cells < cutoff -> reference)",
            cutoffs,
        )
        axes[1, 0].axvline(0.5, ls="--", color="k", lw=1.0, alpha=0.7)
        axes[1, 0].set_title(
            f"#tumor_cell={n_tum}, MAP-normal pool={int(map_normal.sum())}", fontsize=9
        )

        # panel 3: top-x% of the MAP-normal pool by P(normal) = 1 - tumor_post
        pool_tum = is_tum[map_normal][np.argsort(post[map_normal])]
        n_pool = pool_tum.size
        fracs = np.round(np.arange(0.1, 1.0001, 0.1), 1)
        fn_top = np.array(
            [int(pool_tum[: max(1, int(round(p * n_pool)))].sum()) for p in fracs]
        )
        _draw_fn(
            axes[2, 0],
            fracs * 100,
            fn_top,
            "top-x% of MAP-normal pool by P(normal) -> reference",
            fracs * 100,
        )

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if pdf_pages is not None:
        pdf_pages.savefig(fig)
    elif out_file is not None:
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight", transparent=transparent)
    plt.close(fig)
