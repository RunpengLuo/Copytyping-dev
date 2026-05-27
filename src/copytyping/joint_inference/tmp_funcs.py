import logging
import os

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import sparse

from copytyping.plot.plot_scatter_1d import plot_scatter_1d_pseudobulk, _merge_exp_lines
from copytyping.plot.plot_common import build_wl_coords
from copytyping.plot.plot_copynumber import (
    get_cn_colors,
    plot_ascn_profile,
    plot_ascn_legend,
)
from copytyping.plot.plot_heatmap import plot_heatmap
from copytyping.utils import read_whitelist_segments


# ============================== inference helpers ==============================


def build_cnp_df(
    genome_coords: pd.DataFrame,
    cna_profile: np.ndarray,
    cna_mirrored: np.ndarray,
    cna_int_states: np.ndarray,
    clone_names: list[str] | None = None,
) -> pd.DataFrame:
    """Merge contiguous same-CNP same-chrom bins into an ASCN-style block table.

    Effective ``(A, B)`` per (seg, clone) = canonical state with mirror applied.
    Run-length encoding (HATCHet pattern): boundary = any change in ``#CHR`` or
    any clone's ``"A|B"`` string vs. the previous bin → ``cumsum`` → ``groupby``.

    Returns a DataFrame with ``#CHR, START, END, cn_<clone_name>...`` (one
    ``cn_*`` column per clone). ``clone_names`` defaults to
    ``["clone0", ...]``; pass e.g. ``["normal", "clone1", "clone2"]``.
    """
    G, M = cna_profile.shape
    if clone_names is None:
        clone_names = [f"clone{i}" for i in range(M)]
    assert len(clone_names) == M, (len(clone_names), M)

    major = cna_int_states[cna_profile, 0]  # (G, M)
    minor = cna_int_states[cna_profile, 1]
    A = np.where(cna_mirrored == 1, minor, major).astype(np.int32)
    B = np.where(cna_mirrored == 1, major, minor).astype(np.int32)

    # vectorized "A|B" string per (seg, clone)
    cn_cols = [f"cn_{name}" for name in clone_names]
    cn_data = {
        col: np.char.add(np.char.add(A[:, m].astype(str), "|"), B[:, m].astype(str))
        for m, col in enumerate(cn_cols)
    }

    df = pd.DataFrame(
        {
            "#CHR": genome_coords["#CHR"].to_numpy(),
            "START": genome_coords["START"].to_numpy(),
            "END": genome_coords["END"].to_numpy(),
            **cn_data,
        }
    )

    # boundary: chrom change or any cn_<clone> change vs previous bin
    boundary = df["#CHR"] != df["#CHR"].shift()
    for col in cn_cols:
        boundary |= df[col] != df[col].shift()
    seg_id = boundary.cumsum() - 1

    agg = {
        "#CHR": ("#CHR", "first"),
        "START": ("START", "min"),
        "END": ("END", "max"),
        **{col: (col, "first") for col in cn_cols},
    }
    return df.groupby(seg_id, sort=False).agg(**agg).reset_index(drop=True)


def get_masks_from_cna_profile(
    cna_int_states: np.ndarray,
    cna_profile: np.ndarray,
    cna_mirrored: np.ndarray,
) -> dict[str, np.ndarray]:
    """Partition bins by CN-state pattern. Reconstructs effective ``(A, B)``
    from canonical state + mirror, then returns masks:
      * ``IMBALANCED``        — any clone has A ≠ B.
      * ``ANEUPLOID``         — any clone has total ≠ 2.
      * ``CLONAL_IMBALANCED`` — imbalanced and all tumor clones agree.
      * ``SUBCLONAL``         — tumor clones disagree (non-diploid).
    """
    major = cna_int_states[cna_profile, 0]  # (G, M)
    minor = cna_int_states[cna_profile, 1]  # (G, M)
    a_cn = np.where(cna_mirrored == 1, minor, major)  # A copy number
    b_cn = np.where(cna_mirrored == 1, major, minor)  # B copy number

    imbalanced = np.any(a_cn != b_cn, axis=1)
    aneuploid = np.any((a_cn + b_cn) != 2, axis=1)

    # diploid -> clonal -> subclonal (disjoint, partition all bins)
    diploid = np.all((a_cn == 1) & (b_cn == 1), axis=1)
    tumor_same = np.all(a_cn[:, 1:] == a_cn[:, 1:2], axis=1) & np.all(
        b_cn[:, 1:] == b_cn[:, 1:2], axis=1
    )
    clonal = tumor_same & ~diploid
    subclonal = ~clonal & ~diploid

    return {
        "IMBALANCED": imbalanced,
        "ANEUPLOID": aneuploid,
        "CLONAL_IMBALANCED": imbalanced & clonal,
        "SUBCLONAL": subclonal,
    }


# ================================== plotting ==================================


def plot_pseudobulk_baf(
    genome_coords_seg: pd.DataFrame,
    B_agg: sparse.csr_matrix,
    C_agg: sparse.csr_matrix,
    barcodes_df: pd.DataFrame,
    region_bed: str,
    sample: str,
    out_dir: str,
    out_prefix: str,
    markersize: int = 10,
    dpi: int = 150,
) -> None:
    """Per-rep pseudobulk BAF scatter (one PDF page per REP_ID)."""
    wl_segments = read_whitelist_segments(region_bed)
    wl = build_wl_coords(genome_coords_seg, wl_segments)
    positions = wl["positions"]
    chr_vlines = wl["chr_vlines"]
    chr_end = wl["chr_end"]
    xlab_chrs = wl["xlab_chrs"]
    xtick_chrs = wl["xtick_chrs"]

    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    pdf_path = os.path.join(plot_dir, f"{out_prefix}.pseudobulk_baf.pdf")

    rep_ids = barcodes_df["REP_ID"].unique()
    with PdfPages(pdf_path) as pdf:
        for rep_id in rep_ids:
            rep_mask = (barcodes_df["REP_ID"] == rep_id).to_numpy()
            n_cells = int(rep_mask.sum())

            B_rep = np.asarray(B_agg[:, rep_mask].sum(axis=1)).ravel().astype(float)
            C_rep = np.asarray(C_agg[:, rep_mask].sum(axis=1)).ravel().astype(float)
            obs_baf = np.where(C_rep > 0, B_rep / C_rep, np.nan)

            fig, ax = plt.subplots(1, 1, figsize=(20, 3))
            plot_scatter_1d_pseudobulk(
                ax,
                positions,
                obs_baf,
                chr_vlines,
                chr_end,
                xtick_chrs,
                xlab_chrs,
                ylabel="BAF",
                ylim=(-0.05, 1.05),
                markersize=markersize,
                title=f"{sample} rep={rep_id} (n={n_cells}) BAF",
            )
            fig.tight_layout()
            pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    logging.info(f"saved pseudobulk BAF ({len(rep_ids)} pages) to {pdf_path}")


def plot_clone_rdr_baf(
    genome_coords_seg: pd.DataFrame,
    X_seg: sparse.csr_matrix,
    B_seg: sparse.csr_matrix,
    C_seg: sparse.csr_matrix,
    T_seg: np.ndarray,
    labels: np.ndarray,
    base_props: np.ndarray,
    cnp_df: pd.DataFrame,
    barcodes_df: pd.DataFrame | None,
    region_bed: str,
    sample: str,
    out_dir: str,
    out_prefix: str,
    out_name: str = "clone_rdr_baf",
    clone_names: list[str] | None = None,
    exp_valid: np.ndarray | None = None,
    markersize: int = 6,
    dpi: int = 150,
) -> None:
    """Per-clone, per-rep pseudobulk log2RDR over BAF, with expected step-lines.

    One PDF page per REP_ID; each clone gets an RDR panel above a BAF panel; a
    shared bottom row shows the merged ASCN profile + legend. Dots are colored
    by per-seg ``(A, B)``. Observed and expected share the normalization, so a
    normal clone sits at log2RDR 0:

        observed RDR_g = X_clone_g / (T_clone * base_props_g)
        expected RDR_g = (A+B)/2 / S_m,  S_m = sum_g base_props_g * (A+B)/2
        observed BAF_g = B_clone_g / C_clone_g
        expected BAF_g = B / (A+B)         (A, B already effective in cnp_df)

    ``cnp_df`` (from ``build_cnp_df``) is the merged ASCN block table — one row
    per contiguous same-CNP block, with one ``cn_<clone>`` column per clone.
    Per-seg ``(A, B)`` is recovered via per-chrom ``searchsorted`` from
    ``genome_coords_seg``. ``exp_valid`` (G, M) bool mask blanks expected lines
    where the expectation is undefined (e.g. tumor meta-group on subclonal segs).
    """
    wl_segments = read_whitelist_segments(region_bed)
    wl = build_wl_coords(genome_coords_seg, wl_segments)
    G = X_seg.shape[0]
    cn_cols = [c for c in cnp_df.columns if c.startswith("cn_")]
    n_clones = len(cn_cols)
    if clone_names is None:
        clone_names = [c[3:] for c in cn_cols]  # strip "cn_"
    else:
        cn_cols = [f"cn_{name}" for name in clone_names]
    assert len(clone_names) == n_clones, (len(clone_names), n_clones)

    # cnp_df -> integer (A, B) per (cnp row, clone), vectorized via string split
    n_cnp = len(cnp_df)
    A_cnp = np.zeros((n_cnp, n_clones), dtype=np.int32)
    B_cnp = np.zeros((n_cnp, n_clones), dtype=np.int32)
    for m, col in enumerate(cn_cols):
        parts = cnp_df[col].str.split("|", expand=True).to_numpy()
        A_cnp[:, m] = parts[:, 0].astype(np.int32)
        B_cnp[:, m] = parts[:, 1].astype(np.int32)

    # map each seg in genome_coords_seg to its containing cnp_df row (per-chrom
    # searchsorted on START — cnp_df rows are sorted, contiguous within chrom)
    chrs_seg = genome_coords_seg["#CHR"].to_numpy()
    starts_seg = genome_coords_seg["START"].to_numpy()
    chrs_cnp = cnp_df["#CHR"].to_numpy()
    starts_cnp = cnp_df["START"].to_numpy()
    seg_to_cnp = np.full(G, -1, dtype=np.int64)
    for chrom in pd.unique(chrs_seg):
        seg_mask = chrs_seg == chrom
        cnp_idx_chr = np.where(chrs_cnp == chrom)[0]
        if cnp_idx_chr.size == 0:
            continue
        idx = (
            np.searchsorted(starts_cnp[cnp_idx_chr], starts_seg[seg_mask], side="right")
            - 1
        ).clip(0, cnp_idx_chr.size - 1)
        seg_to_cnp[seg_mask] = cnp_idx_chr[idx]
    assert (seg_to_cnp >= 0).all(), "some segs not covered by cnp_df"
    A_gm = A_cnp[seg_to_cnp]  # (G, M)
    B_gm = B_cnp[seg_to_cnp]

    # expected RDR/BAF from per-seg (A, B); A, B in cnp_df are already effective
    total_gm = (A_gm + B_gm).astype(np.float64)
    rdr_gm = total_gm / 2.0
    clone_norm = (base_props[:, None] * rdr_gm).sum(axis=0)  # (M,) genome-wide S_m
    exp_log2rdr = np.log2(
        np.maximum(rdr_gm / np.clip(clone_norm[None, :], 1e-12, None), 1e-6)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        baf_gm = np.where(total_gm > 0, B_gm / total_gm, 0.5)
    if exp_valid is not None:
        exp_log2rdr = np.where(exp_valid, exp_log2rdr, np.nan)
        baf_gm = np.where(exp_valid, baf_gm, np.nan)

    # per-clone scatter colors by per-seg (A, B) — state_style is symmetric in
    # (a, b) order, so canonical vs effective orientation does not affect color
    state_style, _ = get_cn_colors()
    bin_colors = [
        [
            state_style.get((int(A_gm[g, m]), int(B_gm[g, m])), state_style["default"])
            for g in range(G)
        ]
        for m in range(n_clones)
    ]

    # bottom-panel blocks: cnp_df already merged. CNP = ";a|b;a|b;..."
    cnv_blocks = cnp_df[["#CHR", "START", "END"]].copy()
    cnv_blocks["CNP"] = ";" + cnp_df[cn_cols].agg(";".join, axis=1)

    def pseudobulk(mat, cell_mask):
        return np.asarray(mat[:, cell_mask].sum(axis=1)).ravel().astype(float)

    def exp_lines(values):
        return _merge_exp_lines(wl["abs_starts"], wl["abs_ends"], values, chrs_seg)

    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    pdf_path = os.path.join(plot_dir, f"{out_prefix}.{out_name}.pdf")

    if barcodes_df is None:
        rep_ids = [None]
    else:
        rep_ids = list(barcodes_df["REP_ID"].unique())
    with PdfPages(pdf_path) as pdf:
        for rep_id in rep_ids:
            if rep_id is None:
                rep_mask = np.ones(labels.size, dtype=bool)
            else:
                rep_mask = (barcodes_df["REP_ID"] == rep_id).to_numpy()

            # layout: n_clones (RDR over BAF) rows + a shared bottom (CN profile + legend)
            fig = plt.figure(figsize=(20, 3 * n_clones + 2))
            outer = GridSpec(
                n_clones + 1,
                1,
                figure=fig,
                height_ratios=[2] * n_clones + [0.9],
                hspace=0.35,
                top=0.97,
            )
            row_axes = []
            for ci in range(n_clones):
                inner = GridSpecFromSubplotSpec(
                    2, 1, subplot_spec=outer[ci], height_ratios=[1, 1], hspace=0.25
                )
                row_axes.append((fig.add_subplot(inner[0]), fig.add_subplot(inner[1])))
            inner_bot = GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[n_clones], height_ratios=[0.6, 0.3], hspace=0.2
            )
            ax_cnp = fig.add_subplot(inner_bot[0])
            ax_leg = fig.add_subplot(inner_bot[1])

            for m in range(n_clones):
                clone_mask = rep_mask & (labels == m)
                n_cells = int(clone_mask.sum())
                ax_rdr, ax_baf = row_axes[m]

                # observed log2RDR vs expected (dots colored by CN state)
                X_clone = pseudobulk(X_seg, clone_mask)
                T_clone = float(T_seg[clone_mask].sum())
                obs_rdr = np.full(G, np.nan)
                valid = (base_props > 0) & (X_clone > 0) & (T_clone > 0)
                obs_rdr[valid] = np.log2(X_clone[valid] / (T_clone * base_props[valid]))
                plot_scatter_1d_pseudobulk(
                    ax_rdr,
                    wl["positions"],
                    obs_rdr,
                    wl["chr_vlines"],
                    wl["chr_end"],
                    wl["xtick_chrs"],
                    wl["xlab_chrs"],
                    exp_lines=exp_lines(exp_log2rdr[:, m]),
                    colors=bin_colors[m],
                    ylabel="log2RDR",
                    ylim=(-2, 2),
                    markersize=markersize,
                    title=f"{clone_names[m]} (n={n_cells}) log2RDR",
                    show_xticklabels=False,
                )

                # observed BAF vs expected
                B_clone = pseudobulk(B_seg, clone_mask)
                C_clone = pseudobulk(C_seg, clone_mask)
                obs_baf = np.divide(
                    B_clone, C_clone, out=np.full(G, np.nan), where=C_clone > 0
                )
                plot_scatter_1d_pseudobulk(
                    ax_baf,
                    wl["positions"],
                    obs_baf,
                    wl["chr_vlines"],
                    wl["chr_end"],
                    wl["xtick_chrs"],
                    wl["xlab_chrs"],
                    exp_lines=exp_lines(baf_gm[:, m]),
                    colors=bin_colors[m],
                    ylabel="BAF",
                    ylim=(-0.05, 1.05),
                    markersize=markersize,
                    title=f"{clone_names[m]} BAF",
                    show_xticklabels=False,
                )

            # shared bottom row: allele-specific CN profile + legend
            plot_ascn_profile(ax_cnp, cnv_blocks, wl_segments, plot_chrname=True)
            ax_cnp.set_xlim(0, wl["chr_end"])
            plot_ascn_legend(ax_leg)

            page_title = sample if rep_id is None else f"{sample} rep={rep_id}"
            fig.suptitle(page_title, fontsize=14, fontweight="bold", y=0.99)
            pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    logging.info(f"saved clone RDR/BAF ({len(rep_ids)} pages) to {pdf_path}")


def plot_cnp_segments_nll_probs(
    nll: np.ndarray,
    sampling_probs: np.ndarray,
    clone_labels: np.ndarray,
    cnp_segments_df: pd.DataFrame,
    clone_ids: list[str],
    region_bed: str,
    out_path: str,
    sample: str = "",
    figsize: tuple[float, float] = (20, 13),
    dpi: int = 150,
) -> None:
    """Cell × candidate-seg heatmap PDF for both NLL and sampling probability.

    Two pages: one heatmap per value type. Cells (rows) are sorted by clone
    label so each clone forms a contiguous block; candidate segments (cols)
    are laid out genome-wide via ``plot_heatmap``. Below each heatmap, the
    merged ASCN profile (cn_<clone> from ``cnp_segments_df``) is rendered for
    context. Both colormaps use ``LogNorm`` (NLL spans orders of magnitude;
    sampling probs span ~1/N to 1).

    ``nll`` and ``sampling_probs`` are both ``(N, n_cand)`` matching
    ``cnp_segments_df`` on the seg axis. ``cnp_segments_df`` must have
    ``#CHR / START / END`` plus one ``cn_<clone>`` column per clone (which it
    does by construction from ``derive_cnp_segments``).
    """
    order = np.argsort(clone_labels, kind="stable")
    nll_ord = nll[order]
    probs_ord = sampling_probs[order]
    labels_named = np.array([clone_ids[c] for c in clone_labels[order]])

    wl_fragments = read_whitelist_segments(region_bed)
    n_cells = nll.shape[0]

    # build cnv_blocks with CNP = ";cn_<clone1>;cn_<clone2>;..." for plot_ascn_profile
    cn_cols = [c for c in cnp_segments_df.columns if c.startswith("cn_")]
    cnv_blocks = cnp_segments_df[["#CHR", "START", "END"]].copy()
    cnv_blocks["CNP"] = ";" + cnp_segments_df[cn_cols].agg(";".join, axis=1)

    with PdfPages(out_path) as pdf:
        for matrix, value_name, vmin in [
            (nll_ord, "NLL", 1.0),
            (probs_ord, "sampling prob", 1.0 / n_cells),
        ]:
            vmax = max(float(matrix.max()), vmin * 10.0)
            cmap = plt.get_cmap("Reds").copy()
            cmap.set_bad("white")
            norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

            fig, axes = plt.subplots(
                nrows=3,
                ncols=1,
                figsize=figsize,
                gridspec_kw={"height_ratios": [10, 2, 2]},
            )
            fig.subplots_adjust(top=0.97, right=0.93, hspace=0.05)

            plot_heatmap(
                axes[0],
                labels_named,
                cnp_segments_df,
                matrix,
                wl_fragments,
                height=10,
                cmap=cmap,
                norm=norm,
            )
            plot_ascn_profile(axes[1], cnv_blocks, wl_fragments, plot_chrname=False)
            plot_ascn_legend(axes[2])

            title = f"{sample} per-cell × cand-seg {value_name}".strip()
            fig.suptitle(title, y=0.995, fontsize=14, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 0.93, 0.985])

            fig.canvas.draw()
            bbox = axes[0].get_position()
            cax = fig.add_axes([bbox.x1 + 0.015, bbox.y0, 0.012, bbox.height / 4])
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            fig.colorbar(sm, cax=cax, label=value_name)

            pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    logging.info(f"saved cnp_segments NLL/probs heatmap PDF to {out_path}")
