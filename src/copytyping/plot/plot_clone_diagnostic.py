"""
Per-segment BAF vs RDR scatter with clone-expected contours.
One page per informative segment in a single PDF.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import beta as beta_dist, norm as norm_dist
import seaborn as sns

from copytyping.sx_data.sx_data import SX_Data
from copytyping.utils import is_tumor_label, is_normal_label, NA_CELLTYPE


def plot_clone_diagnostic(
    sample: str,
    sx_data: SX_Data,
    anns: pd.DataFrame,
    model_params: dict,
    data_type: str,
    out_file: str,
    ref_label: str = None,
    dpi: int = 150,
):
    """Generate per-segment BAF vs RDR diagnostic PDF using learned model params."""
    theta_key = f"{data_type}-theta"
    # non-spatial assays have no theta — use 1.0 (pure cells)
    if theta_key in model_params:
        theta = model_params[theta_key]
    else:
        theta = np.ones(sx_data.N, dtype=np.float32)
    lambda_key = f"{data_type}-lambda"
    if lambda_key not in model_params:
        logging.info("skip clone diagnostic: lambda not in model_params")
        return
    lambda_g = model_params[lambda_key]
    tau_arr = model_params[f"{data_type}-tau"]
    invphi_arr = model_params[f"{data_type}-inv_phi"]

    # RDR relative to normal diploid (C[g,k] / C_normal[g])
    # This matches the observed RDR = X / (T * lambda_g) where lambda_g is from normal spots
    C_normal = sx_data.C[:, 0]  # normal clone CN (typically 2)
    rdr_vs_normal = sx_data.C / np.maximum(C_normal[:, None], 1)
    clones = sx_data.clones
    tumor_clones = clones[1:]
    K_tumor = len(tumor_clones)
    clone_colors = sns.color_palette("tab10", n_colors=max(K_tumor + 1, 10))

    # pathology labels
    if ref_label and ref_label in anns.columns:
        is_tumor = anns[ref_label].apply(is_tumor_label).to_numpy()
        is_normal = anns[ref_label].apply(is_normal_label).to_numpy()
    else:
        is_tumor = np.ones(sx_data.N, dtype=bool)
        is_normal = np.zeros(sx_data.N, dtype=bool)

    sub_mask = sx_data.MASK["SUBCLONAL"]
    imb_mask = sx_data.MASK["IMBALANCED"]
    ane_mask = sx_data.MASK["ANEUPLOID"]
    informative = imb_mask | ane_mask
    seg_indices = np.where(informative)[0]

    imb_positions = np.cumsum(imb_mask) - 1
    ane_positions = np.cumsum(ane_mask) - 1

    med_theta = np.median(theta[theta > 0.3]) if (theta > 0.3).sum() > 0 else 0.5

    with PdfPages(out_file) as pdf:
        for global_g in seg_indices:
            row = sx_data.cnv_blocks.iloc[global_g]
            seg_label = (
                f"{row['#CHR']}:{int(row['START']) // 1e6:.0f}"
                f"-{int(row['END']) // 1e6:.0f}Mb"
            )
            cnp_str = row["CNP"]
            is_sub = sub_mask[global_g]
            is_imb = imb_mask[global_g]
            is_ane = ane_mask[global_g]

            tau_g = tau_arr[imb_positions[global_g]] if is_imb else tau_arr[0]
            invphi_g = invphi_arr[ane_positions[global_g]] if is_ane else invphi_arr[0]

            # per-spot BAF and RDR
            Y_g = sx_data.Y[global_g].astype(np.float64)
            D_g = sx_data.D[global_g].astype(np.float64)
            X_g = sx_data.X[global_g].astype(np.float64)
            T = sx_data.T.astype(np.float64)
            lam_g = lambda_g[global_g]
            expected_normal = T * lam_g

            baf = np.divide(Y_g, D_g, out=np.full_like(Y_g, np.nan), where=D_g > 0)
            rdr = np.divide(
                X_g,
                expected_normal,
                out=np.full_like(X_g, np.nan),
                where=expected_normal > 0,
            )

            valid = np.isfinite(baf) & np.isfinite(rdr)
            if valid.sum() < 10:
                continue

            baf_v = baf[valid]
            rdr_v = rdr[valid]
            is_tumor_v = is_tumor[valid]
            is_normal_v = is_normal[valid]

            # figure with marginals
            fig = plt.figure(figsize=(8, 8))
            gs = fig.add_gridspec(
                3,
                3,
                width_ratios=[4, 1, 0.3],
                height_ratios=[1, 4, 0.3],
                hspace=0.05,
                wspace=0.05,
            )
            ax_main = fig.add_subplot(gs[1, 0])
            ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
            ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

            # scatter
            if is_normal_v.any():
                ax_main.scatter(
                    baf_v[is_normal_v],
                    rdr_v[is_normal_v],
                    c="lightgray",
                    s=3,
                    alpha=0.3,
                    label="normal",
                    rasterized=True,
                )
            if is_tumor_v.any():
                ax_main.scatter(
                    baf_v[is_tumor_v],
                    rdr_v[is_tumor_v],
                    c="salmon",
                    s=3,
                    alpha=0.3,
                    label="tumor",
                    rasterized=True,
                )

            # per-clone contours
            for ki, clone in enumerate(tumor_clones):
                p_clone = sx_data.BAF[global_g, ki + 1]
                rdr_clone = rdr_vs_normal[global_g, ki + 1]
                denom = rdr_clone * med_theta + (1.0 - med_theta)
                p_hat = (
                    p_clone * rdr_clone * med_theta + 0.5 * (1.0 - med_theta)
                ) / max(denom, 1e-12)
                p_hat = np.clip(p_hat, 1e-6, 1 - 1e-6)
                expected_rdr = med_theta * rdr_clone + (1 - med_theta)

                ax_main.plot([], [], color=clone_colors[ki], lw=1.5, label=clone)

                baf_grid = np.linspace(-0.1, 1.1, 120)
                a_bb, b_bb = tau_g * p_hat, tau_g * (1 - p_hat)
                baf_pdf = beta_dist.pdf(np.clip(baf_grid, 1e-6, 1 - 1e-6), a_bb, b_bb)

                mu_spot = np.median(expected_normal[expected_normal > 0]) * expected_rdr
                rdr_var = (mu_spot + mu_spot**2 / invphi_g) / mu_spot**2
                rdr_std = np.sqrt(rdr_var)
                rdr_grid = np.linspace(
                    max(0, expected_rdr - 4 * rdr_std),
                    expected_rdr + 4 * rdr_std,
                    100,
                )
                rdr_pdf = norm_dist.pdf(rdr_grid, expected_rdr, rdr_std)

                joint = np.outer(rdr_pdf, baf_pdf)
                joint = joint / joint.max()
                ax_main.contour(
                    baf_grid,
                    rdr_grid,
                    joint,
                    levels=[0.05, 0.3, 0.7],
                    colors=[clone_colors[ki]],
                    linewidths=0.8,
                    alpha=0.6,
                )

                ax_top.plot(
                    baf_grid, baf_pdf, color=clone_colors[ki], lw=1.5, alpha=0.7
                )
                ax_right.plot(
                    rdr_pdf, rdr_grid, color=clone_colors[ki], lw=1.5, alpha=0.7
                )

            # normal expected contour (BAF=0.5, RDR=1.0)
            ax_main.plot([], [], color="gray", ls=":", lw=1, label="normal")
            baf_grid_n = np.linspace(-0.1, 1.1, 120)
            baf_pdf_n = beta_dist.pdf(
                np.clip(baf_grid_n, 1e-6, 1 - 1e-6), tau_g * 0.5, tau_g * 0.5
            )
            mu_normal = np.median(expected_normal[expected_normal > 0])
            if mu_normal > 0 and invphi_g > 0:
                rdr_var_n = (mu_normal + mu_normal**2 / invphi_g) / mu_normal**2
                rdr_std_n = np.sqrt(rdr_var_n)
            else:
                rdr_std_n = 0.1
            rdr_grid_n = np.linspace(
                max(0, 1.0 - 4 * rdr_std_n), 1.0 + 4 * rdr_std_n, 100
            )
            rdr_pdf_n = norm_dist.pdf(rdr_grid_n, 1.0, rdr_std_n)
            joint_n = np.outer(rdr_pdf_n, baf_pdf_n)
            joint_n = joint_n / joint_n.max()
            ax_main.contour(
                baf_grid_n,
                rdr_grid_n,
                joint_n,
                levels=[0.05, 0.3, 0.7],
                colors=["gray"],
                linewidths=0.8,
                alpha=0.4,
                linestyles=":",
            )
            ax_top.plot(baf_grid_n, baf_pdf_n, color="gray", lw=1, ls=":", alpha=0.5)
            ax_right.plot(rdr_pdf_n, rdr_grid_n, color="gray", lw=1, ls=":", alpha=0.5)

            # marginal histograms
            bins_baf = np.linspace(-0.1, 1.1, 55)
            rdr_max = min(np.percentile(rdr_v, 99.5), 5)
            bins_rdr = np.linspace(0, rdr_max, 50)

            if is_normal_v.any():
                ax_top.hist(
                    baf_v[is_normal_v],
                    bins=bins_baf,
                    density=True,
                    alpha=0.3,
                    color="gray",
                    edgecolor="none",
                )
                ax_right.hist(
                    rdr_v[is_normal_v],
                    bins=bins_rdr,
                    density=True,
                    alpha=0.3,
                    color="gray",
                    edgecolor="none",
                    orientation="horizontal",
                )
            if is_tumor_v.any():
                ax_top.hist(
                    baf_v[is_tumor_v],
                    bins=bins_baf,
                    density=True,
                    alpha=0.4,
                    color="salmon",
                    edgecolor="none",
                )
                ax_right.hist(
                    rdr_v[is_tumor_v],
                    bins=bins_rdr,
                    density=True,
                    alpha=0.4,
                    color="salmon",
                    edgecolor="none",
                    orientation="horizontal",
                )

            ax_main.set_xlabel("BAF (Y/D)", fontsize=9)
            ax_main.set_ylabel("RDR (X / T·λ)", fontsize=9)
            ax_main.set_xlim(-0.1, 1.1)
            ax_main.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax_main.set_ylim(0, rdr_max)
            ax_main.legend(fontsize=6, loc="upper left", markerscale=2)

            ax_top.tick_params(labelbottom=False)
            ax_right.tick_params(labelleft=False)
            ax_top.set_ylabel("density", fontsize=7)
            ax_right.set_xlabel("density", fontsize=7)

            sub_tag = " [SUBCLONAL]" if is_sub else " [clonal]"
            title_color = "red" if is_sub else "black"
            fig.suptitle(
                f"{sample} | {seg_label}{sub_tag}\n"
                f"CNP: {cnp_str}\n"
                f"tau={tau_g:.1f}, inv_phi={invphi_g:.1f}, "
                f"lambda={lam_g:.4f}, median_theta={med_theta:.2f}",
                fontsize=10,
                fontweight="bold",
                color=title_color,
            )

            plt.tight_layout(rect=[0, 0, 1, 0.92])
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

    logging.info(f"saved clone diagnostic to {out_file} ({len(seg_indices)} pages)")
