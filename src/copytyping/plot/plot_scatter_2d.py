import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def plot_scatter_2d_per_cell(
    sx_data,
    anns: pd.DataFrame,
    sample: str,
    outfile: str,
    label_col: str,
    base_props=None,
    markersize=4,
    rasterized=True,
    dpi=100,
):
    """Per-cluster 2D scatter (BAF vs log2RDR) with marginal KDEs, one page per cluster."""
    informative = sx_data.MASK["IMBALANCED"] | sx_data.MASK["ANEUPLOID"]
    cluster_indices = np.where(informative)[0]
    labels = anns[label_col].values

    if base_props is None:
        base_props = sx_data.X.sum(axis=1) / max(sx_data.T.sum(), 1)

    with PdfPages(outfile) as pdf:
        for g in cluster_indices:
            row = sx_data.cnv_blocks.iloc[g]
            cnp_str = row.get("CNP", "")
            length_mb = row["LENGTH"] / 1e6 if "LENGTH" in row.index else np.nan
            n_bbc = int(row["#BBC"]) if "#BBC" in row.index else 0

            cn_parts = []
            for k in range(1, sx_data.K):
                cn_parts.append(
                    f"{sx_data.clones[k]}={sx_data.A[g, k]}|{sx_data.B[g, k]}"
                )
            cn_str = ", ".join(cn_parts)

            tag = []
            if sx_data.MASK["IMBALANCED"][g]:
                tag.append("IMB")
            if sx_data.MASK["ANEUPLOID"][g]:
                tag.append("ANE")

            D_g = sx_data.D[g].astype(float)
            Y_g = sx_data.Y[g].astype(float)
            X_g = sx_data.X[g].astype(float)
            T = sx_data.T.astype(float)
            lam_g = base_props[g]

            valid = (D_g > 0) & (T * lam_g > 0)
            if valid.sum() < 10:
                continue

            baf = Y_g[valid] / D_g[valid]
            rdr = X_g[valid] / (T[valid] * lam_g)
            log2rdr = np.log2(np.clip(rdr, 1e-6, None))
            hue = labels[valid]

            uniq = sorted(set(hue), key=lambda x: (x != "normal", x))
            palette = {lab: f"C{i}" for i, lab in enumerate(uniq)}
            palette["normal"] = "lightgray"

            df = pd.DataFrame({"BAF": baf, "log2RDR": log2rdr, label_col: hue})

            xlim = (-0.05, 1.05)
            ylim = (-5, 5)

            g0 = sns.JointGrid(
                data=df,
                x="BAF",
                y="log2RDR",
                hue=label_col,
                palette=palette,
                xlim=xlim,
                ylim=ylim,
            )
            g0.refline(x=0.50, y=0.0)
            g0.plot_joint(
                sns.scatterplot,
                s=markersize,
                edgecolors="none",
            )
            g0.plot_marginals(
                sns.kdeplot,
                common_norm=False,
                linewidth=0.8,
                fill=False,
            )
            scatter = g0.ax_joint.collections[0]
            scatter.set_rasterized(rasterized)
            scatter.set_antialiased(False)

            # Expected (BAF, log2RDR) cross markers per clone
            # Use raw B/(A+B) for BAF (unclipped), log2(C_clone/C_normal) for RDR
            # Group clones that share the same expected position
            C_normal_g = max(sx_data.C[g, 0], 1)
            exp_points = defaultdict(list)
            for k in range(sx_data.K):
                C_k = sx_data.C[g, k]
                exp_baf_k = sx_data.B[g, k] / C_k if C_k > 0 else 0.5
                exp_log2rdr_k = np.log2(max(C_k, 1e-6) / C_normal_g)
                key = (round(exp_baf_k, 4), round(exp_log2rdr_k, 4))
                exp_points[key].append(sx_data.clones[k])

            for (exp_baf, exp_log2rdr), clone_names in exp_points.items():
                g0.ax_joint.plot(
                    exp_baf,
                    exp_log2rdr,
                    marker="x",
                    color="black",
                    markersize=10,
                    markeredgewidth=2.5,
                    zorder=10,
                )
                label_text = ", ".join(clone_names)
                if len(clone_names) > 1:
                    label_text = f"({label_text})"
                g0.ax_joint.annotate(
                    label_text,
                    (exp_baf, exp_log2rdr),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=7,
                    fontweight="bold",
                )

            # Move legend outside, enlarge markers
            leg = g0.ax_joint.get_legend()
            if leg is not None:
                leg.remove()
            handles, labels_leg = g0.ax_joint.get_legend_handles_labels()
            g0.ax_marg_y.legend(
                handles,
                labels_leg,
                loc="upper left",
                fontsize=8,
                framealpha=0.8,
                borderaxespad=0.5,
                markerscale=5,
            )

            g0.figure.suptitle(
                f"{sample} — cluster {g} ({length_mb:.1f}Mb, {n_bbc} BBCs) — "
                f"{'/'.join(tag)}\nCN: {cn_str}",
                fontsize=10,
                fontweight="bold",
                y=1.02,
            )
            g0.figure.tight_layout()
            pdf.savefig(g0.figure, dpi=dpi, bbox_inches="tight")
            plt.close(g0.figure)

    logging.info(f"saved 2d scatter to {outfile}")
