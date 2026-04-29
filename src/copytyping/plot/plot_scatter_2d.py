import logging

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
    markersize=3,
    rasterized=True,
    dpi=100,
):
    """Per-cluster 2D scatter (BAF vs RDR) with marginal KDEs, one page per cluster."""
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
            hue = labels[valid]

            uniq = sorted(set(hue), key=lambda x: (x != "normal", x))
            palette = {lab: f"C{i}" for i, lab in enumerate(uniq)}
            palette["normal"] = "silver"

            df = pd.DataFrame({"BAF": baf, "RDR": rdr, label_col: hue})

            rdr_hi = min(np.percentile(rdr, 99.5), 5.0)
            xlim = (-0.05, 1.05)
            ylim = (0, rdr_hi)

            g0 = sns.JointGrid(
                data=df,
                x="BAF",
                y="RDR",
                hue=label_col,
                palette=palette,
                xlim=xlim,
                ylim=ylim,
            )
            g0.refline(x=0.50)
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
