import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from copytyping.inference.count_data import get_cnp_mask


def plot_scatter_2d_per_cell(
    read_counts: np.ndarray,
    ballele_counts: np.ndarray,
    total_allele_counts: np.ndarray,
    cn_A: np.ndarray,
    cn_B: np.ndarray,
    cn_C: np.ndarray,
    clones: list[str],
    cnprofile: pd.DataFrame,
    anns: pd.DataFrame,
    sample: str,
    outfile: str,
    label_col: str,
    base_props: np.ndarray | None = None,
    markersize: int = 4,
    rasterized: bool = True,
    dpi: int = 100,
):
    """Per-cluster 2D scatter (BAF vs log2RDR) with marginal KDEs, one page per cluster.

    Args:
        read_counts: (G, N) read depth / feature counts.
        ballele_counts: (G, N) B-allele counts.
        total_allele_counts: (G, N) total-allele counts (A + B).
        cn_A/cn_B/cn_C: (G, K) per-clone copy numbers.
        clones: clone names, length K.
        cnprofile: per-row CNP table (CNP/LENGTH columns for titles).
        anns: per-cell annotations (one row per column N), holds ``label_col``.
        base_props: (G,) RDR baseline; defaults to the global read-count fraction.
    """
    mask = get_cnp_mask(cn_A, cn_B, cn_C)
    imbalanced, aneuploid = mask["IMBALANCED"], mask["ANEUPLOID"]
    cluster_indices = np.where(imbalanced | aneuploid)[0]
    labels = anns[label_col].values

    num_clones = cn_A.shape[1]
    library_size = read_counts.sum(axis=0).astype(float)
    if base_props is None:
        base_props = read_counts.sum(axis=1) / max(library_size.sum(), 1)

    with PdfPages(outfile) as pdf:
        for g in cluster_indices:
            row = cnprofile.iloc[g]
            cnp_str = row.get("CNP", "")
            length_mb = row["LENGTH"] / 1e6 if "LENGTH" in row.index else np.nan
            n_bbc = int(row["#BBC"]) if "#BBC" in row.index else 0

            cn_parts = []
            for k in range(1, num_clones):
                cn_parts.append(f"{clones[k]}={cn_A[g, k]}|{cn_B[g, k]}")
            cn_str = ", ".join(cn_parts)

            tag = []
            if imbalanced[g]:
                tag.append("IMB")
            if aneuploid[g]:
                tag.append("ANE")

            total_allele_g = total_allele_counts[g].astype(float)
            ballele_g = ballele_counts[g].astype(float)
            read_g = read_counts[g].astype(float)
            lam_g = base_props[g]

            valid = (total_allele_g > 0) & (library_size * lam_g > 0)
            if valid.sum() < 10:
                continue

            baf = ballele_g[valid] / total_allele_g[valid]
            rdr = read_g[valid] / (library_size[valid] * lam_g)
            log2rdr = np.log2(np.clip(rdr, 1e-6, None))
            hue = labels[valid]

            uniq = sorted(set(hue), key=lambda x: (x != "normal", x))
            palette = {lab: f"C{i}" for i, lab in enumerate(uniq)}
            palette["normal"] = "lightgray"

            df = pd.DataFrame({"BAF": baf, "log2RDR": log2rdr, label_col: hue})

            # Pre-compute expected (BAF, log2RDR) per clone (used for ylim + markers below).
            # BAF: B/(A+B). log2RDR: log2(C_{g,k} / sum_g(lambda_g * C_{g,k})).
            exp_points = defaultdict(list)
            for k in range(num_clones):
                C_k = cn_C[g, k]
                exp_baf_k = cn_B[g, k] / C_k if C_k > 0 else 0.5
                denom_k = float(np.sum(base_props * cn_C[:, k]))
                if denom_k > 0 and C_k > 0:
                    exp_log2rdr_k = np.log2(C_k / denom_k)
                else:
                    exp_log2rdr_k = 0.0
                key = (round(exp_baf_k, 4), round(exp_log2rdr_k, 4))
                exp_points[key].append(clones[k])

            xlim = (-0.05, 1.05)
            # Adaptive ylim: tighten to (-2, 2) if all obs + expected fit, else (-5, 5)
            obs_finite = log2rdr[np.isfinite(log2rdr)]
            exp_finite = np.array([y for _, y in exp_points.keys()], dtype=float)
            all_y = np.concatenate([obs_finite, exp_finite])
            if all_y.size > 0 and all_y.min() >= -2 and all_y.max() <= 2:
                ylim = (-2, 2)
            else:
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
