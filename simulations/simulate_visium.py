#!/usr/bin/env python3
"""
Simulate Visium spatial transcriptomics data with known CNAs for copytyping validation.

Generative model (following CalicoST S15 / S17):
  RDR:  X_{g,n} ~ NegBin(mu=T_n * lam_g * (theta_n * mu_{g,k} + (1-theta_n)), phi)
  BAF:  Y_{g,n} ~ BetaBin(D_{g,n}, tau * p_hat, tau * (1 - p_hat))
        where p_hat = (theta_n * mu_{g,k} * p_{g,k} + 0.5*(1-theta_n))
                     / (theta_n * mu_{g,k} + (1-theta_n))

Can either:
  (a) take an existing cnv_segments.tsv as input (reuse real CNP), or
  (b) generate random CNA events on a diploid background.

Outputs copytyping-compatible files:
  barcodes.tsv.gz, cnv_segments.tsv, X_count.npz, Y_count.npz, D_count.npz,
  VISIUM.h5ad, ground_truth.tsv
"""

import os
import argparse
import logging
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from anndata import AnnData
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CNA = namedtuple("CNA", ["chr", "start", "end", "A_copy", "B_copy"])

# candidate allele-specific copy numbers for random CNA generation
CANDIDATE_CN = [
    (1, 0),
    (0, 1),  # deletion
    (2, 0),
    (0, 2),  # CNLOH
    (2, 1),
    (1, 2),  # gain
    (2, 2),  # balanced amp
    (3, 1),
    (1, 3),  # unbalanced amp
]


# ──────────────────────────────────────────────
# Spatial layout
# ──────────────────────────────────────────────
def simulate_spatial_hexagon(
    n_spots, n_clones, max_tumor_prop=0.9, make_pure=False, random_state=0
):
    """
    Simulate hexagonal Visium-like spatial layout with clone regions
    assigned by KMeans Voronoi partitioning (1 iteration).

    Returns coords (N,2), clone_labels (N,), tumor_proportions (N,).
    """
    rng = np.random.default_rng(random_state)
    # build hexagonal grid to get ~n_spots, keep all spots for spatial coherence
    side = int(np.ceil(np.sqrt(n_spots * 2)))
    coords = np.array(
        [[i, j] for i in range(side) for j in range(side) if i % 2 == j % 2]
    )
    if len(coords) > n_spots:
        idx = rng.choice(len(coords), size=n_spots, replace=False)
        idx.sort()
        coords = coords[idx]
    n_spots = len(coords)
    # scale to pixel-like coordinates matching real Visium spacing (~200 px)
    # and offset to center in a ~20000 px field of view
    coords = coords * 200 + 2000

    labels = KMeans(
        n_clusters=n_clones + 1,
        init="random",
        n_init=1,
        random_state=random_state,
        max_iter=1,
    ).fit_predict(coords)
    clone_names = ["normal"] + [f"clone{i}" for i in range(1, n_clones + 1)]
    clone_map = {i: clone_names[i] for i in range(n_clones + 1)}
    clone_labels = np.array([clone_map[l] for l in labels])

    if make_pure:
        tumor_proportions = np.where(clone_labels == "normal", 0.0, 1.0)
    else:
        normal_center = coords[clone_labels == "normal"].mean(axis=0)
        dists = np.linalg.norm(coords[clone_labels != "normal"] - normal_center, axis=1)
        a = dists.min()
        b_per_clone = []
        for c in clone_names[1:]:
            mask_c = clone_labels[clone_labels != "normal"] == c
            if mask_c.any():
                b_per_clone.append(dists[mask_c].max())
        b = min(b_per_clone) if b_per_clone else a + 1
        tumor_proportions = np.zeros(n_spots)
        tumor_proportions[clone_labels != "normal"] = max_tumor_prop / (
            1 + np.exp(-0.005 * (dists - (a + b) / 2))
        )

    logging.info(f"Spatial layout: {n_spots} spots")
    for c in clone_names:
        logging.info(f"  {c}: {np.sum(clone_labels == c)} spots")
    return coords, clone_labels, tumor_proportions


# ──────────────────────────────────────────────
# CNV profile generation
# ──────────────────────────────────────────────
def generate_random_cnp(
    genome_size_file,
    n_clones,
    n_cnas_per_clone=2,
    cna_size=3e7,
    random_state=0,
):
    """Generate random CNA events on a diploid background."""
    rng = np.random.default_rng(random_state)
    chr_sizes = pd.read_csv(
        genome_size_file, sep="\t", header=None, names=["chr", "size"]
    )
    chr_sizes = chr_sizes[chr_sizes["chr"].str.match(r"^chr([0-9]+|X)$")]
    chr_sizes = chr_sizes.set_index("chr")

    all_cnas = {}
    for k in range(1, n_clones + 1):
        events = []
        for _ in range(n_cnas_per_clone):
            eligible = chr_sizes[chr_sizes["size"] > cna_size]
            chrom = rng.choice(eligible.index)
            start = rng.integers(0, int(eligible.loc[chrom, "size"] - cna_size))
            end = start + int(cna_size)
            cn = CANDIDATE_CN[rng.integers(len(CANDIDATE_CN))]
            events.append(CNA(chrom, start, end, cn[0], cn[1]))
        all_cnas[k] = events

    # build segment breakpoints per chromosome
    rows = []
    for chrom in chr_sizes.index:
        breaks = {0, int(chr_sizes.loc[chrom, "size"])}
        for k, events in all_cnas.items():
            for cna in events:
                if cna.chr == chrom:
                    breaks.add(cna.start)
                    breaks.add(cna.end)
        breaks = sorted(breaks)
        for i in range(len(breaks) - 1):
            seg_start, seg_end = breaks[i], breaks[i + 1]
            cn_per_clone = ["1|1"]  # normal is always diploid
            for k in range(1, n_clones + 1):
                a, b = 1, 1
                for cna in all_cnas[k]:
                    if (
                        cna.chr == chrom
                        and cna.start <= seg_start
                        and cna.end >= seg_end
                    ):
                        a, b = cna.A_copy, cna.B_copy
                cn_per_clone.append(f"{a}|{b}")
            rows.append(
                {
                    "#CHR": chrom,
                    "START": seg_start,
                    "END": seg_end,
                    "CNP": ";".join(cn_per_clone),
                }
            )

    df = pd.DataFrame(rows)
    df = df[df["END"] - df["START"] > 1000].reset_index(drop=True)

    n_total = n_clones + 1
    props = [1.0 / n_total] * n_total
    df["PROPS"] = ";".join(f"{p:.4f}" for p in props)
    df["SAMPLE"] = "simulated"
    df["seg_id"] = range(len(df))
    df["#SNPS"] = 100
    df["#gene"] = 10
    for i, name in enumerate(
        ["normal"] + [f"clone{j}" for j in range(1, n_clones + 1)]
    ):
        df[f"cn_{name}"] = df["CNP"].apply(lambda x, idx=i: x.split(";")[idx])
        df[f"u_{name}"] = props[i]

    logging.info(f"Generated {len(df)} segments, {n_cnas_per_clone} CNAs/clone")
    return df, all_cnas


def load_cnp(cnp_file):
    """Load an existing cnv_segments.tsv."""
    return pd.read_csv(cnp_file, sep="\t")


def parse_cn_arrays(cnv_df, laplace=0.01):
    """Parse CNP column into A, B, C, BAF arrays (G, K)."""
    num_clones = len(cnv_df["CNP"].iloc[0].split(";"))
    G = len(cnv_df)
    A = np.zeros((G, num_clones), dtype=np.int32)
    B = np.zeros((G, num_clones), dtype=np.int32)
    for i in range(num_clones):
        A[:, i] = (
            cnv_df["CNP"]
            .apply(lambda x, idx=i: int(x.split(";")[idx].split("|")[0]))
            .values
        )
        B[:, i] = (
            cnv_df["CNP"]
            .apply(lambda x, idx=i: int(x.split(";")[idx].split("|")[1]))
            .values
        )
    C = A + B
    BAF = np.divide(B, C, out=np.zeros_like(C, dtype=np.float64), where=C > 0)
    BAF = np.clip(BAF, laplace, 1 - laplace)
    return A, B, C, BAF


# ──────────────────────────────────────────────
# Count simulation
# ──────────────────────────────────────────────
def simulate_counts(
    A,
    B,
    C,
    BAF,
    clone_labels,
    tumor_proportions,
    tau=50.0,
    phi=30.0,
    rdr_lognormal_mean=8.0,
    rdr_lognormal_sigma=0.4,
    baf_lognormal_mean=6.0,
    baf_lognormal_sigma=0.4,
    random_state=0,
):
    """
    Simulate X (read depth), Y (B-allele), D (total allele) count matrices.

    Follows CalicoST S15 generative process and copytyping's Spot_Model likelihood.
    """
    rng = np.random.default_rng(random_state)
    G, n_clones_total = C.shape  # n_clones_total = 1 normal + K_tumor tumor clones
    N = len(clone_labels)
    clone_names = ["normal"] + [f"clone{i}" for i in range(1, n_clones_total)]

    # baseline proportions: uniform across bins
    lambda_g = np.ones(G, dtype=np.float64) / G

    # Clone-specific RDR: mu_{g,k} = C_{g,k} / sum_g(lam_g * C_{g,k})
    # Shape (n_clones_total,) denominator, broadcast to (G, n_clones_total).
    denom_k = (lambda_g[:, None] * C).sum(axis=0)
    mu_gk = C / denom_k

    # library sizes from PoissonLogNormal
    T_n = rng.poisson(
        lam=rng.lognormal(mean=rdr_lognormal_mean, sigma=rdr_lognormal_sigma, size=N)
    ).astype(np.float64)
    T_n = np.maximum(T_n, 1)

    D_total_n = rng.poisson(
        lam=rng.lognormal(mean=baf_lognormal_mean, sigma=baf_lognormal_sigma, size=N)
    ).astype(np.float64)
    D_total_n = np.maximum(D_total_n, 1)

    X = np.zeros((G, N), dtype=np.int32)
    Y = np.zeros((G, N), dtype=np.int32)
    D = np.zeros((G, N), dtype=np.int32)

    inv_phi = 1.0 / phi

    for n in range(N):
        k = clone_names.index(clone_labels[n])
        theta = tumor_proportions[n]

        for g in range(G):
            # RDR: X ~ NB(mu, phi)
            rdr_mean = T_n[n] * lambda_g[g] * (theta * mu_gk[g, k] + (1 - theta))
            rdr_mean = max(rdr_mean, 1e-12)
            p_nb = inv_phi / (inv_phi + rdr_mean)
            X[g, n] = rng.negative_binomial(inv_phi, p_nb) if p_nb < 1 else 0

            # D: total allele counts per bin
            d_mean = D_total_n[n] * lambda_g[g] * (theta * mu_gk[g, k] + (1 - theta))
            d_gn = rng.poisson(max(d_mean, 1e-12))
            D[g, n] = d_gn

            # BAF: Y ~ BetaBin(D, tau*p_hat, tau*(1-p_hat))
            if d_gn > 0:
                p_baf = BAF[g, k]
                rdr = mu_gk[g, k]
                denom = rdr * theta + (1 - theta)
                p_hat = (p_baf * rdr * theta + 0.5 * (1 - theta)) / max(denom, 1e-12)
                p_hat = np.clip(p_hat, 1e-6, 1 - 1e-6)
                Y[g, n] = scipy.stats.betabinom.rvs(
                    d_gn,
                    tau * p_hat,
                    tau * (1 - p_hat),
                    random_state=rng,
                )

    logging.info(f"Simulated counts: X sum={X.sum()}, Y sum={Y.sum()}, D sum={D.sum()}")
    return X, Y, D


# ──────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────
def write_copytyping_input(
    out_dir,
    cnv_df,
    X,
    Y,
    D,
    coords,
    clone_labels,
    tumor_proportions,
):
    """Write all files needed by `copytyping inference --assay_type VISIUM`."""
    os.makedirs(out_dir, exist_ok=True)
    input_dir = os.path.join(out_dir, "inputs")
    os.makedirs(input_dir, exist_ok=True)
    N = X.shape[1]
    barcodes = [f"spot_{i:04d}-1_U1" for i in range(N)]

    # copytyping input files → inputs/
    pd.DataFrame(barcodes).to_csv(
        os.path.join(input_dir, "barcodes.tsv.gz"),
        sep="\t",
        header=False,
        index=False,
        compression="gzip",
    )
    cnv_df.to_csv(os.path.join(input_dir, "cnv_segments.tsv"), sep="\t", index=False)
    sparse.save_npz(os.path.join(input_dir, "X_count.npz"), sparse.csr_matrix(X))
    sparse.save_npz(os.path.join(input_dir, "Y_count.npz"), sparse.csr_matrix(Y))
    sparse.save_npz(os.path.join(input_dir, "D_count.npz"), sparse.csr_matrix(D))

    # minimal h5ad with spatial metadata for squidpy compatibility
    adata = AnnData(
        X=sparse.csr_matrix(X.T),
        obs=pd.DataFrame(index=barcodes),
    )
    adata.obsm["spatial"] = coords.astype(np.int64)
    library_id = "U1"
    max_coord = int(coords.max()) + 500
    hires_scalef = 0.1
    img_size = int(max_coord * hires_scalef) + 1
    dummy_hires = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    lowres_scalef = 0.03
    lowres_size = int(max_coord * lowres_scalef) + 1
    dummy_lowres = np.ones((lowres_size, lowres_size, 3), dtype=np.uint8) * 255
    adata.uns["spatial"] = {
        library_id: {
            "images": {"hires": dummy_hires, "lowres": dummy_lowres},
            "scalefactors": {
                "spot_diameter_fullres": 160.0,
                "tissue_hires_scalef": hires_scalef,
                "tissue_lowres_scalef": lowres_scalef,
                "fiducial_diameter_fullres": 260.0,
            },
            "metadata": {},
        }
    }
    adata.write(os.path.join(input_dir, "VISIUM.h5ad"))

    # cell_type reference → inputs/
    pd.DataFrame(
        {
            "BARCODE": barcodes,
            "path_label": np.where(clone_labels == "normal", "normal", "tumor"),
        }
    ).to_csv(
        os.path.join(input_dir, "cell_types.tsv.gz"),
        sep="\t",
        index=False,
        compression="gzip",
    )

    # ground truth → top-level
    pd.DataFrame(
        {
            "BARCODE": barcodes,
            "true_label": clone_labels,
            "true_theta": tumor_proportions,
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    ).to_csv(os.path.join(out_dir, "ground_truth.tsv"), sep="\t", index=False)

    # ground truth spatial plots
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_ground_truth(coords, clone_labels, tumor_proportions, plot_dir)

    logging.info(f"Wrote copytyping input to {out_dir}")


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
def plot_ground_truth(coords, clone_labels, tumor_proportions, plot_dir):
    """Plot ground truth clone labels and tumor purity on spatial coordinates."""
    clone_names = sorted(set(clone_labels), key=lambda x: (x != "normal", x))
    # color map: gray for normal, tab10 for clones
    palette = {"normal": "#b0b0b0"}
    tab10 = plt.cm.tab10.colors
    ci = 0
    for c in clone_names:
        if c != "normal":
            palette[c] = mcolors.to_hex(tab10[ci])
            ci += 1

    spot_size = 8
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=200, facecolor="white")

    # panel 1: clone labels
    for c in clone_names:
        mask = clone_labels == c
        axes[0].scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=palette[c],
            s=spot_size,
            label=c,
            linewidths=0,
        )
    axes[0].set_title("Ground truth clone labels")
    axes[0].legend(markerscale=3, frameon=False)
    axes[0].set_aspect("equal")
    axes[0].invert_yaxis()

    # panel 2: tumor purity
    sc = axes[1].scatter(
        coords[:, 0],
        coords[:, 1],
        c=tumor_proportions,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        s=spot_size,
        linewidths=0,
    )
    axes[1].set_title("Ground truth tumor purity")
    axes[1].set_aspect("equal")
    axes[1].invert_yaxis()
    fig.colorbar(sc, ax=axes[1], shrink=0.7)

    # panel 3: normal vs tumor binary
    is_tumor = clone_labels != "normal"
    axes[2].scatter(
        coords[~is_tumor, 0],
        coords[~is_tumor, 1],
        c="#b0b0b0",
        s=spot_size,
        label="normal",
        linewidths=0,
    )
    axes[2].scatter(
        coords[is_tumor, 0],
        coords[is_tumor, 1],
        c="#1f77b4",
        s=spot_size,
        label="tumor",
        linewidths=0,
    )
    axes[2].set_title("Ground truth tumor/normal")
    axes[2].legend(markerscale=3, frameon=False)
    axes[2].set_aspect("equal")
    axes[2].invert_yaxis()

    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.tight_layout()
    out_path = os.path.join(plot_dir, "ground_truth_spatial.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved ground truth plot to {out_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Simulate Visium data for copytyping validation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # CNP source (mutually exclusive)
    cnp_group = parser.add_mutually_exclusive_group(required=True)
    cnp_group.add_argument(
        "--cnp_file",
        type=str,
        help="Existing cnv_segments.tsv (HATCHet2 format)",
    )
    cnp_group.add_argument(
        "--genome_size",
        type=str,
        help="Chromosome sizes file for random CNA generation",
    )

    # spatial / clone parameters
    parser.add_argument("--n_spots", type=int, default=3000)
    parser.add_argument("--n_clones", type=int, default=3)
    parser.add_argument(
        "--n_cnas", type=int, default=2, help="CNAs per clone (random mode)"
    )
    parser.add_argument(
        "--cna_size",
        type=lambda x: int(float(x)),
        default=int(3e7),
        help="CNA length in bp (random mode)",
    )
    parser.add_argument("--max_tumor_prop", type=float, default=0.9)
    parser.add_argument(
        "--make_pure", action="store_true", help="Pure clones (theta=1 for tumor)"
    )

    # simulation parameters
    parser.add_argument("--tau", type=float, default=50.0, help="BB dispersion")
    parser.add_argument("--phi", type=float, default=30.0, help="NB dispersion")
    parser.add_argument("--rdr_lognormal_mean", type=float, default=8.0)
    parser.add_argument("--rdr_lognormal_sigma", type=float, default=0.4)
    parser.add_argument("--baf_lognormal_mean", type=float, default=6.0)
    parser.add_argument("--baf_lognormal_sigma", type=float, default=0.4)

    parser.add_argument("--sample", type=str, default="simulated")
    parser.add_argument("-o", "--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # 1. CNV profile
    if args.cnp_file:
        cnv_df = load_cnp(args.cnp_file)
        n_clones = len(cnv_df["CNP"].iloc[0].split(";")) - 1
        logging.info(
            f"Loaded CNP from {args.cnp_file}: "
            f"{len(cnv_df)} segments, {n_clones} tumor clones"
        )
    else:
        cnv_df, all_cnas = generate_random_cnp(
            args.genome_size,
            args.n_clones,
            args.n_cnas,
            args.cna_size,
            args.seed,
        )
        n_clones = args.n_clones
        os.makedirs(args.out_dir, exist_ok=True)
        cna_rows = []
        for k, events in all_cnas.items():
            for cna in events:
                cna_rows.append(
                    {
                        "clone": f"clone{k}",
                        "chr": cna.chr,
                        "start": cna.start,
                        "end": cna.end,
                        "A_copy": cna.A_copy,
                        "B_copy": cna.B_copy,
                    }
                )
        pd.DataFrame(cna_rows).to_csv(
            os.path.join(args.out_dir, "truth_cna_events.tsv"),
            sep="\t",
            index=False,
        )

    A, B, C, BAF = parse_cn_arrays(cnv_df)

    # 2. Spatial layout
    coords, clone_labels, tumor_proportions = simulate_spatial_hexagon(
        args.n_spots,
        n_clones,
        args.max_tumor_prop,
        args.make_pure,
        args.seed,
    )

    # 3. Simulate counts
    X, Y, D = simulate_counts(
        A,
        B,
        C,
        BAF,
        clone_labels,
        tumor_proportions,
        tau=args.tau,
        phi=args.phi,
        rdr_lognormal_mean=args.rdr_lognormal_mean,
        rdr_lognormal_sigma=args.rdr_lognormal_sigma,
        baf_lognormal_mean=args.baf_lognormal_mean,
        baf_lognormal_sigma=args.baf_lognormal_sigma,
        random_state=args.seed,
    )

    # 4. Write output
    write_copytyping_input(
        args.out_dir,
        cnv_df,
        X,
        Y,
        D,
        coords,
        clone_labels,
        tumor_proportions,
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()
