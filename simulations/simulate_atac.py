"""
Simulate minimal scATAC test data for copytyping inference.

Layout (single chromosome, chr1):
  - 100 cells: 30 normal, 30 clone1, 40 clone2
  - Clone copy numbers (normal | clone1 | clone2):
      normal  1|1  (diploid, balanced)
      clone1  1|0  (haploid, LOH)
      clone2  2|1  (triploid, imbalanced)
  - 20 genomic bins of 5 Mb each on chr1

Output directory: tests/atac_test/
  cnv_segments.tsv   -- CNP + PROPS per bin
  barcodes.tsv.gz    -- 100 barcode lines
  X_count.npz        -- total read depth (G x N)  [NB model: observed counts]
  Y_count.npz        -- B-allele counts  (G x N)  [BB model: b-allele]
  D_count.npz        -- total allele counts (G x N) [BB model: total allele]
"""

import gzip
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

RNG = np.random.default_rng(42)

# ---------- parameters ----------
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "atac_test")
os.makedirs(OUT_DIR, exist_ok=True)

N_NORMAL, N_CLONE1, N_CLONE2 = 30, 30, 40
N = N_NORMAL + N_CLONE1 + N_CLONE2          # 100 cells
G = 20                                       # genomic bins
CHROM = "chr1"
BIN_SIZE = 5_000_000                         # 5 Mb

# clone index per cell: 0=normal, 1=clone1, 2=clone2
cell_clone = np.array([0] * N_NORMAL + [1] * N_CLONE1 + [2] * N_CLONE2)

# bulk proportions used by HATCHet (normal, clone1, clone2)
BULK_PROPS = [N_NORMAL / N, N_CLONE1 / N, N_CLONE2 / N]   # 0.3, 0.3, 0.4

# ---------- 1. cnv_segments.tsv ----------
# CNP string: "normal_A|normal_B;clone1_A|clone1_B;clone2_A|clone2_B"
CNP_STR = "1|1;1|0;2|1"
PROPS_STR = ";".join(f"{p:.4f}" for p in BULK_PROPS)

starts = [i * BIN_SIZE for i in range(G)]
ends   = [(i + 1) * BIN_SIZE for i in range(G)]

cnv_df = pd.DataFrame({
    "#CHR":  CHROM,
    "START": starts,
    "END":   ends,
    "CNP":   CNP_STR,
    "PROPS": PROPS_STR,
})
cnv_df.to_csv(os.path.join(OUT_DIR, "cnv_segments.tsv"), sep="\t", index=False)

# ---------- 2. barcodes.tsv.gz ----------
# Format: {barcode}_{rep_id}   (rep_id = "1" for all cells here)
barcodes = [f"CELL{str(i).zfill(4)}_1" for i in range(N)]
bc_path = os.path.join(OUT_DIR, "barcodes.tsv.gz")
with gzip.open(bc_path, "wt") as f:
    for bc in barcodes:
        f.write(bc + "\n")

# ---------- 3. count matrices ----------
# Clone copy numbers
cn_A = np.array([1, 1, 2])   # A-allele copies per clone
cn_B = np.array([1, 0, 1])   # B-allele copies per clone
cn_T = cn_A + cn_B            # total copies: [2, 1, 3]

# BAF = B / (A + B) per clone; clone1 BAF=0 (LOH)
baf = cn_B / cn_T.astype(float)   # [0.5, 0.0, 0.333...]

# Average read depth per bin per cell, scaled by copy number relative to diploid baseline
BASE_DEPTH = 30   # diploid baseline mean depth

X = np.zeros((G, N), dtype=np.int32)
Y = np.zeros((G, N), dtype=np.int32)
D = np.zeros((G, N), dtype=np.int32)

NB_DISPERSION = 15   # negative-binomial dispersion (higher = less overdispersion)

for n in range(N):
    clone = cell_clone[n]
    mean_d = BASE_DEPTH * cn_T[clone] / 2.0   # RDR-scaled depth

    # D ~ NegBin(r, p) with mean=mean_d, dispersion=NB_DISPERSION
    p = NB_DISPERSION / (NB_DISPERSION + mean_d)
    d = RNG.negative_binomial(NB_DISPERSION, p, size=G).astype(np.int32)
    D[:, n] = d

    # Y = allele-informative reads, ~half of D at heterozygous SNPs
    y = RNG.binomial(d, 0.5).astype(np.int32)
    Y[:, n] = y

    # X = B-allele reads
    clone_baf = baf[clone]
    if clone_baf == 0.0:
        # LOH: no B-allele reads; add a tiny Poisson noise floor
        x = RNG.poisson(0.05, size=G).astype(np.int32)
    else:
        x = RNG.binomial(y, clone_baf).astype(np.int32)
    X[:, n] = x

sp.save_npz(os.path.join(OUT_DIR, "X_count.npz"), sp.csr_matrix(D))  # total depth -> NB
sp.save_npz(os.path.join(OUT_DIR, "Y_count.npz"), sp.csr_matrix(X))  # B-allele -> BB Y
sp.save_npz(os.path.join(OUT_DIR, "D_count.npz"), sp.csr_matrix(Y))  # total allele -> BB D

# ---------- summary ----------
print(f"Wrote test data to: {OUT_DIR}")
print(f"  cnv_segments.tsv : {G} bins, CNP={CNP_STR}, PROPS={PROPS_STR}")
print(f"  barcodes.tsv.gz  : {N} cells (normal={N_NORMAL}, clone1={N_CLONE1}, clone2={N_CLONE2})")
print(f"  X_count.npz (depth)  : shape {D.shape}, mean total depth by clone:")
print(f"  Y_count.npz (B-allele): shape {X.shape}")
print(f"  D_count.npz (T-allele): shape {Y.shape}")
for k, name in enumerate(["normal", "clone1", "clone2"]):
    mask = cell_clone == k
    print(f"    {name:8s}: mean depth={D[:, mask].mean():.2f}  mean B-allele={X[:, mask].mean():.2f}  mean T-allele={Y[:, mask].mean():.2f}")
