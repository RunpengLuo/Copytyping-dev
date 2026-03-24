# Visium Simulation for Copytyping Validation

## Overview

Simulates Visium spatial transcriptomics data with known ground-truth CNA profiles,
clone labels, and tumor purities. Generates count matrices (X, Y, D) following the
same generative model used in copytyping's Spot_Model (NB for RDR, BetaBinom for BAF).

Based on CalicoST simulation framework (Supplementary S15).

## Steps

### 1. Simulate data

**Using an existing CNP (e.g., HT112C1):**

```bash
conda run -n infercnvpy python simulations/simulate_visium.py \
    --cnp_file test/HT112C1/bb_diploid_n4/VISIUM/cnv_segments.tsv \
    --n_spots 3000 --make_pure --seed 42 \
    -o simulations/sim_HT112C1_pure
```

**With random CNAs:**

```bash
conda run -n infercnvpy python simulations/simulate_visium.py \
    --genome_size data/hg38.chrom.sizes \
    --n_clones 3 --n_cnas 2 --cna_size 3e7 \
    --seed 42 \
    -o simulations/sim_random
```

### 2. Run copytyping inference on simulated data

```bash
conda run -n infercnvpy copytyping inference \
    --assay_type VISIUM \
    --gex_dir simulations/sim_HT112C1_pure/inputs \
    --sample simulated \
    --cell_type simulations/sim_HT112C1_pure/inputs/cell_types.tsv.gz \
    --ref_label path_label \
    --genome_size data/hg38.chrom.sizes \
    --region_bed data/chm13v2.0_region.bed \
    --fit_mode hybrid \
    -o simulations/sim_HT112C1_pure/outs
```

### 3. Evaluate clone-level accuracy

```bash
conda run -n infercnvpy python simulations/evaluate_simulation.py \
    --ground_truth simulations/sim_HT112C1_pure/ground_truth.tsv \
    --annotations simulations/sim_HT112C1_pure/outs/sample.VISIUM.annotations.tsv \
    -o simulations/sim_HT112C1_pure/outs/clone_evaluation.tsv
```

Outputs: ARI, clone-level accuracy/F1, confusion matrix, theta correlation.

### 4. Compare fit modes

Run with `--fit_mode allele_only` or `--fit_mode total_only` to compare
BAF-only vs RDR-only vs hybrid performance.

### 5. Check results

- Binary tumor/normal metrics: `<out>/sample.VISIUM.evaluation.tsv`
- Clone-level metrics: `<out>/clone_evaluation.tsv`
- Confusion matrix: `<out>/clone_evaluation.confusion.tsv`
- Ground truth: `<sim_dir>/ground_truth.tsv`

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `--n_spots` | 3000 | Number of Visium spots |
| `--n_clones` | 3 | Number of tumor clones |
| `--n_cnas` | 2 | CNAs per clone (random mode) |
| `--cna_size` | 30Mb | CNA length (random mode) |
| `--make_pure` | False | Pure clones (theta=1) vs sigmoid tumor purity |
| `--max_tumor_prop` | 0.9 | Max tumor purity (mixed mode) |
| `--tau` | 50 | BetaBinomial dispersion |
| `--phi` | 30 | NegBinomial dispersion |
| `--seed` | 42 | Random seed |

## Generative model

- **RDR**: `X_{g,n} ~ NB(T_n * lam_g * (theta_n * mu_{g,k} + (1-theta_n)), phi)`
- **BAF**: `Y_{g,n} ~ BetaBin(D_{g,n}, tau * p_hat, tau * (1-p_hat))`
  - `p_hat = (theta_n * mu_{g,k} * p_{g,k} + 0.5*(1-theta_n)) / (theta_n * mu_{g,k} + (1-theta_n))`

Where `mu_{g,k}` is clone-specific RDR, `p_{g,k}` is clone BAF from CNP, and
`theta_n` is per-spot tumor purity.
