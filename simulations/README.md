# Visium Simulation for Copytyping Validation

## Overview

Simulates Visium spatial transcriptomics data with known ground-truth CNA profiles,
clone labels, and tumor purities. Generates bbc-level count matrices and seg.ucn.tsv
following the same generative model used in copytyping's Spot_Model.

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

Outputs:
- `inputs/cnv_segments.tsv` — bbc-level bin coordinates
- `inputs/X_count.npz, Y_count.npz, D_count.npz` — bbc-level count matrices
- `inputs/barcodes.tsv.gz, VISIUM.h5ad, cell_types.tsv.gz`
- `seg.ucn.tsv` — segment-level copy numbers (HATCHet format)
- `ground_truth.tsv`

### 2. Run copytyping inference

```bash
conda run -n infercnvpy copytyping inference \
    --assay_type VISIUM \
    --gex_dir simulations/sim_HT112C1_pure/inputs \
    --seg_ucn simulations/sim_HT112C1_pure/seg.ucn.tsv \
    --sample simulated \
    --cell_type simulations/sim_HT112C1_pure/inputs/cell_types.tsv.gz \
    --ref_label path_label \
    --genome_size data/hg38.chrom.sizes \
    --region_bed data/chm13v2.0_region.bed \
    -o simulations/sim_HT112C1_pure/outs
```

### 3. Evaluate clone-level accuracy

```bash
conda run -n infercnvpy python simulations/evaluate_simulation.py \
    --ground_truth simulations/sim_HT112C1_pure/ground_truth.tsv \
    --annotations simulations/sim_HT112C1_pure/outs/sample.VISIUM.annotations.tsv \
    -o simulations/sim_HT112C1_pure/outs/clone_evaluation.tsv
```

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `--n_spots` | 3000 | Number of Visium spots |
| `--n_clones` | 3 | Number of tumor clones |
| `--n_cnas` | 2 | CNAs per clone (random mode) |
| `--cna_size` | 30Mb | CNA length (random mode) |
| `--make_pure` | False | Pure clones (theta=1) vs sigmoid purity |
| `--max_tumor_prop` | 0.9 | Max tumor purity (mixed mode) |
| `--tau` | 50 | BetaBinomial dispersion |
| `--phi` | 30 | NegBinomial dispersion |
| `--seed` | 42 | Random seed |
