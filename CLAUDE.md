# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Copy-typing** is a bioinformatics research tool from the Raphael Lab that infers per-cell/spot clone labels and tumor proportions for single-cell and spatial genomics data (scRNA-seq, scATAC-seq, paired scMultiome, Visium) using copy-number profiles from HATCHet2 bulk WGS/WES.

## Setup

```bash
# Create and activate the conda environment
conda env create -f environment.yaml
conda activate copytyping  # (or whatever name is set in environment.yaml)

# Install the package in editable mode
pip install -e .
```

External tools required on PATH: `bedtools`, `bcftools`, `samtools`, `mosdepth`, `whatshap`.

## Running the Tool

```bash
# Main entry point
copytyping inference --assay_type {scRNA|scATAC|multiome|VISIUM} \
    --sample <name> \
    --gex_dir <path> \
    --atac_dir <path> \
    --method {copytyping|kmeans|ward|leiden} \
    -o <output_dir> \
    --genome_size data/GRCh38.sizes \
    --region_bed data/chm13v2.0_region.bed

# See run.sh and run_visium.sh for working examples
# See example/sample.tsv for input file format
```

There is no test suite or linting configuration in this project.

## Architecture

This repo implements only the **inference** stage. Preprocessing (gene annotation, fragment counting) and segmentation (combining counts into segment-level matrices) are handled by external pipelines and are not part of this codebase. The `inference` command consumes their preprocessed outputs.

### Core Data Flow

```
Sample TSV → SX_Data (loads h5ad, count matrices, CNV segments)
    → Cell_Model (scRNA/scATAC/multiome) or Spot_Model (Visium)
    → EM Inference → Clone posteriors
    → Clustering (if method != copytyping)
    → Validation + Plots → Output TSVs + figures
```

### Key Modules

| Module | Role |
|--------|------|
| `sx_data/sx_data.py` | `SX_Data` class — container for BAF (X/Y counts), RDR (D counts), CNV profiles, barcodes |
| `inference/base_model.py` | Abstract base EM model with E-step logic and parameter initialization |
| `inference/cell_model.py` | Single-cell EM model (assumes tumor purity = 1.0) |
| `inference/spot_model.py` | Visium spot model with per-spot tumor purity inference |
| `inference/likelihood_funcs.py` | Beta-Binomial (BAF) and Negative-Binomial (RDR) PMF computations |
| `inference/model_utils.py` | RDR/BAF computation, baseline proportion estimation |
| `inference/clustering.py` | Alternative clustering: k-means, Leiden, Ward |
| `inference/inference.py` | Orchestrates the full inference pipeline |
| `inference/validation.py` | Evaluation metrics (precision, recall, F1, AUC) |
| `external.py` | Wrappers around `bedtools` for genomic interval operations |
| `plot/` | Visualization: heatmaps, UMAP, Visium spatial plots, CNP profiles |

### Statistical Model

The EM algorithm fits two observation types jointly (hybrid mode by default):
- **BAF** (B-allele frequency): Beta-Binomial likelihood
- **RDR** (read depth ratio): Negative-Binomial likelihood

Fit modes: `allele_only`, `total_only`, `hybrid`.

### Assay Type Constants (utils.py)

```python
SPOT_ASSAYS = {"VISIUM"}           # use Spot_Model
CELL_ASSAYS = {"scRNA", "scATAC", "multiome"}  # use Cell_Model
GEX_ASSAYS  = {"scRNA", "multiome", "VISIUM"}
ATAC_ASSAYS = {"scATAC", "multiome"}
```

### Output Files

- `<sample>.<rep_id>.annotations.copytyping.tsv` — clone assignments and posterior probabilities
- `<sample>.<rep_id>.evaluation.copytyping.tsv` — validation metrics
- Heatmap directories and PNG plots for BAF/RDR, UMAP, Visium tissue overlays, CNP

## Reference Data Files

Pre-bundled in `data/`:
- `GRCh38.sizes` / `CHM13v2.sizes` — chromosome size references
- `chm13v2.0_region.bed` — chromosome arm BED file
- `gencode.v38.basic.annotation.gtf.gz` — gene annotation

## Development Notes

- The `cleanup` branch (current) is a major refactoring of `main`: core inference split into `base_model.py` + `likelihood_funcs.py` + `model_utils.py`.
- No automated tests exist; validation is done via `inference/validation.py` metrics on real data runs.
- Only the `inference` CLI subcommand is currently active in `__main__.py`.
