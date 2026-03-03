## Copy-typing

Copy-typing infers per-cell/spot clone labels for single-cell and spatial transcriptomic data using copy-number profiles from [HATCHet2][hatchet] bulk WGS/WES. It supports scRNA-seq, scATAC-seq, paired snMultiome, and Visium. For Visium (~50 µm spots, 5–10 cells/spot), it additionally infers per-spot tumor proportion.

### Installation

```sh
# Create environment and install
conda env create -f environment.yaml
conda activate copytyping
pip install -e .
```

External tools required on `PATH`: `bedtools`, `bcftools`, `samtools`, `mosdepth`, `whatshap`.

### Getting started

```sh
# scRNA-seq
copytyping inference \
    --assay_type scRNA \
    --sample SAMPLE_NAME \
    --gex_dir /path/to/gex \
    -o /path/to/output \
    --genome_size data/GRCh38.sizes \
    --region_bed data/chm13v2.0_region.bed

# Paired snMultiome (GEX + ATAC)
copytyping inference \
    --assay_type multiome \
    --sample SAMPLE_NAME \
    --gex_dir /path/to/gex \
    --atac_dir /path/to/atac \
    -o /path/to/output \
    --genome_size data/CHM13v2.sizes \
    --region_bed data/chm13v2.0_region.bed

# Visium spatial data
copytyping inference \
    --assay_type VISIUM \
    --sample SAMPLE_NAME \
    --gex_dir /path/to/gex \
    -o /path/to/output \
    --genome_size data/GRCh38.sizes \
    --region_bed data/chm13v2.0_region.bed
```

### Assay types

| `--assay_type` | Input modalities | Model |
|---|---|---|
| `scRNA` | GEX only | Cell_Model |
| `scATAC` | ATAC only | Cell_Model |
| `multiome` | GEX + ATAC | Cell_Model |
| `VISIUM` | GEX (spatial) | Spot_Model |

### Input format

Each input directory (`--gex_dir` or `--atac_dir`) must contain:

```
barcodes.tsv.gz        # cell/spot barcodes
cnv_segments.tsv       # HATCHet2 segment copy-number profile
X_count.npz            # B-allele count matrix (cells × segments)
Y_count.npz            # total allele count matrix
D_count.npz            # read depth ratio matrix
{assay_type}.h5ad      # AnnData with cell-type annotations (optional)
```

The `.h5ad` file is optional but enables validation against reference cell-type labels (see **`--ref_label`**).

### Options

#### Inference

| Option | Default | Description |
|---|---|---|
| `--method` | `copytyping` | Assignment method: `copytyping` (EM), `kmeans`, `ward`, `leiden` |
| `--niters` | `3000` | Maximum EM iterations |
| `--tau` | `50.0` | Initial Beta-Binomial dispersion (BAF); over-dispersion = 1/τ |
| `--phi` | `30.0` | Initial Negative-Binomial dispersion (RDR); over-dispersion = 1/φ |
| `--fix_BB_dispersion` | off | Fix BB dispersion after initialization |
| `--fix_NB_dispersion` | off | Fix NB dispersion after initialization |
| `--share_BB_dispersion` | off | Share BB dispersion across CN states |
| `--share_NB_dispersion` | off | Share NB dispersion across CN states |
| `--fix_tumor_pruity` | off | Fix per-spot tumor purity (Visium only) |

#### Assignment thresholds

| Option | Default | Description |
|---|---|---|
| `--posterior_thres` | `0.50` | Label cell/spot as unassigned if max posterior < threshold |
| `--margin_thres` | `0.10` | Label cell/spot as unassigned if top-2 margin < threshold |
| `--ref_label` | `cell_type` | Column in `.h5ad` obs used as reference label for evaluation |
| `--refine_label_by_reference` | off | Mark unassigned if prediction disagrees with reference cell type |

#### Plotting

| Option | Default | Description |
|---|---|---|
| `--img_type` | `png` | Output image format: `png`, `pdf`, `svg` |
| `--dpi` | `300` | Image resolution |
| `--heatmap_agg` | `10` | Number of cells/spots to aggregate per row in heatmaps |
| `--transparent` | off | Transparent figure background |

### Output

```
<out_dir>/
  <prefix>.<assay_type>.annotations.tsv      # clone assignments and posterior probabilities
  <prefix>.<assay_type>.evaluation.tsv       # precision/recall/F1/AUC (if ref_label available)
  plots/
    heatmaps/    # BAF, log2RDR, posterior heatmaps per data type
    scatter/     # 1D genome-wide BAF/RDR scatter
    validation/  # posterior distribution, cross-tabulation with reference labels
    visium/      # H&E tissue overlays with clone and tumor-purity annotations (VISIUM only)
```

### Statistical model

The EM algorithm fits a mixture model over clones defined by the HATCHet2 copy-number profile. Two observation types are used jointly (**hybrid** mode):

- **BAF** (B-allele frequency): Beta-Binomial likelihood over per-cell allele counts
- **RDR** (read-depth ratio): Negative-Binomial likelihood over per-cell total counts

Clone mixture proportions (π) are initialized from HATCHet2 bulk estimates and held fixed. For Visium, each spot has a latent tumor purity parameter (θ) that is inferred jointly with the clone assignments.

### Reference data

Pre-bundled in `data/`:
- `GRCh38.sizes`, `CHM13v2.sizes` — chromosome size references
- `chm13v2.0_region.bed` — chromosome arm BED file
- `gencode.v38.basic.annotation.gtf.gz` — gene annotation (GRCh38)

[hatchet]: https://github.com/raphael-group/hatchet
