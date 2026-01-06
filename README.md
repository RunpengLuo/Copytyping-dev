## Copy-typing
Copy-typing takes copy-number profile and phased genotype inferred by HATCHet2 using bulk WGS/WES data, and infer per-cell/spot clone labels for single-cell/spot DNA/RNA measurements including scRNA-seq, scATAC-seq, paired scMultiome, and Visium. For Visium data with resolution ~50um per spot (5~10 cells per spot), copy-typing also infers per-spot tumor proportion.

### Dependencies
See `environment.yaml`.

### Install
After creating a python environment using either `conda` or `pip venv` and having all dependencies installed, install copy-typing via `pip install -e .`

### Commands
```sh
# Step 1. Preprocess
# annotate gene locations, scRNA-seq or Visium.
python ./src/copytyping/preprocess/prep_rna_annadata.py -h

# obtain tile level fragment counts, scATAC-seq
python ./src/copytyping/preprocess/prep_atac_annadata.py -h

# Step 2. Segmentation
# transform gene/tile-level data matrix into segment-level data matrix.
copytyping combine-counts -h

# Step 3. Inference
# infer cell/spot clone labels and per-spot clone proportion (for Visium data)
copytyping inference -h
```

### Inputs
#### Sample file
The TSV-format sample file records all the samples retrived from same patient, refers to `./example/sample.tsv`.
* `SAMPLE`: patient ID, all rows should be the same.
* `REP_ID`: replicate ID, e.g., `U<num>`. For paired multiome data, use same `REP_ID` for both rows.
* `DATA_TYPE`: pick from: GEX, ATAC, VISIUM.
* `PATH_to_h5ad`: e.g., `/path/to/10x_rna.h5ad`. preprocessed from 10x data by `./src/copytyping/preprocess/prep_rna_annadata.py` (RNA) or `./src/copytyping/preprocess/prep_atac_annadata.py` (ATAC)
* `PATH_to_barcodes`: e.g., `/path/to/samples.tsv`. For paired multiome data, use same file.
* `PATH_to_cellsnp_lite`: e.g., `/path/to/cellsnp_lite_output/`.
* `PATH_to_annotation`: e.g., `/path/to/celltype.tsv`

#### HATCHet-related files
* `/path/to/sample.seg.ucn`: copy-number profile.
* `/path/to/snp_info.phased.tsv.gz`: phased SNP information.

#### Auxiliary files
* `/path/to/reference.sizes`: genome size file.
* `/path/to/reference.bed`: genome arm BED file.

### Outputs
* `<out_prefix>_copytyping_assignment/<sample>.<rep_id>.annotations.copytyping.tsv`: per-cell/spot clone label, per-clone posterior probabilities, spot tumor proportion (if avail).
* `<out_prefix>_copytyping_plot/<data_type>_heatmap/`: BAF/log2RDR cell/spot-level segmented heatmap.
* `<out_prefix>_copytyping_plot/<sample>.<rep_id>.posteriors.copytyping.png`: posterior probability histograms.
* ...
