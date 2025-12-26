## Copy-typing
Copy-typing takes copy-number profile and phased genotype inferred by HATCHet2 using bulk WGS/WES data, and infer per-cell/spot clone labels for single-cell/spot measurements including scRNA-seq, scATAC-seq, scMulitome, and Visium data types. For Visium-like spot-level data, copy-typing also infers per-spot tumor proportion.

### Dependencies
See `environment.yaml`

### Install
After creating a python environment using either conda or pip venv and having all dependencies installed, install copy-typing via `pip install -e .`

### Inputs
#### Sample file
```txt
SAMPLE\tREP_ID\tDATA_TYPE\tPATH_to_h5ad\tPATH_to_barcodes\tPATH_to_cellsnp_lite\tPATH_to_annotation
```
same `REP_ID` refers to paired measurements with consistent barcode labels, such as scMultiome data.

### Commands
