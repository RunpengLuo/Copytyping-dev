## Copy-typing
Copy-typing takes copy-number profile and phased genotype inferred by HATCHet3 using bulk WGS/WES data, and infer per-cell/spot clone labels for single-cell/spot DNA/RNA measurements including scRNA-seq, scATAC-seq, paired scMultiome, and Visium. For Visium data with resolution ~50um per spot (5~10 cells per spot), copy-typing also infers per-spot tumor proportion.

### Dependencies
See `environment.yaml`.

### Install
After creating a python environment using either `conda` or `pip venv` and having all dependencies installed, install copy-typing via `pip install -e .`

### Commands
```sh
# infer cell/spot clone labels and per-spot clone proportion (for Visium data)
copytyping inference -h
```
