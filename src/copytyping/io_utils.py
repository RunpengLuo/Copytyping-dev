import logging

import numpy as np
import pandas as pd
from scipy import sparse

from anndata import AnnData

from copytyping.utils import read_seg_ucn_file, sort_df_chr


##################################################
# cell types & barcodes
##################################################


def read_cell_types(ct_file: str | None, req_cols: set[str]):
    """Load a cell-type annotation TSV; assert every column in ``req_cols`` exists."""
    if ct_file is None:
        return None
    cell_type_df = pd.read_table(ct_file)
    for req_col in req_cols:
        assert req_col in cell_type_df.columns, (
            f"read_cell_types {ct_file}: {req_col} is missing"
        )
    return cell_type_df


def annotate_adata_celltype(
    adata: AnnData,
    cell_type_df: pd.DataFrame,
    ref_label: str,
    assay_type: str,
):
    """Add cell_type annotations to adata.obs from cell_type_df."""
    ct_map = cell_type_df.set_index("BARCODE")[ref_label]
    if ref_label in adata.obs.columns:
        logging.warning(
            f"overwriting existing '{ref_label}' column "
            f"in {assay_type} h5ad obs with cell_type_df"
        )
    adata.obs[ref_label] = (
        adata.obs_names.to_series().map(ct_map).fillna("Unknown").values
    )


def read_barcodes(
    barcode_file: str,
    cell_type_df: pd.DataFrame | None = None,
    ref_label: str | None = None,
):
    """Read the per-cell barcode list into a DataFrame (BARCODE + REP_ID).

    REP_ID is everything after the first underscore ("ACGT-1_U1" -> "U1"). When
    ``cell_type_df`` + ``ref_label`` are given, the ref_label column (and its
    optional ``{ref_label}-tumor_purity`` column) is left-merged in; unmatched
    cells get "Unknown". Filtering of uninformative labels is left to
    ``exclude_barcodes``.
    """
    barcodes_df = pd.read_table(
        barcode_file, sep="\t", header=None, names=["BARCODE"], dtype=str
    )
    barcodes_df["REP_ID"] = barcodes_df["BARCODE"].str.split("_", n=1).str[1]
    if cell_type_df is not None and ref_label is not None:
        merge_cols = ["BARCODE", ref_label]
        ref_purity_col = f"{ref_label}-tumor_purity"
        if ref_purity_col in cell_type_df.columns:
            merge_cols.append(ref_purity_col)
        barcodes_df = pd.merge(
            left=barcodes_df,
            right=cell_type_df[merge_cols],
            on="BARCODE",
            how="left",
            validate="1:1",
            sort=False,
        )
        barcodes_df[ref_label] = barcodes_df[ref_label].fillna("Unknown").astype(str)
    return barcodes_df


def exclude_barcodes(
    barcodes_df: pd.DataFrame,
    exclude_labels: set[str],
    ref_label: str,
    *mats: sparse.csr_matrix,
):
    """Drop barcodes whose ref_label is in exclude_labels and column-subset
    each aligned matrix in ``mats``.

    Returns ``(barcodes_df, *mats)`` filtered to kept cells.
    """
    if ref_label not in barcodes_df.columns:
        return (barcodes_df, *mats)
    keep = ~barcodes_df[ref_label].isin(exclude_labels)
    n_excl = int((~keep).sum())
    if n_excl == 0:
        return (barcodes_df, *mats)
    idx = np.where(keep.values)[0]
    barcodes_df = barcodes_df[keep].reset_index(drop=True)
    mats = tuple(m[:, idx] for m in mats)
    logging.info(
        f"excluded {n_excl} cells with ref_label in {exclude_labels}, "
        f"{len(barcodes_df)} remaining"
    )
    return (barcodes_df, *mats)


##################################################
# bulk CN profile (HATCHet seg.ucn + phasing)
##################################################


def read_bbc_phases(bbc_phases_file: str):
    """Load BBC-level bulk WGS phasing, chr-pos sorted.

    The PHASE column is 1=keep B, 0=swap A/B.
    """
    bbc_df = pd.read_table(bbc_phases_file, sep="\t")
    bbc_df = sort_df_chr(bbc_df, pos="START")
    logging.info(f"loaded bulk phases: {len(bbc_df)} BBC bins.")
    return bbc_df


def load_bulk_cnprofile(
    seg_ucn_file: str,
    solfile: str | None = None,
    baf_clip: float = 0.01,
):
    """Load bulk HATCHet seg.ucn CN profile. Returns ``(seg_df, clones,
    clone_props, cn_A, cn_B, cn_C, cn_BAF)``; keeps the first SAMPLE and applies
    ``solfile`` if given. ``cn_A/cn_B/cn_C/cn_BAF`` are per-segment per-clone copy
    numbers parsed from the CNP column.
    """
    seg_df, clones, clone_props = read_seg_ucn_file(seg_ucn_file)
    if "SAMPLE" in seg_df.columns:
        first_sample = seg_df["SAMPLE"].iloc[0]
        seg_df = seg_df[seg_df["SAMPLE"] == first_sample].reset_index(drop=True)
    if solfile is not None:
        seg_df, clones, clone_props = _apply_solfile(seg_df, solfile)
    _, cn_A, cn_B, cn_C, cn_BAF = parse_cnv_profile(seg_df, baf_clip=baf_clip)
    logging.info(f"loaded bulk CN profile: {len(seg_df)} segments, clones={clones}")
    return seg_df, clones, clone_props, cn_A, cn_B, cn_C, cn_BAF


def parse_cnv_profile(
    haplo_blocks: pd.DataFrame,
    baf_clip: float = 0.01,
):
    """Parse the per-segment CNP column into per-clone arrays.

    Returns ``(clones, cn_A, cn_B, cn_C, cn_BAF)`` where ``cn_*``/``cn_BAF`` are
    ``(num_segment, num_clone)``; cn_BAF is clipped to ``[baf_clip, 1 - baf_clip]``.
    """
    num_clones = len(str(haplo_blocks["CNP"].iloc[0]).split(";"))
    clones = ["normal"] + [f"clone{i}" for i in range(1, num_clones)]
    cn_A = np.zeros((len(haplo_blocks), num_clones), dtype=np.int32)
    cn_B = np.zeros((len(haplo_blocks), num_clones), dtype=np.int32)
    for i in range(num_clones):
        cn_A[:, i] = haplo_blocks.apply(
            func=lambda r: int(r["CNP"].split(";")[i].split("|")[0]), axis=1
        ).to_numpy()
        cn_B[:, i] = haplo_blocks.apply(
            func=lambda r: int(r["CNP"].split(";")[i].split("|")[1]), axis=1
        ).to_numpy()
    cn_C = cn_A + cn_B
    cn_BAF = np.divide(
        cn_B,
        cn_C,
        out=np.zeros_like(cn_C, dtype=np.float32),
        where=(cn_C > 0),
    )
    cn_BAF = np.clip(cn_BAF, baf_clip, 1 - baf_clip)
    return clones, cn_A, cn_B, cn_C, cn_BAF


def _apply_solfile(
    seg_df: pd.DataFrame,
    solfile: str,
):
    """Override cn_* and u_* columns in seg_df using a HATCHet solution file.

    Matches rows by CLUSTER ID and rebuilds CNP/PROPS columns.
    """
    sol_df = pd.read_table(solfile, sep="\t")
    if "SAMPLE" in sol_df.columns:
        sol_df = sol_df[sol_df["SAMPLE"] == sol_df["SAMPLE"].iloc[0]]

    cn_cols = [c for c in sol_df.columns if c.startswith("cn_")]
    u_cols = [c for c in sol_df.columns if c.startswith("u_")]
    clones = [c.replace("cn_", "") for c in cn_cols]

    logging.info(
        f"applying solfile: {solfile}, "
        f"{len(cn_cols)} clones ({clones}), {len(sol_df)} clusters"
    )

    # build cluster -> (cn, u) mapping
    sol_map = sol_df.set_index("CLUSTER")[cn_cols + u_cols].to_dict("index")

    # drop old cn_/u_ columns that don't match the solution's clone count
    old_cn = [c for c in seg_df.columns if c.startswith("cn_")]
    old_u = [c for c in seg_df.columns if c.startswith("u_")]
    seg_df = seg_df.drop(columns=old_cn + old_u)

    # add new columns from solution
    for col in cn_cols + u_cols:
        seg_df[col] = seg_df["CLUSTER"].map(
            lambda cid, c=col: sol_map.get(cid, {}).get(c)
        )

    # segments with clusters not in solfile → default diploid (1|1)
    unmapped = seg_df[cn_cols[0]].isna()
    if unmapped.any():
        logging.warning(
            f"{unmapped.sum()} segments have CLUSTER IDs not in solfile, "
            "defaulting to diploid 1|1"
        )
        for col in cn_cols:
            seg_df.loc[unmapped, col] = "1|1"
        for col in u_cols:
            seg_df.loc[unmapped, col] = 0.0

    # rebuild CNP and PROPS
    seg_df["CNP"] = seg_df.apply(
        lambda r: ";".join(str(r[f"cn_{c}"]) for c in clones), axis=1
    )
    clone_props = [float(seg_df[f"u_{c}"].iloc[0]) for c in clones]
    seg_df["PROPS"] = ";".join(str(p) for p in clone_props)
    return seg_df, clones, clone_props


##################################################
# spatial neighbor graphs
##################################################


def load_spatial_neighbors(h5ad_path: str, n_neighs: int = 6):
    """Load spatial coordinates from h5ad and compute per-rep neighbor graphs.

    Args:
        h5ad_path: path to h5ad file with spatial coordinates in obsm['spatial'].
        n_neighs: number of neighbors (default 6 for Visium hexagonal).

    Returns:
        dict[rep_id -> {"BARCODE": array, "coords": (N,2), "W": sparse}]
    """
    import scanpy as sc
    import squidpy as sq

    adata = sc.read_h5ad(h5ad_path)
    assert "spatial" in adata.obsm, f"no spatial coordinates in {h5ad_path}"

    # extract rep_id from barcode suffix (everything after first underscore)
    rep_ids = np.array([bc.split("_", 1)[1] for bc in adata.obs_names])
    result = {}
    for rep_id in np.unique(rep_ids):
        mask = rep_ids == rep_id
        adata_rep = adata[mask].copy()
        sq.gr.spatial_neighbors(adata_rep, n_neighs=n_neighs, coord_type="generic")
        W = adata_rep.obsp["spatial_connectivities"]
        row_sums = np.asarray(W.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        W = W.multiply(1.0 / row_sums[:, None])
        result[rep_id] = {
            "BARCODE": adata_rep.obs_names.to_numpy(),
            "coords": adata_rep.obsm["spatial"],
            "W": W,
        }
        logging.info(
            f"spatial neighbors rep={rep_id}: {mask.sum()} spots, "
            f"n_neighs={n_neighs}, edges={W.nnz}"
        )
    return result


def build_spatial_graphs(
    assay_types: list[str],
    h5ad_paths: dict[str, str | None],
    n_neighs: int,
):
    """Build per-modality spatial neighbor graphs from h5ad spatial coords.

    Skips assays without an h5ad or without ``obsm['spatial']``. Each value is
    a per-rep graph dict from ``load_spatial_neighbors``.
    """
    import scanpy as sc

    spatial_graphs = {}
    for assay_type in assay_types:
        h5ad_path = h5ad_paths[assay_type]
        if h5ad_path is not None:
            adata = sc.read_h5ad(h5ad_path)
            if "spatial" in adata.obsm:
                spatial_graphs[assay_type] = load_spatial_neighbors(
                    h5ad_path, n_neighs=n_neighs
                )
    return spatial_graphs
