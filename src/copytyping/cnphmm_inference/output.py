"""Writers for CNP-HMM results. Four artifacts:

  * ``<prefix>.cnphmm.cell_cn_states.npz`` — cell x segment CN-state index matrix
  * ``<prefix>.cnphmm.cn_states.tsv``      — per-state table (A|B, rdr, baf, dispersions)
  * ``<prefix>.cnphmm.cells.tsv``          — per-cell barcode, unique-path id, clone, posterior
  * ``<prefix>.cnphmm.parameters.npz``     — all learned parameters + per-segment info

The per-cell ``clone`` is a per-REP flat clustering of the decoded paths over the
CNT distance (see :func:`cnt_clustering.assign_clones`).
"""

import logging
import os
from typing import Any

import numpy as np
import pandas as pd


def write_outputs(
    result: dict[str, Any],
    genome_coords: pd.DataFrame,
    cn_states: np.ndarray,
    rdr_baf_cn: np.ndarray,
    rdr_baf_params: np.ndarray,
    base_props: np.ndarray,
    alpha: np.ndarray,
    barcodes_df: pd.DataFrame,
    clone_labels: np.ndarray,
    out_dir: str,
    out_prefix: str,
) -> None:
    """Write the four CNP-HMM artifacts and log the number of unique decoded
    CN paths and per-REP clone clusters. ``clone_labels`` is the ``(N,)`` per-cell
    clone id (1-based within a REP, ``-1`` if unclustered).
    """
    Z = result["Z"]  # (N, G) masked state indices
    H = np.asarray(result["H"])  # (G,) shared phasing path
    N, G = Z.shape

    unique_paths, inverse = np.unique(Z, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n_unique = unique_paths.shape[0]
    logging.info(f"CNP-HMM: {n_unique} unique decoded CN paths (CNPs) over {N} cells")

    prefix = os.path.join(out_dir, f"{out_prefix}.cnphmm")

    # 1. cell x segment CN-state index matrix (int16: state count is small)
    np.savez_compressed(f"{prefix}.cell_cn_states.npz", Z=Z.astype(np.int16))

    # 2. per-state table: index -> (A, B), rdr, baf, learned dispersions
    pd.DataFrame(
        {
            "state_idx": np.arange(cn_states.shape[0]),
            "A": cn_states[:, 0],
            "B": cn_states[:, 1],
            "rdr": rdr_baf_cn[:, 0],
            "baf": rdr_baf_cn[:, 1],
            "invphi": rdr_baf_params[:, 0],  # learned NB dispersion
            "tau": rdr_baf_params[:, 1],  # learned BB dispersion
        }
    ).to_csv(f"{prefix}.cn_states.tsv", sep="\t", index=False)

    # 3. per-cell table: barcode, unique-path id, per-REP clone, decode confidence
    cells = barcodes_df[["BARCODE", "REP_ID"]].copy().reset_index(drop=True)
    cells["unique_cnp_id"] = inverse.astype(np.int32)
    cells["clone"] = np.asarray(clone_labels).astype(np.int32)
    if result.get("posterior") is not None:
        cells["posterior"] = np.asarray(result["posterior"])
    cells.to_csv(f"{prefix}.cells.tsv", sep="\t", index=False)
    clustered = np.asarray(clone_labels) >= 0
    n_groups = cells.loc[clustered, ["REP_ID", "clone"]].drop_duplicates().shape[0]
    logging.info(
        f"CNP-HMM: {n_groups} per-REP clone clusters "
        f"({int((~clustered).sum())} cells unclustered)"
    )

    # 4. all learned parameters + per-segment coordinates / phasing / diagnostics
    np.savez_compressed(
        f"{prefix}.parameters.npz",
        pi=result["pi"],
        A=result["A"],
        Xi=result["Xi"],
        alpha=alpha,
        rdr_baf_params=rdr_baf_params,
        base_props=base_props,
        cn_states=cn_states,
        H=H,
        seg_chrom=genome_coords["#CHR"].to_numpy().astype(str),
        seg_start=genome_coords["START"].to_numpy(),
        seg_end=genome_coords["END"].to_numpy(),
        ll_hist=np.asarray(result["ll_hist"], dtype=np.float64),
        entropy_hist=np.asarray(result["entropy_hist"], dtype=np.float64),
        seg_entropy=np.asarray(result["seg_entropy"], dtype=np.float64),
        pi_hist=result["pi_hist"],
        tau_hist=result["tau_hist"],
        invphi_hist=result["invphi_hist"],
        phase_frac_hist=result["phase_frac_hist"],
    )
    logging.info(f"CNP-HMM: wrote outputs to {prefix}.*")


def write_nj_trees(trees: dict[str, str], out_dir: str, out_prefix: str) -> None:
    """Write the per-REP CNT cluster trees as Newick. ``trees`` maps REP_ID ->
    Newick (one per rep where a tree was built; method = cnt_nj / cnt_upgma /
    cnt_complete); each line is prefixed with the rep as a Newick comment. A no-op
    (logged) when no tree was built (too many unique profiles)."""
    prefix = os.path.join(out_dir, f"{out_prefix}.cnphmm")
    if not trees:
        logging.info(
            "CNP-HMM: no cluster tree written (too many unique profiles for CNT)"
        )
        return
    path = f"{prefix}.cluster_tree.nwk"
    with open(path, "w") as fh:
        for rep, nwk in trees.items():
            fh.write(f"[{rep}] {nwk}\n")
    logging.info(f"CNP-HMM: wrote {len(trees)} cluster tree(s) to {path}")
