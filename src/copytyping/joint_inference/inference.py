import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

from copytyping.copytyping_parser import check_arguments_inference
from copytyping.utils import normalize_args, add_file_logging
from copytyping.io_utils import read_cell_types
from copytyping.joint_inference.io import (
    load_single_cell_data,
    load_bulk_phases,
    load_bulk_cnp,
)
from copytyping.joint_inference.initialize import (
    adaptive_segmentation,
    perform_segmentation,
    initialize_bulk_cnp_copytyping,
    get_clone_norm,
)
from copytyping.joint_inference.optimize import block_coordinate_ascent_fixed_cnp
from copytyping.joint_inference.tmp_funcs import (
    plot_pseudobulk_baf,
    plot_clone_rdr_baf,
    get_masks_from_cna_profile,
    build_cnp_df,
)


def bulk_cnp_copytyping(
    X_seg,
    B_seg,
    C_seg,
    T_seg,
    cna_profile_seg,
    cna_mirrored_seg,
    rdr_baf_states,
    rdr_baf_params,
    base_props,
    masks,
    spot_purities=None,
    fit_mode="hybrid",
    clone_ids=None,
    em_kwargs=None,
):
    """Clone-EM with bulk CNP and phasing held fixed; init params (baseline,
    spot purity, dispersions, masks) come from
    ``initialize_bulk_cnp_copytyping``.

    ``clone_norm`` (the per-clone genome-wide RDR normalizer S_m) is computed
    here from the current ``cna_profile_seg`` so callers that grow the clone
    set between EM passes (e.g. ``bulk_anchored_copytyping``) always see a
    fresh value.

    ``fit_mode`` picks ``bb_mask = masks["IMBALANCED"]`` for ``hybrid /
    allele_only`` and ``nb_mask = masks["ANEUPLOID"]`` for ``hybrid /
    total_only``, then runs ``block_coordinate_ascent_fixed_cnp``. Returns ``(labels, pi, rdr_baf_params)``.
    """
    em_kwargs = em_kwargs or {}
    n_clones = cna_profile_seg.shape[1]
    clone_norm = get_clone_norm(base_props, cna_profile_seg, rdr_baf_states)

    bb_mask = masks["IMBALANCED"] if fit_mode in ("hybrid", "allele_only") else None
    nb_mask = masks["ANEUPLOID"] if fit_mode in ("hybrid", "total_only") else None
    bb_n = int(bb_mask.sum()) if bb_mask is not None else 0
    nb_n = int(nb_mask.sum()) if nb_mask is not None else 0
    logging.info(
        f"clone EM (fit_mode={fit_mode}, clones={clone_ids}): "
        f"BB on {bb_n} imbalanced, NB on {nb_n} aneuploid segs"
    )

    labels, pi, rdr_baf_params = block_coordinate_ascent_fixed_cnp(
        B_seg,
        C_seg,
        X_seg,
        T_seg,
        cna_profile_seg,
        cna_mirrored_seg,
        rdr_baf_states,
        rdr_baf_params,
        base_props=base_props,
        clone_norm=clone_norm,
        bb_mask=bb_mask,
        nb_mask=nb_mask,
        spot_purities=spot_purities,
        **em_kwargs,
    )

    logging.info(
        f"final: pi=[{', '.join(f'{p:.3f}' for p in pi)}], "
        f"labels={np.bincount(labels, minlength=n_clones).tolist()}, clone_ids={clone_ids}"
    )
    return labels, pi, rdr_baf_params


def run(args=None):
    logging.info("run copytyping joint inference")
    args = normalize_args(args)
    args = check_arguments_inference(args)
    sample = args["sample"]
    platform = args["platform"]
    data_types = args["data_types"]
    ref_label = args["ref_label"]
    out_dir = args["out_dir"]
    out_prefix = args["out_prefix"] or str(sample)
    os.makedirs(out_dir, exist_ok=True)
    _file_handler = add_file_logging(out_dir)
    proc_dir = os.path.join(out_dir, "processed_data")
    os.makedirs(proc_dir, exist_ok=True)

    logging.info(f"sample={sample}, platform={platform}, data_types={data_types}")

    # load bulk data
    coord_df_bbc, phase_post_bbc, phase_map_bbc, switchprobs_bbc = load_bulk_phases(
        args["bbc_phases"]
    )

    (
        cna_int_states,
        rdr_baf_states,
        cna_profile,
        cna_mirrored,
        bulk_segmentation,
        clone_ids,
    ) = load_bulk_cnp(
        args["seg_ucn"],
        coord_df_bbc,
        solfile=args["solfile"],
        baf_clip=args["baf_clip"],
    )

    # load single-cell data
    cell_type_df = read_cell_types(args["cell_type"], req_cols={"BARCODE", ref_label})
    barcodes_df, X_bbc, B_bbc, C_bbc = load_single_cell_data(
        args, data_types, cell_type_df=cell_type_df, celltype_col=ref_label
    )

    # step 4. adaptive binning
    # bulk_phased: True if any clone is not 1|1 (phasing matters for BAF)
    diploid_idx = int(
        np.where((cna_int_states[:, 0] == 1) & (cna_int_states[:, 1] == 1))[0][0]
    )
    bulk_phased = ~np.all(cna_profile == diploid_idx, axis=1)
    coord_df_seg, ind_mat_seg = adaptive_segmentation(
        coord_df_bbc,
        C_bbc,
        bulk_segmentation,
        bulk_phased=bulk_phased,
        min_snp_count=args["min_snp_count"],
        max_bin_length=args["max_bin_length"],
    )
    X_seg, B_seg, C_seg, cna_profile_seg, cna_mirrored_seg, bulk_phased_seg = (
        perform_segmentation(
            ind_mat_seg,
            X_bbc,
            B_bbc,
            C_bbc,
            cna_profile,
            cna_mirrored,
            bulk_phased,
            phase_map_bbc,
        )
    )

    # step 5. pseudobulk BAF plots after segmentation
    plot_pseudobulk_baf(
        coord_df_seg,
        B_seg,
        C_seg,
        barcodes_df,
        region_bed=args["region_bed"],
        sample=sample,
        out_dir=out_dir,
        out_prefix=out_prefix,
    )

    if args["no_normal"]:
        assert clone_ids[0] == "normal", clone_ids
        clone_ids = clone_ids[1:]
        cna_profile_seg = cna_profile_seg[:, 1:]
        cna_mirrored_seg = cna_mirrored_seg[:, 1:]
        logging.info(f"--no_normal: dropped normal clone -> {clone_ids}")

    # bin-level CN-state masks (IMBALANCED / ANEUPLOID / CLONAL_IMBALANCED /
    # SUBCLONAL); computed once over the final cna_profile_seg / cna_mirrored_seg
    # and threaded through init + clone EM + the labels_init debug plot so they
    # share the same partition.
    masks = get_masks_from_cna_profile(
        cna_int_states, cna_profile_seg, cna_mirrored_seg
    )

    # step 6. initialize dispersion parameters per CN state
    # rdr_baf_params: (S, 2) array — (invphi, tau) per state
    n_states = len(rdr_baf_states)
    rdr_baf_params = np.column_stack(
        [
            np.full(n_states, args["max_invphi"]),
            np.full(n_states, args["max_tau"]),
        ]
    )

    # step 7. classify cells/spots into clones
    T_seg = np.asarray(X_seg.sum(axis=0)).ravel().astype(np.float64)
    is_spot = platform == "spatial"

    if args["method"] == "copytyping":
        em_kwargs = dict(
            niters=args["niters"],
            tol=1e-4,
            min_tau=args["min_tau"],
            max_tau=args["max_tau"],
            min_invphi=args["min_invphi"],
            max_invphi=args["max_invphi"],
            update_tau=args["update_tau"],
            update_invphi=args["update_invphi"],
            chunk_size=1000,
        )

        # 1) initialization: baseline, spot purity, normal-cell labels
        init_params = initialize_bulk_cnp_copytyping(
            X_seg,
            B_seg,
            C_seg,
            T_seg,
            cna_profile_seg,
            cna_mirrored_seg,
            rdr_baf_states,
            rdr_baf_params,
            masks,
            clone_ids,
            is_spot=is_spot,
            em_kwargs=em_kwargs,
        )
        base_props = init_params["base_props"]
        spot_purities = init_params["spot_purities"]

        # 2) clone-EM
        labels, pi, rdr_baf_params = bulk_cnp_copytyping(
            X_seg,
            B_seg,
            C_seg,
            T_seg,
            cna_profile_seg,
            cna_mirrored_seg,
            rdr_baf_states,
            rdr_baf_params=init_params["rdr_baf_params"],
            base_props=base_props,
            masks=masks,
            spot_purities=spot_purities,
            fit_mode=args["fit_mode"],
            clone_ids=clone_ids,
            em_kwargs=em_kwargs,
        )

    # ASCN block table (one row per contiguous same-CNP block) used by the
    # plotter for both dot colors / expected lines (per-seg via interval lookup)
    # and the bottom merged CN profile panel.
    cnp_df = build_cnp_df(
        coord_df_seg,
        cna_profile_seg,
        cna_mirrored_seg,
        cna_int_states,
        clone_names=clone_ids,
    )

    plot_clone_rdr_baf(
        coord_df_seg,
        X_seg,
        B_seg,
        C_seg,
        T_seg,
        labels,
        base_props,
        cnp_df,
        barcodes_df,
        region_bed=args["region_bed"],
        sample=sample,
        out_dir=out_dir,
        out_prefix=out_prefix,
        clone_names=clone_ids,
    )

    logging.info(f"joint inference complete. outputs in {out_dir}")
    logging.root.removeHandler(_file_handler)
    _file_handler.close()
