import logging
import os

import numpy as np

from copytyping.copytyping_parser import check_arguments_inference
from copytyping.utils import normalize_args, add_file_logging
from copytyping.io_utils import read_cell_types
from copytyping.joint_inference.io import (
    load_single_cell_data,
    load_bulk_phases,
    load_bulk_cnp,
)
from copytyping.joint_inference.initialize import initialize_copytyping
from copytyping.joint_inference.segmentation import (
    adaptive_segmentation,
    perform_segmentation,
)
from copytyping.joint_inference.cnp_copytyping import cnp_copytyping
from copytyping.joint_inference.bulk_cnp_anchored_copytyping import (
    bulk_cnp_anchored_copytyping,
)
from copytyping.joint_inference.tmp_funcs import (
    plot_pseudobulk_baf,
    plot_clone_rdr_baf,
    build_cnp_df,
)


def run(args: dict | None = None) -> None:
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

    genome_coords_bbc, _, phase_bbc, switchprobs_bbc = load_bulk_phases(
        args["bbc_phases"]
    )

    (
        cna_int_states,
        rdr_baf_states,
        cna_profile_bbc,
        cna_mirrored_bbc,
        bulk_segmentation_bbc,
        clone_ids,
    ) = load_bulk_cnp(
        args["seg_ucn"],
        genome_coords_bbc,
        solfile=args["solfile"],
        baf_clip=args["baf_clip"],
        no_normal=args["no_normal"],
    )

    cell_type_df = read_cell_types(args["cell_type"], req_cols={"BARCODE", ref_label})
    barcodes_df, X_bbc, B_bbc, C_bbc = load_single_cell_data(
        args, data_types, cell_type_df=cell_type_df, celltype_col=ref_label
    )

    diploid_idx = int(
        np.where((cna_int_states[:, 0] == 1) & (cna_int_states[:, 1] == 1))[0][0]
    )
    phase_bbc[np.all(cna_profile_bbc == diploid_idx, axis=1)] = 1

    genome_coords, ind_mat = adaptive_segmentation(
        genome_coords_bbc,
        C_bbc,
        bulk_segmentation_bbc,
        min_snp_count=args["min_snp_count"],
        max_bin_length=args["max_bin_length"],
    )
    (
        X,
        B,
        C,
        cna_profile,
        cna_mirrored,
        phase,
        switchprobs,
    ) = perform_segmentation(
        ind_mat,
        X_bbc,
        B_bbc,
        C_bbc,
        cna_profile_bbc,
        cna_mirrored_bbc,
        phase_bbc,
        switchprobs_bbc,
    )

    plot_pseudobulk_baf(
        genome_coords,
        B,
        C,
        barcodes_df,
        region_bed=args["region_bed"],
        sample=sample,
        out_dir=out_dir,
        out_prefix=out_prefix,
    )

    T = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
    em_kwargs = dict(
        is_spot=(platform == "spatial"),
        fit_mode=args["fit_mode"],
        max_clones=args["max_clones"],
        n_bootstrap=args["n_bootstrap"],
        rng_seed=args["rng_seed"],
        anchored_tol=args["anchored_tol"],
        top_segments=args["top_segments"],
        niters=args["niters"],
        min_tau=args["min_tau"],
        max_tau=args["max_tau"],
        min_invphi=args["min_invphi"],
        max_invphi=args["max_invphi"],
        update_tau=args["update_tau"],
        update_invphi=args["update_invphi"],
        chunk_size=1000,
        tol=1e-4,
        eps=1e-12,
    )

    _, _, model_params = initialize_copytyping(
        X,
        B,
        C,
        T,
        cna_int_states,
        cna_profile,
        cna_mirrored,
        rdr_baf_states,
        clone_ids,
        em_kwargs=em_kwargs,
    )

    if args["method"] == "copytyping":
        labels, _resp = cnp_copytyping(
            X,
            B,
            C,
            T,
            cna_profile,
            cna_mirrored,
            rdr_baf_states,
            clone_ids,
            phase=phase,
            model_params=model_params,
            em_kwargs=em_kwargs,
        )
        pi = model_params["pi"]
        rdr_baf_params = model_params["rdr_baf_params"]
        base_props = model_params["base_props"]

    elif args["method"] == "bulk_anchored_copytyping":
        (
            labels,
            cna_profile,
            cna_mirrored,
            clone_ids,
            phase,
            parent_map,
        ) = bulk_cnp_anchored_copytyping(
            X,
            B,
            C,
            T,
            genome_coords,
            cna_int_states,
            cna_profile,
            cna_mirrored,
            rdr_baf_states,
            clone_ids,
            phase=phase,
            switchprobs=switchprobs,
            model_params=model_params,
            em_kwargs=em_kwargs,
            workdir=os.path.join(out_dir, "anchored_iters"),
            region_bed=args["region_bed"],
            sample=sample,
        )
        pi = model_params["pi"]
        rdr_baf_params = model_params["rdr_baf_params"]
        base_props = model_params["base_props"]

    cnp_df = build_cnp_df(
        genome_coords,
        cna_profile,
        cna_mirrored,
        cna_int_states,
        clone_names=clone_ids,
    )

    plot_clone_rdr_baf(
        genome_coords,
        X,
        B,
        C,
        T,
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
