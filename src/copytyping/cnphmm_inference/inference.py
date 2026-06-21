"""Factorial CNP-HMM copy-typing: orchestration and the two inference engines.

Each cell gets its own latent CN path through a state space ``C`` of allele-
specific ``(a, b)`` states, under segment-dependent transition matrices with a
bulk-CNP-informed Dirichlet prior and a shared phasing path ``H``.

The ``soft_em`` method splits training from decoding: ``cnp_hmm_baum_welch`` runs
batched forward-backward to train ``Theta`` (soft posteriors, MAP-decoded ``H``),
then ``cnp_hmm_map_decoding`` (posterior max-marginal) or ``cnp_hmm_viterbi``
(joint-MAP) decodes the per-cell paths. ``cnp_hmm_block_ascent`` runs
block-coordinate Viterbi over ``(Z, H)`` (training and decoding fused). Data
loading (``data_io``), segmentation (``segmentation``), and reference-cell
seeding (``initialize``) live in sibling modules.
"""

import logging
import os
from typing import Any

import numpy as np
from scipy import sparse

from copytyping.copytyping_parser import check_arguments_cnphmm
from copytyping.utils import normalize_args, add_file_logging
from copytyping.io_utils import read_cell_types
from copytyping.cnphmm_inference.data_io import (
    load_single_cell_data,
    load_bulk_phases,
    load_bulk_cnp,
)
from copytyping.cnphmm_inference.initialize import initialize_copytyping
from copytyping.cnphmm_inference.segmentation import (
    adaptive_segmentation,
    perform_segmentation,
)
from copytyping.cnphmm_inference.states import (
    build_state_space,
    build_transition_prior,
    prior_mean_transitions,
)
from copytyping.cnphmm_inference.emissions import build_log_emissions
from copytyping.cnphmm_inference.forward_backward import estep, accumulate_eH_hard
from copytyping.cnphmm_inference.viterbi import viterbi_Z, viterbi_H
from copytyping.cnphmm_inference.optimize import (
    update_transitions,
    update_pi,
    transition_logprior,
    transition_entropy,
    update_dispersions_hard,
)
from copytyping.cnphmm_inference.cnt_clustering import assign_clones
from copytyping.cnphmm_inference.output import write_outputs, write_nj_trees
from copytyping.cnphmm_inference.plots import (
    compute_cell_order,
    plot_cell_seg_heatmaps,
    plot_transition_trellis,
    plot_transition_entropy,
    plot_training_diagnostics,
)


class _ParamHistory:
    """Accumulates per-iteration parameter snapshots for training diagnostics:
    ``pi`` (K,), per-state NB ``invphi`` / BB ``tau`` (K,), and the phasing
    fraction (share of segments kept in the bulk orientation, ``H == 1``). Each
    :meth:`record` appends the values *used* in that iteration; :meth:`asdict`
    returns stacked arrays ``(n_iter, K)`` / ``(n_iter,)`` for plotting."""

    def __init__(self) -> None:
        self.pi: list[np.ndarray] = []
        self.invphi: list[np.ndarray] = []
        self.tau: list[np.ndarray] = []
        self.phase_frac: list[float] = []

    def record(self, pi: np.ndarray, rdr_baf_params: np.ndarray, H: np.ndarray) -> None:
        self.pi.append(np.asarray(pi).copy())
        self.invphi.append(rdr_baf_params[:, 0].copy())
        self.tau.append(rdr_baf_params[:, 1].copy())
        self.phase_frac.append(float(np.mean(np.asarray(H) == 1)))

    def asdict(self) -> dict[str, np.ndarray]:
        return {
            "pi_hist": np.array(self.pi, dtype=np.float64),
            "invphi_hist": np.array(self.invphi, dtype=np.float64),
            "tau_hist": np.array(self.tau, dtype=np.float64),
            "phase_frac_hist": np.array(self.phase_frac, dtype=np.float64),
        }


def cnp_hmm_baum_welch(
    X: sparse.csr_matrix,
    B: sparse.csr_matrix,
    C: sparse.csr_matrix,
    T: np.ndarray,
    rdr_baf_cn: np.ndarray,
    rdr_baf_params: np.ndarray,
    base_props: np.ndarray,
    alpha: np.ndarray,
    pi: np.ndarray,
    H: np.ndarray,
    switchprobs: np.ndarray,
    chrom_break: np.ndarray,
    chrom_start: np.ndarray,
    em_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Baum-Welch / soft EM **training** of ``Theta = {pi, A, E}`` with MAP-decoded
    phasing ``H``. TRAINING ONLY — it does not decode per-cell paths; it returns
    the trained parameters and values needed for decoding (``pi``, ``A``, ``H``;
    the dispersions are trained in place in ``rdr_baf_params``) plus training
    diagnostics. Decode the per-cell CNPs afterwards with
    :func:`cnp_hmm_map_decoding` (posterior) or :func:`cnp_hmm_viterbi` (joint-MAP).

    Inner loop per iteration: streamed forward-backward E-step -> MAP ``H`` update
    -> M-step (``pi``, MAP ``A``, optional dispersions). Converges on the
    complete-data MAP objective ``total_ll + log P(A)``.

    Chromosomes are independent chains: ``chrom_break`` (G-1,) marks transitions
    that cross a chromosome boundary; those transition matrices are set to
    ``pi``-broadcast rows. ``chrom_start`` (G,) marks each chromosome's first
    segment, over which ``pi`` is pooled.
    """
    G, N = X.shape
    niters, tol = em_kwargs["niters"], em_kwargs["tol"]
    chunk_size, eps = em_kwargs["chunk_size"], em_kwargs["eps"]
    fit_mode, pi_alpha = em_kwargs["fit_mode"], em_kwargs["pi_alpha"]
    keep = ~chrom_break  # within-chromosome transitions

    A = prior_mean_transitions(alpha)
    A[chrom_break] = pi
    log_pi = np.log(np.clip(pi, eps, None))
    Z = np.zeros((N, G), dtype=np.int32)  # scratch path for the dispersion M-step
    Xi = np.zeros((G - 1, A.shape[1], A.shape[2]), dtype=np.float64)
    ll_hist: list[float] = []
    entropy_hist: list[float] = []
    param_hist = _ParamHistory()  # per-iter pi / dispersions / phasing trajectories
    prev_obj = -np.inf

    for it in range(niters):
        logA = np.log(np.clip(A, eps, None))
        pi_acc, Xi, total_ll, eH = estep(
            X,
            B,
            C,
            T,
            rdr_baf_cn,
            rdr_baf_params,
            base_props,
            H,
            log_pi,
            logA,
            fit_mode,
            chunk_size,
            eps=eps,
            Z_out=Z,
            chrom_start=chrom_start,
        )
        obj = total_ll + transition_logprior(A[keep], alpha[keep])
        ll_hist.append(obj)
        _, ent = transition_entropy(A[keep])
        entropy_hist.append(ent)
        param_hist.record(pi, rdr_baf_params, H)  # state used in this iteration
        logging.info(
            f"  baum_welch iter {it}: obj={obj:.1f} (ll={total_ll:.1f}), "
            f"mean transition entropy={ent:.4f}"
        )
        if it > 0 and abs(obj - prev_obj) / (abs(prev_obj) + 1e-10) < tol:
            logging.info(f"  baum_welch converged at iter {it}")
            break
        prev_obj = obj

        pi = update_pi(pi_acc, pi_alpha)
        log_pi = np.log(np.clip(pi, eps, None))
        A = update_transitions(Xi, alpha)
        A[chrom_break] = pi
        if em_kwargs["update_tau"] or em_kwargs["update_invphi"]:
            update_dispersions_hard(
                Z, X, B, C, T, rdr_baf_cn, rdr_baf_params, base_props, H, em_kwargs
            )
        H = viterbi_H(eH, switchprobs, eps)

    seg_entropy, _ = transition_entropy(A, Xi)  # occupancy-weighted by converged Xi
    seg_entropy[chrom_break] = np.nan  # boundary transitions are not real
    return {
        "pi": pi,
        "A": A,
        "H": H,
        "Xi": Xi,
        "ll_hist": ll_hist,
        "entropy_hist": entropy_hist,
        "seg_entropy": seg_entropy,
        "obj_label": "MAP objective (log-lik + log prior)",
        **param_hist.asdict(),
    }


def cnp_hmm_map_decoding(
    X: sparse.csr_matrix,
    B: sparse.csr_matrix,
    C: sparse.csr_matrix,
    T: np.ndarray,
    rdr_baf_cn: np.ndarray,
    rdr_baf_params: np.ndarray,
    base_props: np.ndarray,
    H: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    chrom_start: np.ndarray,
    em_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Posterior (maximum-marginal) decoding of per-cell CNPs under trained
    parameters: ``Z_{i,g} = argmax_c gamma_{i,g}(c)`` from the forward-backward
    posterior. Returns ``{Z (N, G), posterior (N,)}`` where ``posterior`` is each
    cell's mean max-marginal (a [0, 1] decode-confidence)."""
    G, N = X.shape
    eps, chunk_size = em_kwargs["eps"], em_kwargs["chunk_size"]
    fit_mode = em_kwargs["fit_mode"]
    log_pi = np.log(np.clip(pi, eps, None))
    logA = np.log(np.clip(A, eps, None))
    Z = np.zeros((N, G), dtype=np.int32)
    conf = np.empty(N, dtype=np.float64)
    estep(
        X,
        B,
        C,
        T,
        rdr_baf_cn,
        rdr_baf_params,
        base_props,
        H,
        log_pi,
        logA,
        fit_mode,
        chunk_size,
        eps=eps,
        Z_out=Z,
        conf_out=conf,
        chrom_start=chrom_start,
    )
    logging.info("cnp_hmm_map_decoding: posterior (max-marginal) decode complete")
    return {"Z": Z, "posterior": conf}


def cnp_hmm_viterbi(
    X: sparse.csr_matrix,
    B: sparse.csr_matrix,
    C: sparse.csr_matrix,
    T: np.ndarray,
    rdr_baf_cn: np.ndarray,
    rdr_baf_params: np.ndarray,
    base_props: np.ndarray,
    H: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
    chrom_start: np.ndarray,
    em_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Viterbi decoding of per-cell CNPs under trained parameters: the joint-MAP
    path ``argmax_{Z_i} P(Z_i | D, Theta)`` via the Viterbi DP (chromosome breaks
    are encoded in ``A``'s pi-broadcast boundary rows). Returns ``{Z (N, G),
    posterior: None}`` (no marginal posterior under Viterbi)."""
    G, N = X.shape
    eps, chunk_size = em_kwargs["eps"], em_kwargs["chunk_size"]
    fit_mode = em_kwargs["fit_mode"]
    log_pi = np.log(np.clip(pi, eps, None))
    logA = np.log(np.clip(A, eps, None))
    Xc, Bc, Cc = X.tocsc(), B.tocsc(), C.tocsc()
    Z = np.zeros((N, G), dtype=np.int32)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        log_e = build_log_emissions(
            Xc[:, start:end].tocsr(),
            Bc[:, start:end].tocsr(),
            Cc[:, start:end].tocsr(),
            T[start:end],
            rdr_baf_cn,
            rdr_baf_params,
            base_props,
            H,
            fit_mode,
            eps=eps,
        )
        Z[start:end] = viterbi_Z(log_e, log_pi, logA)
    logging.info("cnp_hmm_viterbi: joint-MAP (Viterbi) decode complete")
    return {"Z": Z, "posterior": None}


def cnp_hmm_block_ascent(
    X: sparse.csr_matrix,
    B: sparse.csr_matrix,
    C: sparse.csr_matrix,
    T: np.ndarray,
    rdr_baf_cn: np.ndarray,
    rdr_baf_params: np.ndarray,
    base_props: np.ndarray,
    alpha: np.ndarray,
    pi: np.ndarray,
    H: np.ndarray,
    switchprobs: np.ndarray,
    chrom_break: np.ndarray,
    chrom_start: np.ndarray,
    em_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Block-coordinate ascent on the complete-data MAP objective.

    Per iteration: per-cell ``Z`` Viterbi given ``H`` (streamed over cell
    chunks, accumulating the hard transition counts and phasing emission) ->
    2-state ``H`` Viterbi given ``Z`` -> hard-count M-step (``pi``, MAP ``A``,
    optional dispersions). Stops when the decoded ``Z`` is unchanged.

    Chromosome boundaries break the chain: boundary transitions are fixed to
    ``pi``-broadcast rows and ``pi`` is pooled over the first segment of every
    chromosome (``chrom_start``).
    """
    G, N = X.shape
    K = rdr_baf_cn.shape[0]
    niters, tol = em_kwargs["niters"], em_kwargs["tol"]
    chunk_size, eps = em_kwargs["chunk_size"], em_kwargs["eps"]
    fit_mode, pi_alpha = em_kwargs["fit_mode"], em_kwargs["pi_alpha"]
    keep = ~chrom_break

    A = prior_mean_transitions(alpha)
    A[chrom_break] = pi
    log_pi = np.log(np.clip(pi, eps, None))
    Z = np.zeros((N, G), dtype=np.int32)
    Xc, Bc, Cc = X.tocsc(), B.tocsc(), C.tocsc()
    ll_hist: list[float] = []
    entropy_hist: list[float] = []
    param_hist = _ParamHistory()
    prev_Z = None
    Xi = np.zeros((G - 1, K, K), dtype=np.float64)

    for it in range(niters):
        logA = np.log(np.clip(A, eps, None))
        Xi = np.zeros((G - 1, K, K), dtype=np.float64)
        pi_acc = np.zeros(K, dtype=np.float64)
        eH = np.zeros((G, 2), dtype=np.float64)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            X_sub = Xc[:, start:end].tocsr()
            B_sub = Bc[:, start:end].tocsr()
            C_sub = Cc[:, start:end].tocsr()
            log_e, bb_pack = build_log_emissions(
                X_sub,
                B_sub,
                C_sub,
                T[start:end],
                rdr_baf_cn,
                rdr_baf_params,
                base_props,
                H,
                fit_mode,
                eps=eps,
                return_bb_orient=True,
            )
            Z_chunk = viterbi_Z(log_e, log_pi, logA)
            Z[start:end] = Z_chunk
            pi_acc += np.bincount(Z_chunk[:, chrom_start].ravel(), minlength=K)
            # vectorized transition counts: flat bin = g*K*K + from*K + to, summed
            # over cells (replaces a per-segment Python loop of np.add.at)
            flat = (
                (np.arange(G - 1) * K * K)[None, :]
                + Z_chunk[:, :-1] * K
                + Z_chunk[:, 1:]
            )
            Xi += np.bincount(flat.ravel(), minlength=(G - 1) * K * K).reshape(
                G - 1, K, K
            )
            if bb_pack is not None:
                accumulate_eH_hard(eH, Z_chunk, bb_pack)

        n_changed = N * G if prev_Z is None else int((Z != prev_Z).sum())
        ll_hist.append(float(n_changed))
        _, ent = transition_entropy(A[keep])
        entropy_hist.append(ent)
        param_hist.record(pi, rdr_baf_params, H)  # state used in this iteration
        logging.info(
            f"  block_ascent iter {it}: {n_changed}/{N * G} states changed, "
            f"mean transition entropy={ent:.4f}"
        )
        if prev_Z is not None and n_changed <= tol * N * G:
            logging.info(f"  block_ascent converged at iter {it}")
            break
        prev_Z = Z.copy()

        pi = update_pi(pi_acc, pi_alpha)
        log_pi = np.log(np.clip(pi, eps, None))
        A = update_transitions(Xi, alpha)
        A[chrom_break] = pi
        if em_kwargs["update_tau"] or em_kwargs["update_invphi"]:
            update_dispersions_hard(
                Z, X, B, C, T, rdr_baf_cn, rdr_baf_params, base_props, H, em_kwargs
            )
        H = viterbi_H(eH, switchprobs, eps)

    seg_entropy, _ = transition_entropy(A, Xi)
    seg_entropy[chrom_break] = np.nan
    return {
        "Z": Z,
        "H": H,
        "pi": pi,
        "A": A,
        "Xi": Xi,
        "posterior": None,  # no per-cell marginal posterior under hard decoding
        "ll_hist": ll_hist,
        "entropy_hist": entropy_hist,
        "seg_entropy": seg_entropy,
        "obj_label": "# decoded states changed",
        **param_hist.asdict(),
    }


def run(args: dict | None = None) -> None:
    logging.info("run copytyping CNP-HMM inference")
    args = normalize_args(args)
    args = check_arguments_cnphmm(args)
    sample = args["sample"]
    platform = args["platform"]
    assay_types = args["assay_types"]
    ref_label = args["ref_label"]
    out_dir = args["out_dir"]
    out_prefix = args["out_prefix"] or str(sample)
    os.makedirs(out_dir, exist_ok=True)
    _file_handler = add_file_logging(out_dir, command="cnphmm_copytyping")

    logging.info(f"sample={sample}, platform={platform}, assay_types={assay_types}")

    genome_coords_bbc, _, phase_bbc, switchprobs_bbc = load_bulk_phases(
        args["bbc_phases"]
    )
    (
        cna_int_states,
        rdr_baf_states,
        cna_profile_bbc,
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
        args, assay_types, cell_type_df=cell_type_df, celltype_col=ref_label
    )

    # diploid segments carry no allelic signal — pin them to the bulk orientation
    diploid = np.where((cna_int_states[:, 0] == 1) & (cna_int_states[:, 1] == 1))[0]
    if diploid.size > 0:
        phase_bbc[np.all(cna_profile_bbc == int(diploid[0]), axis=1)] = 1

    genome_coords, ind_mat = adaptive_segmentation(
        genome_coords_bbc,
        C_bbc,
        bulk_segmentation_bbc,
        min_snp_count=args["min_snp_count"],
        max_bin_length=args["max_bin_length"],
    )
    X, B, C, cna_profile_seg, phase, switchprobs = perform_segmentation(
        ind_mat, X_bbc, B_bbc, C_bbc, cna_profile_bbc, phase_bbc, switchprobs_bbc
    )
    T = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)

    em_kwargs = dict(
        is_spot=(platform == "spatial"),
        fit_mode=args["fit_mode"],
        niters=args["niters"],
        tol=1e-4,
        chunk_size=1000,
        eps=1e-12,
        pi_alpha=args["pi_alpha"],
        min_tau=args["min_tau"],
        max_tau=args["max_tau"],
        min_invphi=args["min_invphi"],
        max_invphi=args["max_invphi"],
        update_tau=not args["fix_dispersion"],
        update_invphi=not args["fix_dispersion"],
    )

    # init base_props / per-state dispersions via the bulk-clone reference EM
    _, _, model_params = initialize_copytyping(
        X,
        B,
        C,
        T,
        cna_int_states,
        cna_profile_seg,
        phase,
        rdr_baf_states,
        clone_ids,
        em_kwargs=em_kwargs,
    )
    base_props = model_params["base_props"]

    # masked CN-state space + remap bulk-indexed arrays into it
    cn_states, rdr_baf_cn, bulk_to_masked, cna_profile_masked = build_state_space(
        args["c_max"],
        args["mask_mode"],
        cna_int_states,
        cna_profile_seg,
        args["baf_clip"],
    )
    K = cn_states.shape[0]
    rdr_baf_params = np.column_stack(
        [np.full(K, args["max_invphi"]), np.full(K, args["max_tau"])]
    )
    valid = bulk_to_masked >= 0
    rdr_baf_params[bulk_to_masked[valid]] = model_params["rdr_baf_params"][valid]

    alpha = build_transition_prior(
        cna_profile_masked,
        K,
        args["prior_s"],
        args["prior_omega"],
        args["prior_t"],
        args["prior_eps"],
    )
    seg0 = cna_profile_masked[0]
    seg0 = seg0[seg0 >= 0]  # skip clones in a dropped (out-of-grid) state
    pi = np.bincount(seg0, minlength=K).astype(np.float64)
    pi += args["pi_alpha"]
    pi /= pi.sum()
    H = np.ones(X.shape[0], dtype=np.int8)

    # chromosomes are independent chains: mark transitions that cross a boundary
    chrom = genome_coords["#CHR"].to_numpy()
    chrom_break = chrom[1:] != chrom[:-1]  # (G-1,)
    chrom_start = np.r_[True, chrom_break]  # (G,) first segment of each chromosome

    logging.info(
        f"CNP-HMM: N={X.shape[1]} cells, G={X.shape[0]} segments, K={K} states, "
        f"{int(chrom_break.sum()) + 1} chromosome chains, "
        f"method={args['cnphmm_method']}"
    )
    plot_transition_trellis(
        genome_coords,
        prior_mean_transitions(alpha),
        cn_states,
        region_bed=args["region_bed"],
        sample=sample,
        out_dir=out_dir,
        out_prefix=out_prefix,
        dpi=args["dpi"],
        out_name="transition_prior",
        title=f"{sample} CN-state transition prior (pre-EM)",
    )
    if args["cnphmm_method"] == "baum_welch":
        # Baum-Welch trains Theta + MAP H; a separate pass decodes per-cell CNPs.
        trained = cnp_hmm_baum_welch(
            X,
            B,
            C,
            T,
            rdr_baf_cn,
            rdr_baf_params,
            base_props,
            alpha,
            pi,
            H,
            switchprobs,
            chrom_break,
            chrom_start,
            em_kwargs,
        )
        decoder = (
            cnp_hmm_viterbi if args["decode"] == "viterbi" else cnp_hmm_map_decoding
        )
        dec = decoder(
            X,
            B,
            C,
            T,
            rdr_baf_cn,
            rdr_baf_params,
            base_props,
            trained["H"],
            trained["pi"],
            trained["A"],
            chrom_start,
            em_kwargs,
        )
        result = {**trained, **dec}
    else:
        result = cnp_hmm_block_ascent(
            X,
            B,
            C,
            T,
            rdr_baf_cn,
            rdr_baf_params,
            base_props,
            alpha,
            pi,
            H,
            switchprobs,
            chrom_break,
            chrom_start,
            em_kwargs,
        )

    # per-REP flat clone assignment by cutting the CNT-distance hierarchy
    clone_labels = assign_clones(
        result["Z"],
        barcodes_df["REP_ID"].to_numpy(),
        cn_states,
        genome_coords["#CHR"].to_numpy(),
        args["n_clones"],
    )
    write_outputs(
        result,
        genome_coords,
        cn_states,
        rdr_baf_cn,
        rdr_baf_params,
        base_props,
        alpha,
        barcodes_df,
        clone_labels,
        out_dir,
        out_prefix,
    )
    # shared row order: partition by REP_ID, then cluster decoded paths within rep
    cell_order, cell_labels, nj_trees, dendro_polys = compute_cell_order(
        result["Z"],
        barcodes_df["REP_ID"].to_numpy(),
        cn_states,
        genome_coords["#CHR"].to_numpy(),
        method=args["cluster_method"],
    )
    write_nj_trees(nj_trees, out_dir, out_prefix)
    seg_info = (
        f"min-snp-reads={args['min_snp_count'] / 1000:g}k, "
        f"max-block-length={args['max_bin_length'] / 1e6:g}Mbp"
    )
    plot_cell_seg_heatmaps(
        genome_coords,
        B,
        C,
        X,
        T,
        result["H"],
        base_props,
        result["Z"],
        cn_states,
        cell_order,
        cell_labels,
        region_bed=args["region_bed"],
        sample=sample,
        out_dir=out_dir,
        out_prefix=out_prefix,
        dpi=args["dpi"],
        dendro_polys=dendro_polys,
        c_max=args["c_max"],
        seg_info=seg_info,
    )
    plot_transition_trellis(
        genome_coords,
        result["A"],
        cn_states,
        region_bed=args["region_bed"],
        sample=sample,
        out_dir=out_dir,
        out_prefix=out_prefix,
        dpi=args["dpi"],
    )
    seg_ent_prior, _ = transition_entropy(prior_mean_transitions(alpha), result["Xi"])
    seg_ent_prior[chrom_break] = np.nan
    plot_transition_entropy(
        genome_coords,
        seg_ent_prior,
        result["seg_entropy"],
        np.asarray(result["entropy_hist"]),
        region_bed=args["region_bed"],
        sample=sample,
        out_dir=out_dir,
        out_prefix=out_prefix,
        dpi=args["dpi"],
    )
    plot_training_diagnostics(
        result["ll_hist"],
        result["entropy_hist"],
        result["pi_hist"],
        result["tau_hist"],
        result["invphi_hist"],
        result["phase_frac_hist"],
        cn_states,
        result["obj_label"],
        sample=sample,
        out_dir=out_dir,
        out_prefix=out_prefix,
        dpi=args["dpi"],
    )

    logging.info(f"CNP-HMM inference complete. outputs in {out_dir}")
    logging.root.removeHandler(_file_handler)
    _file_handler.close()
