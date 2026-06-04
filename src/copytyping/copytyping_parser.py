import argparse
import os

from copytyping.utils import ALL_PLATFORMS


def add_arguments_inference_inputs(parser: argparse.ArgumentParser):
    """I/O paths and run-control flags for a single inference run."""
    parser.add_argument(
        "--platform",
        required=True,
        type=str,
        choices=ALL_PLATFORMS,
        help="Platform: single_cell (Cell_Model) or spatial (Spot_Model)",
    )
    parser.add_argument("--sample", required=True, type=str, help="sample name")
    parser.add_argument(
        "--gex_dir",
        required=False,
        type=str,
        default=None,
        help="/path/to/gex modality directory",
    )
    parser.add_argument(
        "--atac_dir",
        required=False,
        type=str,
        default=None,
        help="/path/to/atac modality directory",
    )
    parser.add_argument(
        "--seg_ucn",
        required=True,
        type=str,
        help="HATCHet seg.ucn.tsv file with segment-level copy numbers. "
        "--gex_dir contains bbc-level data which will be aggregated "
        "to segment level.",
    )
    parser.add_argument(
        "--bbc_phases",
        required=True,
        type=str,
        help="TSV file with BBC block phase assignments "
        "(must contain #CHR, START, END, PHASE columns).",
    )
    parser.add_argument(
        "--solfile",
        required=False,
        type=str,
        default=None,
        help="Optional HATCHet solution.tsv to override "
        "CN profiles (matched by CLUSTER ID).",
    )
    parser.add_argument(
        "--no_normal",
        required=False,
        action="store_true",
        default=argparse.SUPPRESS,
        help="If set, use major clone CNP to derive baseline.",
    )
    parser.add_argument(
        "--cell_type",
        required=False,
        type=str,
        default=None,
        help="/path/to/cell_types.tsv.gz, with header BARCODE and <ref_label>",
    )
    parser.add_argument(
        "--ref_label",
        required=False,
        default=argparse.SUPPRESS,
        type=str,
        help="Reference label colname in --cell_type",
    )
    parser.add_argument(
        "--method",
        required=False,
        type=str,
        default=argparse.SUPPRESS,
        choices=["copytyping", "bulk_anchored_copytyping", "kmeans"],
        help="copytyping (bulk-CNP-fixed EM), bulk_anchored_copytyping "
        "(divisive clone-adding outer loop on top of the EM), or kmeans "
        "(feature-based clustering)",
    )
    parser.add_argument("-o", "--out_dir", required=True, type=str)
    parser.add_argument(
        "--out_prefix",
        required=False,
        default=argparse.SUPPRESS,
        type=str,
        help="<out_dir>/<out_prefix>.",
    )
    parser.add_argument(
        "--genome_size",
        required=True,
        type=str,
        help="Reference chromosome sizes file (e.g., hg19.chrom.sizes)",
    )
    parser.add_argument(
        "--region_bed",
        required=True,
        type=str,
        help="Reference chromosome BED file (e.g., hg19.chrom.bed)",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        required=False,
        default=argparse.SUPPRESS,
        type=int,
        help="verbose level, 0, 1, 2",
    )
    return parser


def add_arguments_inference_parameters(parser: argparse.ArgumentParser):
    """Model, smoothing, and plot parameters (all defaulted, tunable from CLI)."""
    parser.add_argument(
        "--exclude",
        required=False,
        type=str,
        default=argparse.SUPPRESS,
        help="comma-separated ref_label values to exclude from analysis. "
        "Requires --cell_type.",
    )
    parser.add_argument(
        "--keep_cn_row",
        required=False,
        type=str,
        default=argparse.SUPPRESS,
        help="Comma-separated whitelist of CNP rows (each row is the existing "
        "';'-joined per-clone format, e.g. '1|1;2|0;2|1,1|1;1|1;2|0'). "
        "Segments whose CNP is not in the list are dropped. Default: keep all.",
    )
    parser.add_argument(
        "--save_processed_data",
        required=False,
        action="store_true",
        default=argparse.SUPPRESS,
        help="If set, also save heavy processed_data files "
        "(seg.{X,Y,D}.npz, bbc.{X,Y,D}.npz, bbc.tsv.gz, labeling_trace.npz). "
        "Default: skip them (~300MB/run). Always saved regardless: "
        "annotations.tsv, cnp_profile.tsv, metadata.tsv, model_params.npz.",
    )
    parser.add_argument(
        "--ascn_profile",
        required=False,
        action="store_true",
        default=argparse.SUPPRESS,
        help="If set, render the CN profile track in plots as allele-specific "
        "(A/B sub-bars per clone, inferno palette) instead of the default "
        "joint (A|B) integer-CN palette.",
    )
    parser.add_argument(
        "--fit_mode",
        required=False,
        type=str,
        default=argparse.SUPPRESS,
        choices=["hybrid", "allele_only", "total_only"],
        help="Likelihood mode: hybrid (BAF+RDR), "
        "allele_only (BAF only), total_only (RDR only)",
    )
    parser.add_argument(
        "--niters",
        required=False,
        type=int,
        default=argparse.SUPPRESS,
        help="Max EM iterations",
    )
    parser.add_argument(
        "--baf_clip",
        required=False,
        default=argparse.SUPPRESS,
        type=float,
        help="Clip expected clone BAF to [baf_clip, 1-baf_clip] to avoid log(0)",
    )
    parser.add_argument(
        "--tau_bounds",
        required=False,
        default=argparse.SUPPRESS,
        type=str,
        help="Bounds for BB tau MLE (comma-separated). "
        "tau is a concentration parameter: larger=less dispersed",
    )
    parser.add_argument(
        "--invphi_bounds",
        required=False,
        default=argparse.SUPPRESS,
        type=str,
        help="Bounds for NB inv_phi MLE (comma-separated). "
        "inv_phi is a concentration parameter: larger=less dispersed",
    )
    parser.add_argument(
        "--pi_alpha",
        required=False,
        default=argparse.SUPPRESS,
        type=float,
        help="symmetric Dirichlet prior alpha for pi. "
        "1: non-informative, <1: sparse, >1: smoothing",
    )
    parser.add_argument(
        "--update_pi",
        required=False,
        action="store_true",
        default=argparse.SUPPRESS,
        help="Update pi (clone mixing proportions) during EM. "
        "Default: fix pi at its initial value.",
    )
    parser.add_argument(
        "--update_tau",
        required=False,
        action="store_true",
        default=argparse.SUPPRESS,
        help="if set, update BB dispersion (tau) in M-step (cell model only)",
    )
    parser.add_argument(
        "--update_invphi",
        required=False,
        action="store_true",
        default=argparse.SUPPRESS,
        help="if set, update NB dispersion (inv_phi) in M-step (cell model only)",
    )
    parser.add_argument(
        "--n_neighs",
        required=False,
        type=int,
        default=argparse.SUPPRESS,
        help="number of spatial neighbors (6 for Visium hexagonal)",
    )
    parser.add_argument(
        "--max_smooth_k",
        required=False,
        type=int,
        default=argparse.SUPPRESS,
        help="max adaptive smoothing level (0=none). Low-count spots get "
        "progressively smoothed k=1..max_smooth_k until thresholds met.",
    )
    parser.add_argument(
        "--min_umi_per_spot",
        required=False,
        type=int,
        default=argparse.SUPPRESS,
        help="minimum total UMI per spot for adaptive smoothing (0=disabled)",
    )
    parser.add_argument(
        "--min_snp_umi_per_spot",
        required=False,
        type=int,
        default=argparse.SUPPRESS,
        help="minimum total allele UMI per spot for adaptive smoothing (0=disabled)",
    )
    parser.add_argument(
        "--dpi",
        required=False,
        type=int,
        default=argparse.SUPPRESS,
        help="DPI for plots",
    )
    parser.add_argument(
        "--heatmap_agg",
        required=False,
        type=int,
        default=argparse.SUPPRESS,
        help="aggregate observations in heatmap",
    )
    parser.add_argument(
        "--min_snp_count",
        required=False,
        type=int,
        default=argparse.SUPPRESS,
        help="min SNP count per adaptive BBC bin for 1d scatter",
    )
    parser.add_argument(
        "--max_bin_length",
        required=False,
        type=int,
        default=argparse.SUPPRESS,
        help="max bin length (bp) for adaptive BBC binning",
    )
    return parser


def add_arguments_inference(parser: argparse.ArgumentParser):
    add_arguments_inference_inputs(parser)
    add_arguments_inference_parameters(parser)
    return parser


def check_arguments_inference(args: dict):
    gex_dir = args["gex_dir"]
    atac_dir = args["atac_dir"]

    args["gex_h5ad"] = None
    args["atac_h5ad"] = None

    data_types = []
    if gex_dir is not None:
        assert os.path.isdir(gex_dir), f"--gex_dir={gex_dir} is not a directory"
        data_types.append("gex")
        cnv_segments = os.path.join(gex_dir, "cnv_segments.tsv")
        barcodes = os.path.join(gex_dir, "barcodes.tsv.gz")
        x_count = os.path.join(gex_dir, "bb.Xcount.npz")
        a_allele = os.path.join(gex_dir, "bb.Aallele.npz")
        b_allele = os.path.join(gex_dir, "bb.Ballele.npz")
        for file in [cnv_segments, barcodes, x_count, a_allele, b_allele]:
            assert os.path.exists(file), f"missing file: {file}"
        args["gex_barcodes"] = barcodes
        args["gex_cnv_segments"] = cnv_segments
        args["gex_X_count"] = x_count
        args["gex_A_allele"] = a_allele
        args["gex_B_allele"] = b_allele

        h5ad_files = sorted(f for f in os.listdir(gex_dir) if f.endswith(".h5ad"))
        if h5ad_files:
            args["gex_h5ad"] = os.path.join(gex_dir, h5ad_files[0])

    if atac_dir is not None:
        assert os.path.isdir(atac_dir), f"--atac_dir={atac_dir} is not a directory"
        data_types.append("atac")
        cnv_segments = os.path.join(atac_dir, "cnv_segments.tsv")
        barcodes = os.path.join(atac_dir, "barcodes.tsv.gz")
        x_count = os.path.join(atac_dir, "bb.Xcount.npz")
        a_allele = os.path.join(atac_dir, "bb.Aallele.npz")
        b_allele = os.path.join(atac_dir, "bb.Ballele.npz")
        for file in [cnv_segments, barcodes, x_count, a_allele, b_allele]:
            assert os.path.exists(file), f"missing file: {file}"
        args["atac_barcodes"] = barcodes
        args["atac_cnv_segments"] = cnv_segments
        args["atac_X_count"] = x_count
        args["atac_A_allele"] = a_allele
        args["atac_B_allele"] = b_allele

    assert len(data_types) > 0, (
        "at least one of --gex_dir or --atac_dir must be provided"
    )
    args["data_types"] = data_types
    platform = args["platform"]
    if platform == "spatial":
        assert gex_dir is not None, (
            "--gex_dir is required for spatial platform (h5ad with spatial coords)"
        )

    assert os.path.exists(args["seg_ucn"]), f"invalid --seg_ucn file"
    assert os.path.exists(args["bbc_phases"]), f"invalid --bbc_phases file"
    if args["solfile"] is not None:
        assert os.path.exists(args["solfile"]), f"invalid --solfile: {args['solfile']}"
    assert args["region_bed"] is not None and os.path.exists(args["region_bed"]), (
        f"invalid --region_bed: {args['region_bed']}"
    )

    def _parse_bounds(s):
        """Parse comma-separated bounds string like '1.0,1e6' into (float, float)."""
        lo, hi = s.split(",")
        return (float(lo), float(hi))

    min_tau, max_tau = _parse_bounds(args["tau_bounds"])
    min_invphi, max_invphi = _parse_bounds(args["invphi_bounds"])
    args["min_tau"] = min_tau
    args["max_tau"] = max_tau
    args["min_invphi"] = min_invphi
    args["max_invphi"] = max_invphi

    return args


def add_arguments_cnphmm_parameters(parser: argparse.ArgumentParser):
    """Factorial CNP-HMM knobs (all defaulted in copytyping.yaml)."""
    parser.add_argument(
        "--cnphmm_method",
        required=False,
        type=str,
        default=argparse.SUPPRESS,
        choices=["baum_welch", "block_ascent"],
        help="baum_welch (forward-backward) or block_ascent (Viterbi coordinate)",
    )
    parser.add_argument(
        "--decode",
        required=False,
        type=str,
        default=argparse.SUPPRESS,
        choices=["map", "viterbi"],
        help="baum_welch per-cell decode: map (posterior) or viterbi (joint-MAP)",
    )
    parser.add_argument(
        "--c_max",
        required=False,
        type=int,
        default=argparse.SUPPRESS,
        help="Max total copy number for the (a,b) state grid",
    )
    parser.add_argument(
        "--mask_mode",
        required=False,
        type=str,
        default=argparse.SUPPRESS,
        choices=["full", "bulk", "bulk_neighbor"],
        help="State-space mask: full grid, bulk-observed, or bulk+neighbors",
    )
    parser.add_argument(
        "--prior_s",
        required=False,
        type=float,
        default=argparse.SUPPRESS,
        help="Dirichlet concentration s for the transition prior",
    )
    parser.add_argument(
        "--prior_omega",
        required=False,
        type=float,
        default=argparse.SUPPRESS,
        help="Bulk-vs-baseline mix weight (0=baseline, 1=bulk)",
    )
    parser.add_argument(
        "--prior_t",
        required=False,
        type=float,
        default=argparse.SUPPRESS,
        help="Baseline self-transition probability",
    )
    parser.add_argument(
        "--prior_eps",
        required=False,
        type=float,
        default=argparse.SUPPRESS,
        help="Smoothing constant for bulk transition counts",
    )
    parser.add_argument(
        "--fix_dispersion",
        required=False,
        action="store_true",
        default=argparse.SUPPRESS,
        help="Fix BB tau / NB inv_phi at init instead of updating them in M-step "
        "(default: update both)",
    )
    parser.add_argument(
        "--cluster_method",
        required=False,
        type=str,
        default=argparse.SUPPRESS,
        choices=["lexsort", "cnt_nj", "cnt_upgma", "cnt_complete"],
        help="Heatmap cell ordering: lexsort, or a CNT-distance tree "
        "(cnt_nj / cnt_upgma / cnt_complete); distance methods fall back to "
        "lexsort above ~2000 unique profiles",
    )
    parser.add_argument(
        "--n_clones",
        required=False,
        type=int,
        default=argparse.SUPPRESS,
        help="Max clones per REP for flat clustering of decoded paths (CNT-distance)",
    )
    return parser


def add_arguments_cnphmm(parser: argparse.ArgumentParser):
    add_arguments_inference_inputs(parser)
    add_arguments_inference_parameters(parser)
    add_arguments_cnphmm_parameters(parser)
    return parser


def check_arguments_cnphmm(args: dict):
    """Validate shared inference inputs, then the CNP-HMM-specific knobs."""
    args = check_arguments_inference(args)
    assert args["c_max"] >= 1, f"--c_max must be >= 1, got {args['c_max']}"
    assert 0.0 <= args["prior_omega"] <= 1.0, "--prior_omega must be in [0, 1]"
    assert 0.0 < args["prior_t"] < 1.0, "--prior_t must be in (0, 1)"
    assert args["prior_s"] > 0, "--prior_s must be > 0"
    assert args["prior_eps"] > 0, "--prior_eps must be > 0"
    return args


def add_arguments_pipeline(parser):
    parser.add_argument(
        "panel_tsv",
        type=str,
        help="Panel TSV file with one row per sample/run",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        required=True,
        type=str,
        help="Base output directory",
    )
    parser.add_argument(
        "--genome_size",
        required=True,
        type=str,
        help="Chromosome sizes file",
    )
    parser.add_argument(
        "--region_bed",
        required=True,
        type=str,
        help="Chromosome regions BED file",
    )
    parser.add_argument(
        "--platform_filter",
        default=None,
        type=str,
        choices=["spatial", "single_cell"],
        help="Only run samples matching this platform",
    )
    parser.add_argument(
        "--sol_pattern",
        default=argparse.SUPPRESS,
        type=str,
        help="Glob pattern for solfiles under PATH_TO_SOLDIR. "
        "{SOLID} is replaced by the SOLID column value.",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="List of sample names to run (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Re-run even if output already exists",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        default=argparse.SUPPRESS,
        type=int,
        help="Verbose level for each inference run",
    )
    add_arguments_inference_parameters(parser)
    add_arguments_validate_parameters(parser)
    return parser


def add_arguments_validate_parameters(parser: argparse.ArgumentParser):
    """Validate-only parameters (cutoff sweeps). Plot/smoothing params are
    inherited from add_arguments_inference_parameters."""
    parser.add_argument(
        "--purity_cutoff",
        required=False,
        type=str,
        default=argparse.SUPPRESS,
        help="comma-separated purity cutoffs for hard label evaluation. "
        "Spots with purity <= cutoff labeled normal.",
    )
    parser.add_argument(
        "--post_cutoff",
        required=False,
        type=str,
        default=argparse.SUPPRESS,
        help="comma-separated max_posterior cutoffs. "
        "Tumor spots with max_posterior <= cutoff labeled NA (excluded from metrics).",
    )
    return parser


def add_arguments_validate(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--processed_data",
        required=True,
        type=str,
        help="Directory with cnp_profile.tsv, X/Y/D.npz, model_params.npz",
    )
    parser.add_argument(
        "--pred_labels",
        required=True,
        type=str,
        help="TSV with BARCODE + predicted label columns",
    )
    parser.add_argument(
        "--pred_label",
        required=False,
        type=str,
        default=argparse.SUPPRESS,
        help="Column name for predicted labels",
    )
    parser.add_argument(
        "--ref_labels",
        required=False,
        type=str,
        default=None,
        help="TSV with BARCODE + reference label columns",
    )
    parser.add_argument(
        "--ref_label",
        required=False,
        type=str,
        default="path_label",
        help="Column name for reference labels (default: path_label)",
    )
    parser.add_argument(
        "--method",
        required=False,
        type=str,
        default=argparse.SUPPRESS,
        help="Method that produced the labels: copytyping or others. "
        "Non-copytyping skips init normal, purity sweep, trace, count histograms.",
    )
    parser.add_argument("--sample", required=True, type=str, help="sample name")
    parser.add_argument(
        "--genome_size",
        required=False,
        type=str,
        default=None,
        help="Chromosome sizes file (for 1d scatter)",
    )
    parser.add_argument(
        "--region_bed",
        required=False,
        type=str,
        default=None,
        help="Chromosome regions BED file (for heatmap/scatter)",
    )
    parser.add_argument(
        "--h5ad",
        required=False,
        type=str,
        default=None,
        help="h5ad file for spatial neighbors (joincount + visium plots)",
    )
    parser.add_argument(
        "-o", "--out_dir", required=True, type=str, help="output directory"
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        required=False,
        type=int,
        default=argparse.SUPPRESS,
        help="Verbose level",
    )
    add_arguments_inference_parameters(parser)
    add_arguments_validate_parameters(parser)
    return parser
