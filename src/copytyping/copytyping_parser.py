import argparse
import os

from copytyping.utils import ALL_PLATFORMS


def add_arguments_inference(parser: argparse.ArgumentParser):
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
        "--gex_h5ad",
        required=False,
        type=str,
        default=None,
        help="Path to gex h5ad file (auto-detected as scRNA.h5ad "
        "in --gex_dir if not provided)",
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
        default="cell_type",
        type=str,
        help="Reference label colname in --cell_type, (default: cell_type)",
    )
    parser.add_argument(
        "--method",
        required=False,
        type=str,
        default="copytyping",
        choices=["copytyping", "kmeans", "ward", "leiden"],
        help="copytyping, kmeans, ward, leiden",
    )
    parser.add_argument("-o", "--out_dir", required=True, type=str)
    parser.add_argument(
        "--out_prefix",
        required=False,
        default="sample",
        type=str,
        help="<out_dir>/<out_prefix>.",
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
        "--genome_size",
        required=True,
        type=str,
        help="Reference chromosome sizes file (e.g., hg19.chrom.sizes)",
    )
    parser.add_argument(
        "--region_bed",
        required=True,
        type=str,
        default="./data/chm13v2.0_region.bed",
        help="Reference chromosome BED file (e.g., hg19.chrom.bed)",
    )

    ##################################################
    # Parameters
    parser.add_argument(
        "--fit_mode",
        required=False,
        type=str,
        default="hybrid",
        choices=["hybrid", "allele_only", "total_only"],
        help="Likelihood mode: hybrid (BAF+RDR), "
        "allele_only (BAF only), total_only (RDR only)",
    )
    parser.add_argument(
        "--aggr_mode",
        required=False,
        default="clust",
        choices=["seg", "clust"],
        help="Aggregation mode: 'clust' merges segments "
        "with identical CNP (default), 'seg' uses raw segments.",
    )
    parser.add_argument(
        "--niters",
        required=False,
        type=int,
        default=100,
        help="num_iters=100",
    )
    parser.add_argument(
        "--laplace",
        required=False,
        default=0.01,
        type=float,
        help="Laplace smoothing term when computing clone BAF",
    )
    parser.add_argument(
        "--tau_bounds",
        required=False,
        default="50.0,5000.0",
        type=str,
        help="Bounds for BB tau MLE (comma-separated, default: 50.0,5000.0). "
        "tau is a concentration parameter: larger=less dispersed (BB→Binomial as tau→∞)",
    )
    parser.add_argument(
        "--invphi_bounds",
        required=False,
        default="20.0,5000.0",
        type=str,
        help="Bounds for NB inv_phi MLE (comma-separated, default: 20.0,5000.0). "
        "inv_phi is a concentration parameter: larger=less dispersed (NB→Poisson as inv_phi→∞)",
    )
    parser.add_argument(
        "--pi_alpha",
        required=False,
        default=1.0,
        type=float,
        help="symmetric Dirichlet prior alpha for pi. "
        "1: non-informative (default), <1: sparse, >1: smoothing",
    )

    parser.add_argument(
        "--update_purity",
        required=False,
        action="store_true",
        default=False,
        help="if set, update per-spot tumor purity (theta) "
        "in M-step (default: fixed after init, spatial only)",
    )
    # post selection
    parser.add_argument(
        "--posterior_thres",
        required=False,
        default=0.50,
        type=float,
        help="Assign cells/spots to NA if posterior < threshold",
    )
    parser.add_argument(
        "--margin_thres",
        required=False,
        default=0.10,
        type=float,
        help="Assign cells/spots to NA if top-2 margin < threshold",
    )
    ##################################################
    # plot parameters
    parser.add_argument(
        "--dpi", required=False, type=int, help="image resolution", default=300
    )
    parser.add_argument(
        "--transparent",
        required=False,
        action="store_true",
        default=False,
        help="transparent background",
    )
    parser.add_argument(
        "--img_type",
        required=False,
        choices=["pdf", "png", "svg"],
        type=str,
        help="file format (pdf, png, svg)",
        default="pdf",
    )
    parser.add_argument(
        "--purity_min",
        required=False,
        type=float,
        help="min purity for tumor-gated spots (default: 0.1)",
        default=0.1,
    )
    parser.add_argument(
        "--purity_tol",
        required=False,
        type=float,
        help="tumor spots with purity <= purity_min + purity_tol "
        "are labeled low_purity (default: 0.01)",
        default=0.01,
    )
    parser.add_argument(
        "--min_snp_count",
        required=False,
        type=int,
        help="min SNP count per adaptive BBC bin (default: 300)",
        default=300,
    )
    parser.add_argument(
        "--max_bin_length",
        required=False,
        type=int,
        help="max bin length (bp) for adaptive BBC binning (default: 5000000)",
        default=5_000_000,
    )
    parser.add_argument(
        "--heatmap_agg",
        required=False,
        type=int,
        help="aggregate observations in heatmap (default: 10)",
        default=10,
    )
    parser.add_argument(
        "--umap",
        required=False,
        action="store_true",
        default=False,
        help="plot UMAP using BAF&RDR features",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        required=False,
        default=0,
        type=int,
        help="verbose level, 0, 1, 2",
    )
    return parser


def check_arguments_inference(args: dict):
    gex_dir = args["gex_dir"]
    atac_dir = args["atac_dir"]

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

        # h5ad: use explicit --gex_h5ad, else pick first .h5ad in gex_dir
        gex_h5ad = args.get("gex_h5ad")
        if gex_h5ad is None:
            h5ad_files = sorted(f for f in os.listdir(gex_dir) if f.endswith(".h5ad"))
            if h5ad_files:
                gex_h5ad = os.path.join(gex_dir, h5ad_files[0])
        if gex_h5ad is not None:
            assert os.path.exists(gex_h5ad), f"missing file: {gex_h5ad}"
        args["gex_h5ad"] = gex_h5ad

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
    platform = args["platform"]
    if platform == "spatial":
        assert gex_dir is not None, (
            "--gex_dir is required for spatial platform (h5ad with spatial coords)"
        )

    seg_ucn = args.get("seg_ucn", None)
    if seg_ucn is not None:
        assert os.path.exists(seg_ucn), f"missing --seg_ucn file: {seg_ucn}"
    args["seg_ucn"] = seg_ucn
    bbc_phases = args.get("bbc_phases", None)
    if bbc_phases is not None:
        assert os.path.exists(bbc_phases), f"missing --bbc_phases file: {bbc_phases}"
    args["bbc_phases"] = bbc_phases
    solfile = args.get("solfile", None)
    if solfile is not None:
        assert os.path.exists(solfile), f"missing --solfile: {solfile}"
    args["data_types"] = data_types
    return args


def get_inference_defaults():
    """Get default inference args from the inference parser."""
    tmp = argparse.ArgumentParser()
    add_arguments_inference(tmp)
    return vars(
        tmp.parse_args(
            [
                "--platform",
                "single_cell",
                "--sample",
                "_",
                "--seg_ucn",
                "_",
                "--bbc_phases",
                "_",
                "-o",
                "_",
                "--genome_size",
                "_",
                "--region_bed",
                "_",
            ]
        )
    )


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
        default="*sol{SOLID}*.tsv",
        type=str,
        help="Glob pattern for solfiles under PATH_TO_SOLDIR. "
        "{SOLID} is replaced by the SOLID column value "
        "(default: '*sol{SOLID}*.tsv')",
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
        default=False,
        help="Re-run even if output already exists",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        default=0,
        type=int,
        help="Verbose level for each inference run",
    )
    parser.add_argument(
        "--update_purity",
        action="store_true",
        default=False,
        help="Update per-spot purity in M-step",
    )
    return parser
