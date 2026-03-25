import os
import sys
import argparse

from copytyping.utils import *


def add_arguments_inference(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--assay_type",
        required=True,
        type=str,
        choices=ALL_ASSAYS,
    )
    parser.add_argument("--sample", required=True, type=str, help="sample name")
    parser.add_argument(
        "--gex_dir",
        required=False,
        type=str,
        default=None,
        help="/path/to/allele/<assay_type>",
    )
    parser.add_argument(
        "--atac_dir",
        required=False,
        type=str,
        default=None,
        help="/path/to/allele/<assay_type>",
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
        "--gex_dir contains bbc-level data which will be aggregated to segment level.",
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
        help="Likelihood mode: hybrid (BAF+RDR), allele_only (BAF only), total_only (RDR only)",
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
        "--min_tau",
        required=False,
        default=50.0,
        type=float,
        help="minimum BB dispersion (also used as init value)",
    )
    parser.add_argument(
        "--max_tau",
        required=False,
        default=500.0,
        type=float,
        help="maximum BB dispersion bound for MLE",
    )
    parser.add_argument(
        "--min_phi",
        required=False,
        default=0.001,
        type=float,
        help="minimum NB dispersion phi (inv_phi upper bound = 1/min_phi)",
    )
    parser.add_argument(
        "--max_phi",
        required=False,
        default=100.0,
        type=float,
        help="maximum NB dispersion phi (inv_phi lower bound = 1/max_phi)",
    )
    parser.add_argument(
        "--pi_alpha",
        required=False,
        default=1.0,
        type=float,
        help="symmetric Dirichlet prior alpha for pi. 1: MLE (default), <1: sparse, >1: smoothing",
    )

    parser.add_argument(
        "--fix_NB_dispersion",
        required=False,
        action="store_true",
        default=False,
        help="fix Negative-binomial dispersion after init",
    )
    parser.add_argument(
        "--share_NB_dispersion",
        required=False,
        action="store_true",
        default=False,
        help="if set, all CN states share same NB dispersion parameter.",
    )
    parser.add_argument(
        "--fix_BB_dispersion",
        required=False,
        action="store_true",
        default=False,
        help="if set, all CN states share same BB dispersion parameter.",
    )
    parser.add_argument(
        "--share_BB_dispersion",
        required=False,
        action="store_true",
        default=False,
        help="share Beta-binomial dispersion across CN states in M-step",
    )
    parser.add_argument(
        "--fix_tumor_purity",
        required=False,
        action="store_true",
        default=False,
        help="fix tumor purity after init. Default: update theta in M-step.",
    )
    parser.add_argument(
        "--no-fix_tumor_purity",
        dest="fix_tumor_purity",
        action="store_false",
    )
    # post selection
    parser.add_argument(
        "--posterior_thres",
        required=False,
        default=0.50,
        type=float,
        help="Assign cells/spots to NA if posterior less than <posterior_thres>",
    )
    parser.add_argument(
        "--margin_thres",
        required=False,
        default=0.10,
        type=float,
        help="Assign cells/spots to NA if top-2 margin less than <posterior_thres>",
    )
    parser.add_argument(
        "--tumorprop_threshold",
        required=False,
        default=0.50,
        type=float,
        help="For spatial assays, assign spots to normal if tumor purity < threshold",
    )
    parser.add_argument(
        "--refine_label_by_reference",
        required=False,
        action="store_true",
        default=False,
        help=f"mark unassigned if predicted label disagree with cell type (if available).",
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
        default="png",
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
        help=f"plot UMAP using BAF&RDR features",
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
    assay_type = args["assay_type"]
    gex_dir = args["gex_dir"]
    atac_dir = args["atac_dir"]

    data_types = []
    if assay_type in GEX_ASSAYS:
        assert gex_dir is not None and os.path.isdir(gex_dir), (
            f"missing --gex_dir={gex_dir} for assay_type={assay_type}"
        )
        data_types.append("gex")
        cnv_segments = os.path.join(gex_dir, "cnv_segments.tsv")
        barcodes = os.path.join(gex_dir, "barcodes.tsv.gz")
        x_count = os.path.join(gex_dir, "X_count.npz")
        y_count = os.path.join(gex_dir, "Y_count.npz")
        d_count = os.path.join(gex_dir, "D_count.npz")
        mod = "scRNA" if assay_type == "multiome" else assay_type
        h5ad = os.path.join(gex_dir, f"{mod}.h5ad")
        for file in [cnv_segments, barcodes, x_count, y_count, d_count, h5ad]:
            assert os.path.exists(file), f"missing file: {file}"
        args["gex_barcodes"] = barcodes
        args["gex_cnv_segments"] = cnv_segments
        args["gex_X_count"] = x_count
        args["gex_Y_count"] = y_count
        args["gex_D_count"] = d_count
        args["gex_h5ad"] = h5ad

    if assay_type in ATAC_ASSAYS:
        assert atac_dir is not None and os.path.isdir(atac_dir), (
            f"missing --atac_dir={atac_dir} for assay_type={assay_type}"
        )
        data_types.append("atac")
        cnv_segments = os.path.join(atac_dir, "cnv_segments.tsv")
        barcodes = os.path.join(atac_dir, "barcodes.tsv.gz")
        x_count = os.path.join(atac_dir, "X_count.npz")
        y_count = os.path.join(atac_dir, "Y_count.npz")
        d_count = os.path.join(atac_dir, "D_count.npz")
        for file in [cnv_segments, barcodes, x_count, y_count, d_count]:
            assert os.path.exists(file), f"missing file: {file}"
        args["atac_barcodes"] = barcodes
        args["atac_cnv_segments"] = cnv_segments
        args["atac_X_count"] = x_count
        args["atac_Y_count"] = y_count
        args["atac_D_count"] = d_count
    seg_ucn = args.get("seg_ucn", None)
    if seg_ucn is not None:
        assert os.path.exists(seg_ucn), f"missing --seg_ucn file: {seg_ucn}"
    args["seg_ucn"] = seg_ucn
    args["data_types"] = data_types
    return args
