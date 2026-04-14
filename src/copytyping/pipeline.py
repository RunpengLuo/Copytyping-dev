"""copytyping run_pipeline — batch runner from a panel TSV.

Panel TSV columns (tab-separated, header required):
Required columns:
    SAMPLE            : sample name
    PLATFORM          : "spatial" or "single_cell"
    PATH_TO_BB_INPUT  : path to bb dir (contains VISIUM/ or scRNA/ scATAC/)
    PATH_TO_SEG       : path to seg.ucn.tsv
    PATH_TO_BBC_PHASE : path to bbc phases TSV
    PATH_TO_SOLFILE   : path to solfile (empty = use seg file directly)
    REF_LABEL         : reference label column name (e.g. "path_label")

Optional columns:
    CELLTYPE_FILE     : path to cell type annotation TSV
"""

import logging
import os

import pandas as pd

from copytyping.copytyping_parser import get_inference_defaults
from copytyping.inference.inference import run as run_inference


def _build_run_args(row):
    """Build inference args dict from a panel row. Returns None to skip."""
    sample = row["SAMPLE"]
    platform = row["PLATFORM"]
    seg = row["PATH_TO_SEG"]
    bbc_phase = row["PATH_TO_BBC_PHASE"]
    bb_input = row["PATH_TO_BB_INPUT"]
    solfile = row["PATH_TO_SOLFILE"]
    ref_label = row["REF_LABEL"]
    celltype_file = row.get("CELLTYPE_FILE", "")

    for name, val in [("PATH_TO_SEG", seg), ("PATH_TO_BBC_PHASE", bbc_phase)]:
        if not val or not os.path.isfile(val):
            logging.warning(f"SKIP {sample}: missing {name}={val}")
            return None
    if not bb_input or not os.path.isdir(bb_input):
        logging.warning(f"SKIP {sample}: missing PATH_TO_BB_INPUT={bb_input}")
        return None
    if solfile and not os.path.isfile(solfile):
        logging.warning(f"SKIP {sample}: missing PATH_TO_SOLFILE={solfile}")
        return None

    args = {
        "platform": platform,
        "sample": sample,
        "seg_ucn": seg,
        "bbc_phases": bbc_phase,
        "out_prefix": sample,
        "ref_label": ref_label,
    }

    if platform == "spatial":
        args["gex_dir"] = os.path.join(bb_input, "VISIUM")
    else:
        scrna = os.path.join(bb_input, "scRNA")
        scatac = os.path.join(bb_input, "scATAC")
        if os.path.isdir(scrna):
            args["gex_dir"] = scrna
        if os.path.isdir(scatac):
            args["atac_dir"] = scatac

    if celltype_file and os.path.isfile(celltype_file):
        args["cell_type"] = celltype_file

    if solfile:
        args["solfile"] = solfile

    return args


def run(args):
    if hasattr(args, "__dict__"):
        args = vars(args)

    panel_tsv = args["panel_tsv"]
    out_dir = args["out_dir"]
    genome_size = args["genome_size"]
    region_bed = args["region_bed"]
    platform_filter = args.get("platform_filter")
    force = args.get("force", False)
    verbosity = args.get("verbosity", 0)

    panel = pd.read_table(panel_tsv, dtype=str).fillna("")
    required = [
        "SAMPLE", "PLATFORM", "PATH_TO_SEG", "PATH_TO_BBC_PHASE",
        "PATH_TO_BB_INPUT", "PATH_TO_SOLFILE", "REF_LABEL",
    ]
    for col in required:
        assert col in panel.columns, f"missing column: {col}"

    summary_file = os.path.join(out_dir, "pipeline_summary.tsv")
    summary_rows = []

    n_runs = 0
    n_skipped = 0
    for _, row in panel.iterrows():
        if platform_filter and row["PLATFORM"] != platform_filter:
            continue

        run_args = _build_run_args(row)
        if run_args is None:
            continue

        solfile = row["PATH_TO_SOLFILE"]
        if solfile and os.path.isfile(solfile):
            parent = os.path.basename(os.path.dirname(solfile))
            basename = os.path.basename(solfile).replace(".tsv", "")
            tag = f"{parent}_{basename}"
        else:
            tag = "default"
        run_dir = os.path.join(
            out_dir, run_args["sample"], run_args["platform"], tag
        )

        prefix = run_args["out_prefix"]
        platform = run_args["platform"]
        ann_file = os.path.join(run_dir, f"{prefix}.{platform}.annotations.tsv")
        eval_file = os.path.join(run_dir, f"{prefix}.{platform}.evaluation.tsv")
        out_tag = os.path.relpath(run_dir, out_dir)

        # Build the input row for this run
        input_info = dict(row)
        input_info["OUT_PREFIX"] = out_tag

        if not force and os.path.isfile(ann_file):
            logging.info(f"SKIP (exists): {run_dir}")
            n_skipped += 1
        else:
            inf_args = {**get_inference_defaults(), **run_args}
            inf_args["out_dir"] = run_dir
            inf_args["genome_size"] = genome_size
            inf_args["region_bed"] = region_bed
            inf_args["verbosity"] = verbosity
            os.makedirs(run_dir, exist_ok=True)

            logging.info(f"RUN: {run_dir}")
            root = logging.getLogger()
            prev_level = root.level
            try:
                root.setLevel(logging.WARNING)
                run_inference(inf_args)
                root.setLevel(prev_level)
                n_runs += 1
            except Exception as e:
                root.setLevel(prev_level)
                logging.error(f"FAILED {run_dir}: {e}")
                input_info["STATUS"] = "FAILED"
                summary_rows.append(input_info)
                continue

        if os.path.isfile(eval_file):
            eval_df = pd.read_table(eval_file)
            input_info.update(eval_df.iloc[0].to_dict())
            logging.info(eval_df.to_string(index=False))

        input_info["STATUS"] = "OK"
        summary_rows.append(input_info)

    logging.info(f"pipeline: {n_runs} completed, {n_skipped} skipped")

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(summary_file, sep="\t", index=False, na_rep="")
        logging.info(f"saved {len(summary)} rows to {summary_file}")
