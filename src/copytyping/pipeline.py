"""copytyping run_pipeline — batch runner from a panel TSV.

Panel TSV columns (tab-separated, header required):
Required columns:
    SAMPLE            : sample name
    PLATFORM          : "spatial" or "single_cell"
    PATH_TO_BB_INPUT  : path to bb dir (contains VISIUM/ or scRNA/ scATAC/)
    PATH_TO_SEG       : path to seg.ucn.tsv
    PATH_TO_BBC_PHASE : path to bbc phases TSV
    PATH_TO_SOLDIR    : path to directory containing solfiles
    SOLID             : "DEFAULT" (no solfile) or pattern to match solfiles
    PLOIDY            : ploidy value (used in output directory naming)
    CLONE             : clone count (used in output directory naming)
    REF_LABEL         : reference label column name (e.g. "path_label")

Optional columns:
    CELLTYPE_FILE     : path to cell type annotation TSV

When SOLID is not "DEFAULT", the pipeline globs PATH_TO_SOLDIR for files
matching the user-specified --sol_pattern (default: "*{SOLID}*.tsv"),
creating one run per matched solfile.
"""

import glob
import logging
import os

import pandas as pd

from copytyping.plot.plot_common import plot_joincount_boxplot, plot_metrics_barplot

from copytyping.copytyping_parser import get_inference_defaults
from copytyping.inference.inference import run as run_inference
from copytyping.inference.validation import _eval_subset


def _build_base_args(row):
    """Build base inference args from a panel row. Returns None to skip."""
    sample = row["SAMPLE"]
    platform = row["PLATFORM"]
    seg = row["PATH_TO_SEG"]
    bbc_phase = row["PATH_TO_BBC_PHASE"]
    bb_input = row["PATH_TO_BB_INPUT"]
    ref_label = row["REF_LABEL"]
    celltype_file = row.get("CELLTYPE_FILE", "")

    for name, val in [("PATH_TO_SEG", seg), ("PATH_TO_BBC_PHASE", bbc_phase)]:
        if not val or not os.path.isfile(val):
            logging.warning(f"SKIP {sample}: missing {name}={val}")
            return None
    if not bb_input or not os.path.isdir(bb_input):
        logging.warning(f"SKIP {sample}: missing PATH_TO_BB_INPUT={bb_input}")
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

    return args


def _resolve_solfiles(row, sol_pattern):
    """Resolve solfiles for a panel row. Returns list of (solfile_path, tag)."""
    soldir = row["PATH_TO_SOLDIR"]
    solid = row["SOLID"]

    if solid == "DEFAULT" or not solid:
        return [(None, "default")]

    if not soldir or not os.path.isdir(soldir):
        logging.warning(f"SKIP {row['SAMPLE']}: missing PATH_TO_SOLDIR={soldir}")
        return []

    pattern = os.path.join(soldir, sol_pattern.format(SOLID=solid))
    matches = sorted(glob.glob(pattern))
    if not matches:
        logging.warning(f"SKIP {row['SAMPLE']}: no solfiles matching {pattern}")
        return []

    results = []
    for sf in matches:
        basename = os.path.basename(sf).replace(".tsv", "")
        results.append((sf, basename))
    return results


def _run_one(
    run_args, run_dir, genome_size, region_bed, verbosity, force, extra_args=None
):
    """Run inference for one configuration. Returns (status, eval_dict)."""
    prefix = run_args["out_prefix"]
    platform = run_args["platform"]
    ann_file = os.path.join(run_dir, f"{prefix}.{platform}.annotations.tsv")
    eval_file = os.path.join(run_dir, f"{prefix}.{platform}.evaluation.tsv")

    if not force and os.path.isfile(ann_file):
        logging.info(f"SKIP (exists): {run_dir}")
        ref_label = run_args.get("ref_label", "")
        anns = pd.read_table(ann_file)
        # Detect qry_label: column ending with "-label" but not "-refined"
        qry_labels = [
            c
            for c in anns.columns
            if c.endswith("-label") and not c.endswith("-refined")
        ]
        metrics = {}
        if qry_labels and ref_label in anns.columns:
            qry_label = qry_labels[0]
            tumor_post = "tumor_purity" if platform == "spatial" else "tumor"
            metrics = _eval_subset(anns, qry_label, ref_label, tumor_post)
            # Preserve JC_* from old eval file (requires spatial coords)
            if os.path.isfile(eval_file):
                old = pd.read_table(eval_file).iloc[0].to_dict()
                for k, v in old.items():
                    if k.startswith("JC_"):
                        metrics[k] = v
        elif os.path.isfile(eval_file):
            metrics = pd.read_table(eval_file).iloc[0].to_dict()
        return "SKIPPED", metrics

    inf_args = {**get_inference_defaults(), **run_args}
    inf_args["out_dir"] = run_dir
    inf_args["genome_size"] = genome_size
    inf_args["region_bed"] = region_bed
    inf_args["verbosity"] = verbosity
    if extra_args:
        inf_args.update(extra_args)
    os.makedirs(run_dir, exist_ok=True)

    logging.info(f"RUN: {run_dir}")
    run_inference(inf_args)

    metrics = {}
    if os.path.isfile(eval_file):
        metrics = pd.read_table(eval_file).iloc[0].to_dict()
        logging.info(pd.read_table(eval_file).to_string(index=False))
    return "OK", metrics


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
    sol_pattern = args.get("sol_pattern", "*{SOLID}*.tsv")
    extra_args = {}
    if args.get("update_purity"):
        extra_args["update_purity"] = True
    if args.get("update_NB_dispersion"):
        extra_args["update_NB_dispersion"] = True
    if args.get("update_BB_dispersion"):
        extra_args["update_BB_dispersion"] = True
    if args.get("init_baseline_by_cell_type"):
        extra_args["init_baseline_by_cell_type"] = True

    panel = pd.read_table(panel_tsv, dtype=str).fillna("")
    required = [
        "SAMPLE",
        "PLATFORM",
        "PATH_TO_SEG",
        "PATH_TO_BBC_PHASE",
        "PATH_TO_BB_INPUT",
        "PATH_TO_SOLDIR",
        "SOLID",
        "PLOIDY",
        "CLONE",
        "REF_LABEL",
    ]
    for col in required:
        assert col in panel.columns, f"missing column: {col}"

    summary_file = os.path.join(out_dir, "pipeline_summary.tsv")
    summary_rows = []
    n_runs = 0
    n_skipped = 0

    sample_filter = args.get("samples")
    for _, row in panel.iterrows():
        if platform_filter and row["PLATFORM"] != platform_filter:
            continue
        if sample_filter and row["SAMPLE"] not in sample_filter:
            continue

        base_args = _build_base_args(row)
        if base_args is None:
            continue

        solfiles = _resolve_solfiles(row, sol_pattern)
        for solfile, tag in solfiles:
            run_args = {**base_args}
            if solfile:
                run_args["solfile"] = solfile

            ploidy = row["PLOIDY"]
            clone = row["CLONE"]
            run_dir = os.path.join(
                out_dir,
                run_args["sample"],
                run_args["platform"],
                f"{ploidy}_n{clone}_{tag}",
            )

            status, metrics = _run_one(
                run_args,
                run_dir,
                genome_size,
                region_bed,
                verbosity,
                force,
                extra_args=extra_args,
            )

            dtypes = []
            if "gex_dir" in run_args:
                dtypes.append("gex")
            if "atac_dir" in run_args:
                dtypes.append("atac")
            info = dict(row)
            info["OUT_PREFIX"] = os.path.relpath(run_dir, out_dir)
            info["SOLFILE"] = solfile or ""
            info["DATA_TYPES"] = "+".join(dtypes) if dtypes else ""
            info["STATUS"] = status
            info.update(metrics)
            summary_rows.append(info)

            if status == "OK":
                n_runs += 1
            elif status == "SKIPPED":
                n_skipped += 1

    logging.info(f"pipeline: {n_runs} completed, {n_skipped} skipped")

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        # Sort metric columns: #clone*, purity_*, JC_* grouped
        fixed = [
            c
            for c in summary.columns
            if not c.startswith("#clone")
            and not c.startswith("purity_")
            and not c.startswith("JC_")
        ]
        clone_cols = sorted([c for c in summary.columns if c.startswith("#clone")])
        purity_cols = sorted([c for c in summary.columns if c.startswith("purity_")])
        jc_cols = sorted([c for c in summary.columns if c.startswith("JC_")])
        summary = summary[fixed + clone_cols + purity_cols + jc_cols]
        summary.to_csv(summary_file, sep="\t", index=False, na_rep="")
        logging.info(f"saved {len(summary)} rows to {summary_file}")

        plot_metrics_barplot(
            summary,
            os.path.join(out_dir, "pipeline_metrics.pdf"),
        )
        plot_joincount_boxplot(
            summary,
            os.path.join(out_dir, "pipeline_joincount.pdf"),
        )
