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

import argparse
import glob
import logging
import os

import pandas as pd

from copytyping.copytyping_parser import (
    add_arguments_pipeline,
    check_arguments_inference,
)
from copytyping.inference.inference import run as run_inference
from copytyping.utils import normalize_args, setup_logging

from analysis_plots import _eval_subset, plot_joincount_boxplot, plot_metrics_barplot
from validate import run as run_validate


def _build_base_args(row):
    """Build per-row I/O args from a panel row. Returns None to skip."""
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

    base = {
        "platform": platform,
        "sample": sample,
        "seg_ucn": seg,
        "bbc_phases": bbc_phase,
        "out_prefix": sample,
        "ref_label": ref_label,
        "method": "copytyping",
        "gex_dir": None,
        "atac_dir": None,
        "cell_type": celltype_file
        if celltype_file and os.path.isfile(celltype_file)
        else None,
    }

    if platform == "spatial":
        base["gex_dir"] = os.path.join(bb_input, "VISIUM")
    else:
        scrna = os.path.join(bb_input, "scRNA")
        scatac = os.path.join(bb_input, "scATAC")
        if os.path.isdir(scrna):
            base["gex_dir"] = scrna
        if os.path.isdir(scatac):
            base["atac_dir"] = scatac

    return base


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


def _read_loglik(run_dir, prefix):
    """Read final log_likelihood from the saved model_params.npz, or NaN."""
    import numpy as np

    npz_path = os.path.join(run_dir, "processed_data", f"{prefix}.model_params.npz")
    if not os.path.isfile(npz_path):
        return float("nan")
    try:
        with np.load(npz_path) as data:
            if "log_likelihood" in data:
                return float(np.atleast_1d(data["log_likelihood"])[0])
    except Exception:
        pass
    return float("nan")


def _eval_metrics(ann_file, eval_file, ref_label, platform, prefix, run_dir):
    """Read metrics from an existing annotations/eval file."""
    metrics = {"log_likelihood": _read_loglik(run_dir, prefix)}
    anns = pd.read_table(ann_file)
    qry_labels = [
        c for c in anns.columns if c.endswith("_label") and not c.endswith("-refined")
    ]
    if qry_labels and ref_label in anns.columns:
        qry_label = qry_labels[0]
        tumor_post = "tumor_purity" if platform == "spatial" else "tumor"
        metrics.update(_eval_subset(anns, qry_label, ref_label, tumor_post))
        if os.path.isfile(eval_file):
            old = pd.read_table(eval_file).iloc[0].to_dict()
            for k, v in old.items():
                if k.startswith("JC_"):
                    metrics[k] = v
    elif os.path.isfile(eval_file):
        metrics.update(pd.read_table(eval_file).iloc[0].to_dict())
    return metrics


def _run_one(pipeline_args, base_args, run_dir):
    """Run inference + validate for one configuration. Returns (status, eval_dict)."""
    prefix = base_args["out_prefix"]
    platform = base_args["platform"]
    ref_label = base_args["ref_label"]
    sample = base_args["sample"]
    ann_file = os.path.join(run_dir, f"{prefix}.annotations.tsv")
    eval_file = os.path.join(run_dir, f"{prefix}.{platform}.evaluation.tsv")

    if not pipeline_args["force"] and os.path.isfile(ann_file):
        logging.info(f"SKIP (exists): {run_dir}")
        return "SKIPPED", _eval_metrics(
            ann_file, eval_file, ref_label, platform, prefix, run_dir
        )

    inf_args = {**pipeline_args, **base_args}
    inf_args["out_dir"] = run_dir
    check_arguments_inference(inf_args)
    os.makedirs(run_dir, exist_ok=True)

    logging.info(f"RUN: {run_dir}")
    run_inference(inf_args)

    metrics = {"log_likelihood": _read_loglik(run_dir, prefix)}
    if os.path.isfile(ann_file):
        anns = pd.read_table(ann_file)
        qry_labels = [
            c
            for c in anns.columns
            if c.endswith("_label") and not c.endswith("-refined")
        ]
        if qry_labels and ref_label in anns.columns:
            qry_label = qry_labels[0]
            tumor_post = "tumor_purity" if platform == "spatial" else "tumor"
            metrics.update(_eval_subset(anns, qry_label, ref_label, tumor_post))
            logging.info(pd.DataFrame([metrics]).to_string(index=False))

    val_dir = os.path.join(run_dir, "validate")
    if not os.path.isfile(os.path.join(val_dir, f"{sample}.evaluation.tsv")):
        val_args = {**pipeline_args}
        val_args["sample"] = sample
        val_args["processed_data"] = os.path.join(run_dir, "processed_data")
        val_args["pred_labels"] = ann_file
        val_args["pred_label"] = "copytyping_label"
        val_args["ref_labels"] = inf_args["cell_type"]
        val_args["ref_label"] = ref_label
        val_args["method"] = inf_args["method"]
        val_args["out_dir"] = val_dir
        val_args["h5ad"] = inf_args["gex_h5ad"]
        run_validate(val_args)
    return "OK", metrics


def run(args):
    args = normalize_args(args)

    panel_tsv = args["panel_tsv"]
    out_dir = args["out_dir"]
    platform_filter = args["platform_filter"]
    sol_pattern = args["sol_pattern"]

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

    sample_filter = args["samples"]
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
            row_args = {**base_args, "solfile": solfile}

            ploidy = row["PLOIDY"]
            clone = row["CLONE"]
            run_dir = os.path.join(
                out_dir,
                row_args["sample"],
                row_args["platform"],
                f"{ploidy}_n{clone}_{tag}",
            )

            status, metrics = _run_one(args, row_args, run_dir)

            assay_types = []
            if row_args["gex_dir"]:
                assay_types.append("gex")
            if row_args["atac_dir"]:
                assay_types.append("atac")
            info = dict(row)
            info["OUT_PREFIX"] = os.path.relpath(run_dir, out_dir)
            info["SOLFILE"] = solfile or ""
            info["ASSAY_TYPES"] = "+".join(assay_types) if assay_types else ""
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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pipeline")
    add_arguments_pipeline(parser)
    args = normalize_args(parser.parse_args(argv))
    setup_logging(args)
    run(args)


if __name__ == "__main__":
    main()
