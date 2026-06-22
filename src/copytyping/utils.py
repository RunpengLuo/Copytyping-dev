import argparse
import logging
import os
import sys
from collections import OrderedDict
from importlib.resources import files

import pandas as pd
import yaml


##################################################
# config / runtime defaults
##################################################


def load_defaults():
    """Load packaged tuning defaults from src/copytyping/copytyping.yaml."""
    text = files("copytyping").joinpath("copytyping.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def normalize_args(args: argparse.Namespace | dict):
    """Convert Namespace→dict if needed and merge YAML defaults under it (args win).

    Argparse parsers use ``default=argparse.SUPPRESS`` so unparsed flags are
    absent from the Namespace; this fills them from the bundled yaml.
    """
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    return {**load_defaults(), **args}


##################################################
# constants
##################################################

ALL_PLATFORMS = ["single_cell", "spatial", "multiome"]
SPATIAL_PLATFORMS = {"spatial"}

TUMOR_LABELS = {"Tumor_cell", "tumor", "Tumor"}
TUMOR_PREFIXES = ("tumor", "clone", "clone_")

INVALID_LABELS = {"Doublet", "doublet", "Unknown", "NA"}
NA_CELLTYPE = {"Unknown", "NA"}


##################################################
# label predicates
##################################################


def is_tumor_label(label: str):
    return label.lower().startswith(TUMOR_PREFIXES) or label in TUMOR_LABELS


def is_normal_label(label: str):
    return not is_tumor_label(label) and label not in INVALID_LABELS


##################################################
# chromosome utilities
##################################################


def chrom_sort_key(chrom: str | int):
    """Genomic sort key: autosomes numerically (any count), then X, Y, M, then unknowns.

    Accepts ``chr``-prefixed or bare names, as str or int.
    """
    core = str(chrom)
    if core.lower().startswith("chr"):
        core = core[3:]
    if core.isdigit():
        return (0, int(core), "")
    special = {"X": 1, "Y": 2, "M": 3, "MT": 3}
    if core.upper() in special:
        return (1, special[core.upper()], "")
    return (2, 0, core)


def sort_chroms(chromosomes: list):
    """Sort chromosome names in genomic order. See :func:`chrom_sort_key`."""
    assert len(chromosomes) != 0
    return sorted((str(c) for c in chromosomes), key=chrom_sort_key)


def get_chr_sizes(sz_file: str):
    chr_sizes = OrderedDict()
    with open(sz_file, "r") as rfd:
        for line in rfd.readlines():
            ch, sizes = line.strip().split()
            chr_sizes[ch] = int(sizes)
        rfd.close()
    return chr_sizes


def sort_df_chr(df: pd.DataFrame, ch: str = "#CHR", pos: str = "POS"):
    chs = sort_chroms(df[ch].unique().tolist())
    df[ch] = pd.Categorical(df[ch], categories=chs, ordered=True)
    df.sort_values(by=[ch, pos], inplace=True, ignore_index=True)
    return df


##################################################
# bulk file readers
##################################################


def read_seg_ucn_file(
    seg_ucn_file: str,
):
    segs_df = pd.read_table(seg_ucn_file, sep="\t")
    segs_df = sort_df_chr(segs_df, pos="START")

    n_clones = len([cname for cname in segs_df.columns if cname.startswith("cn_")])
    clones = ["normal"] + [f"clone{c}" for c in range(1, n_clones)]
    clone_props = segs_df[[f"u_{clone}" for clone in clones]].iloc[0].tolist()
    segs_df.loc[:, "CNP"] = segs_df.apply(
        func=lambda r: ";".join(r[f"cn_{c}"] for c in clones), axis=1
    )
    segs_df["PROPS"] = ";".join([str(p) for p in clone_props])
    return segs_df, clones, clone_props


def read_whitelist_segments(bed_file: str):
    wl_fragments = pd.read_table(
        bed_file,
        sep="\t",
        header=None,
        names=["#CHR", "START", "END", "NAME"],
    )
    return wl_fragments


##################################################
# logging
##################################################


def setup_logging(args: argparse.Namespace | dict):
    v = args["verbosity"] if isinstance(args, dict) else args.verbosity
    level = logging.DEBUG if v >= 2 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )


def log_arguments(args: argparse.Namespace | dict):
    d = vars(args) if hasattr(args, "__dict__") else args
    lines = "\n".join(f"  {k}: {v}" for k, v in sorted(d.items()) if k != "func")
    logging.info(f"parsed arguments:\n{lines}")


def add_file_logging(out_dir: str, command: str = "copytyping"):
    """Attach a FileHandler to the root logger. Returns the handler for later removal."""
    os.makedirs(out_dir, exist_ok=True)
    level = (
        logging.root.level if logging.root.level != logging.WARNING else logging.INFO
    )
    fh = logging.FileHandler(os.path.join(out_dir, f"{command}.log"), mode="w")
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.root.addHandler(fh)
    if logging.root.level > level:
        logging.root.setLevel(level)
    return fh
