import os
import sys
import pandas as pd
import logging

from collections import OrderedDict


ALL_PLATFORMS = ["single_cell", "spatial"]
SPATIAL_PLATFORMS = {"spatial"}

TUMOR_LABELS = {"Tumor_cell", "tumor", "Tumor"}
TUMOR_PREFIXES = ("tumor", "clone", "clone_")

INVALID_LABELS = {"Doublet", "doublet", "Unknown", "NA"}
NA_CELLTYPE = {"Unknown", "NA"}


def is_tumor_label(label: str) -> bool:
    return label.lower().startswith(TUMOR_PREFIXES) or label in TUMOR_LABELS


def is_normal_label(label: str) -> bool:
    return not is_tumor_label(label) and label not in INVALID_LABELS


def get_chr2ord(ch):
    chr2ord = {}
    for i in range(1, 23):
        chr2ord[f"{ch}{i}"] = i
    chr2ord[f"{ch}X"] = 23
    chr2ord[f"{ch}Y"] = 24
    chr2ord[f"{ch}M"] = 25
    return chr2ord


def sort_chroms(chromosomes: list):
    assert len(chromosomes) != 0
    ch = "chr" if str(chromosomes[0]).startswith("chr") else ""
    chr2ord = get_chr2ord(ch)
    return sorted(chromosomes, key=lambda x: chr2ord[x])


def get_chr_sizes(sz_file: str):
    chr_sizes = OrderedDict()
    with open(sz_file, "r") as rfd:
        for line in rfd.readlines():
            ch, sizes = line.strip().split()
            chr_sizes[ch] = int(sizes)
        rfd.close()
    return chr_sizes


def sort_df_chr(df: pd.DataFrame, ch="#CHR", pos="POS"):
    chs = sort_chroms(df[ch].unique().tolist())
    df[ch] = pd.Categorical(df[ch], categories=chs, ordered=True)
    df.sort_values(by=[ch, pos], inplace=True, ignore_index=True)
    return df


def read_seg_ucn_file(seg_ucn_file: str):
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


def setup_logging(args) -> None:
    v = (
        getattr(args, "verbosity", None)
        or (args.get("verbosity") if isinstance(args, dict) else 0)
        or 0
    )
    level = logging.DEBUG if v >= 2 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )


def log_arguments(args) -> None:
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


def save_cnp_profile(sx_data, out_file):
    """Save input CNP profile as TSV with columns #CHR, START, END, cn_<clone>."""
    cnp_df = sx_data.cnv_blocks[["#CHR", "START", "END"]].copy()
    for ki, clone_name in enumerate(sx_data.clones):
        cnp_df[f"cn_{clone_name}"] = [
            f"{a}|{b}" for a, b in zip(sx_data.A[:, ki], sx_data.B[:, ki])
        ]
    cnp_df.to_csv(out_file, sep="\t", index=False)
