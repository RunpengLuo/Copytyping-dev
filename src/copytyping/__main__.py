# src/copytyping/__main__.py
import os
import sys
import yaml
import argparse

from copytyping.combine_counts.combine_counts import run as copytyping_combine_counts
from copytyping.inference.inference import run as copytyping_inference

from copytyping.copytyping_parser import *


def copytyping_pipeline(args):
    return


def main(argv=None):
    parser = argparse.ArgumentParser(prog="copytyping")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_run = subparsers.add_parser("run", help="run pipeline from YAML config")
    p_run.add_argument("config", default="copytyping.yaml")
    p_run.set_defaults(func=copytyping_pipeline)

    p_bin = subparsers.add_parser("combine-counts", help="run combine-counts")
    add_arguments_combine_counts(p_bin)
    p_bin.set_defaults(func=copytyping_combine_counts)

    p_inf = subparsers.add_parser("inference", help="run inference")
    add_arguments_inference(p_inf)
    p_inf.set_defaults(func=copytyping_inference)

    args = parser.parse_args(argv)
    print(f"parsed arguments: {args}")
    args.func(args)


if __name__ == "__main__":
    main()
