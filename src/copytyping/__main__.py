# src/copytyping/__main__.py
import os
import sys
import logging
import yaml
import argparse

from copytyping.inference.inference import run as copytyping_inference

from copytyping.copytyping_parser import *
from copytyping.utils import setup_logging


def main(argv=None):
    parser = argparse.ArgumentParser(prog="copytyping")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_inf = subparsers.add_parser("inference", help="run inference")
    add_arguments_inference(p_inf)
    p_inf.set_defaults(func=copytyping_inference)

    args = parser.parse_args(argv)
    setup_logging(args)
    logging.info(f"parsed arguments: {args}")
    args.func(args)


if __name__ == "__main__":
    main()
