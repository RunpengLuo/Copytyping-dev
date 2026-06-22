import argparse

from copytyping.copytyping_parser import add_arguments_inference
from copytyping.inference.inference import run as copytyping_inference
from copytyping.utils import log_arguments, setup_logging


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="copytyping")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_inf = subparsers.add_parser("inference", help="run inference on one sample")
    add_arguments_inference(p_inf)
    p_inf.set_defaults(func=copytyping_inference)

    args = parser.parse_args(argv)
    setup_logging(args)
    log_arguments(args)
    args.func(args)


if __name__ == "__main__":
    main()
