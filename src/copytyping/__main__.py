import argparse

from copytyping.copytyping_parser import (
    add_arguments_cnphmm,
    add_arguments_inference,
)
from copytyping.cnphmm_inference.inference import run as copytyping_cnphmm
from copytyping.inference.inference import run as copytyping_inference
from copytyping.utils import log_arguments, setup_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="copytyping")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_inf = subparsers.add_parser("inference", help="run inference on one sample")
    add_arguments_inference(p_inf)
    p_inf.set_defaults(func=copytyping_inference)

    p_cnphmm = subparsers.add_parser(
        "cnphmm_copytyping", help="run factorial CNP-HMM copy-typing on one sample"
    )
    add_arguments_cnphmm(p_cnphmm)
    p_cnphmm.set_defaults(func=copytyping_cnphmm)

    args = parser.parse_args(argv)
    setup_logging(args)
    log_arguments(args)
    args.func(args)


if __name__ == "__main__":
    main()
