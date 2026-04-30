import argparse

from copytyping.copytyping_parser import (
    add_arguments_inference,
    add_arguments_pipeline,
    add_arguments_validate,
)
from copytyping.inference.inference import run as copytyping_inference
from copytyping.pipeline import run as copytyping_pipeline
from copytyping.utils import log_arguments, setup_logging
from copytyping.validation.validate import run as copytyping_validate


def main(argv=None):
    parser = argparse.ArgumentParser(prog="copytyping")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_inf = subparsers.add_parser("inference", help="run inference on one sample")
    add_arguments_inference(p_inf)
    p_inf.set_defaults(func=copytyping_inference)

    p_pipe = subparsers.add_parser("run_pipeline", help="batch run from a panel TSV")
    add_arguments_pipeline(p_pipe)
    p_pipe.set_defaults(func=copytyping_pipeline)

    p_val = subparsers.add_parser("validate", help="evaluate labels against reference")
    add_arguments_validate(p_val)
    p_val.set_defaults(func=copytyping_validate)

    args = parser.parse_args(argv)
    setup_logging(args)
    log_arguments(args)
    args.func(args)


if __name__ == "__main__":
    main()
