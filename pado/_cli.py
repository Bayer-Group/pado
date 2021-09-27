import platform
import sys
import textwrap
from argparse import ArgumentTypeError
from collections import defaultdict
from functools import partial
from pathlib import Path


# -- argparse improvements ---------------------------------------------

def subcommand(*arguments, parent):
    """decorator helper for commandline"""
    def decorator(func):
        fn = func.__name__.rstrip('_')
        started_via_m = Path(sys.argv[0]).name == "__main__.py"
        subparser = parent.add_parser(
            name=fn,
            prog=f"python -m pado {fn}" if started_via_m else f"pado {fn}",
            help=func.__doc__,
        )
        for args, kwargs in arguments:
            subparser.add_argument(*args, **kwargs)
        subparser.set_defaults(cmd_func=partial(func, subparser=subparser))
        return func
    return decorator


def argument(*args, **kwargs):
    """argument helper for subcommand"""
    return args, kwargs


class DirectoryType:
    """Directory parsing for argparse"""
    def __call__(self, cmd_input: str):
        p = Path(cmd_input)
        if p.is_dir():
            return p
        raise ArgumentTypeError(f"'{cmd_input}' is not a directory")


# -- commands -----------------------------------------------------------------

def cli_info_cmd(dataset_path):
    from pado.dataset import PadoDataset

    ds = PadoDataset(dataset_path, mode="r")

    return ds.describe(output_format='json')
