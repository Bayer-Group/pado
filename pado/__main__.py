import argparse
import functools
import sys
from pathlib import Path

from pado._cli import subcommand, argument, DirectoryType
from pado.dataset import PadoDataset


parser = argparse.ArgumentParser(
    prog="python -m pado" if Path(sys.argv[0]).name == "__main__.py" else None,
    description="""\
 ██████╗  █████╗ ██████╗  ██████╗ 
 ██╔══██╗██╔══██╗██╔══██╗██╔═══██╗
 ██████╔╝███████║██║  ██║██║   ██║
 ██╔═══╝ ██╔══██║██║  ██║██║   ██║
 ██║     ██║  ██║██████╔╝╚██████╔╝
 ╚═╝     ╚═╝  ╚═╝╚═════╝  ╚═════╝ """,
    epilog="#### [PA]thological [D]ata [O]bsession ####",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
subparsers = parser.add_subparsers(dest="cmd", title="pado command")
subcommand = functools.partial(subcommand, parent=subparsers)
parser.add_argument('--version', action='store_true', help="print version")


def main(commandline=None):
    """main command line argument handling"""
    args = parser.parse_args(commandline)
    if args.cmd is None:
        if args.version:
            from pado import __version__
            print(f"{__version__}")
        else:
            parser.print_help()
        return 0
    else:
        return args.cmd_func(args)


@subcommand(
    argument('dataset_path', help="path to pado dataset"),
)
def info_(args, subparser):
    """show pado dataset information"""
    print("???")
    if not args.dataset_path:
        print(subparser.format_help())
        return 0

    try:
        dataset_path = PadoDataset(args.dataset_path, mode="r")
    except FileNotFoundError:
        print(dataset_path)
        return -1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
