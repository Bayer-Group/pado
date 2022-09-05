from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import re
import subprocess
import sys
import traceback
from argparse import ArgumentTypeError
from collections import defaultdict
from functools import partial
from pathlib import Path
from textwrap import dedent

import toml
from appdirs import user_config_dir


def subcommand(*arguments, parent):
    """decorator helper for commandline"""

    def decorator(func):
        fn = func.__name__.rstrip("_")
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


# argument parsing objects
parser = argparse.ArgumentParser(
    prog="python -m pado.transporter" if Path(sys.argv[0]).name == __file__ else None,
    description=r"""#### PADO.TRANSPORTER ####
   _                                              _
 _| |_  ____ _____ ____   ___ ____   ___   ____ _| |_ _____  ____
(_   _)/ ___|____ |  _ \ /___)  _ \ / _ \ / ___|_   _) ___ |/ ___)
  | |_| |   / ___ | | | |___ | |_| | |_| | |     | |_| ____| |
   \__)_|   \_____|_| |_(___/|  __/ \___/|_|      \__)_____)_|
                             |_|                               """,
    epilog="#### [PA]thological [D]ata [O]bsession ####",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
subparsers = parser.add_subparsers(dest="cmd", title="pado.transporter command")
subcommand = functools.partial(subcommand, parent=subparsers)
parser.add_argument("--version", action="store_true", help="print version")
parser.add_argument("-v", "--verbose", action="store_true", help="print more info")
parser.add_argument("--target", metavar=("remote",), default=None, help="target remote")
parser.add_argument("--tunnel", metavar=("tunnel",), default=None, help="tunnel remote")
parser.add_argument("--root", action="store_true", help="reference from '/' on remote")

# logging objects
logger = logging.getLogger(__name__)

# might need to refer to a specific ssh
SSH_EXECUTABLE = "ssh"
RSYNC_EXECUTABLE = "rsync"


def _get_default_config_file():
    config_dir = user_config_dir("pado.transporter", version="0.1")
    config_file = Path(config_dir) / "pado-transporter-config.toml"
    return config_file


def _get_default_config():
    config_file = _get_default_config_file()
    with config_file.open("r") as f:
        return toml.load(f)


def _set_default_config(obj):
    config_file = _get_default_config_file()
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with config_file.open("w") as f:
        toml.dump(obj, f)


def _make_ssh_command(remote, *cmd):
    """creates the command list for testing passwordless login on a remote"""
    if not cmd:
        raise ValueError("command required")
    cmd_list = [
        SSH_EXECUTABLE,
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "BatchMode=yes",
        remote,
        "--",
    ]
    cmd_list.extend(cmd)
    return cmd_list


def check_ssh_no_password(target, *, tunnel=None):
    """verify if a passwordless remote connection can be established (tunnel optional)"""

    cmd = _make_ssh_command(target, "exit")

    if tunnel is not None:
        cmd = _make_ssh_command(tunnel, *cmd)

    try:
        subprocess.run(cmd, env=os.environ, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        return False
    else:
        return True


def cli_check_ssh_no_password(target, tunnel):
    if check_ssh_no_password(target=target, tunnel=tunnel):
        print(f"connection to '{target}' established via '{tunnel}'")

    elif check_ssh_no_password(target=tunnel):
        msg = dedent(
            """\
            SSH ERROR: Could not access the requested host '{target}' without password via '{tunnel}'
            SUGGESTED FIX: add your public ssh key from '{tunnel}' to your remote machine '{target}'
        """
        )
        print(msg.format(target=target, tunnel=tunnel))

    else:
        msg = dedent(
            """\
            SSH ERROR: Could not access the requested host '{target}' without password
            SUGGESTED FIX: add your public ssh key to your remote machine '{target}'
        """
        )
        print(msg.format(target=tunnel))


class _CommandIter:
    """iterate over the stdout of a running subprocess"""

    def __init__(self, command, poll_timeout=0.5, max_timeout=10.0):
        if not isinstance(command, list):
            raise TypeError(f"command expected list, got {type(command).__name__!r}")
        self.command = command
        self.return_code = None
        self.max_timeout_counter = max(1, int(max_timeout / poll_timeout))
        self.poll_timeout = float(poll_timeout)
        self.stderr = None

    def __iter__(self):
        logger.info(f"rsync: {subprocess.list2cmdline(self.command)}")
        process = subprocess.Popen(
            self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ
        )
        process_running = True
        try:
            while process_running:
                output = process.stdout.readline()
                if output:
                    yield output.rstrip(b"\n").decode()

                else:
                    self.return_code = process.poll()
                    if self.return_code is not None:
                        break
        except KeyboardInterrupt:
            process.kill()
            raise
        finally:
            self.stderr = process.stderr.read().decode()
            logger.debug(self.stderr)
            process.wait()


def _make_rsync_cmd(*options, remote_shell=None):
    """creates the command list for running rsync commands"""
    cmd = [RSYNC_EXECUTABLE]
    if remote_shell:
        if '"' in remote_shell:
            raise ValueError("double quote character in the remote_shell?")
        cmd.extend(["-e", remote_shell])
    for option in options:
        if not option.startswith("-"):
            raise ValueError(f"option '{option}' does not start with '-'")
        cmd.append(option)
    return cmd


def _make_remote_shell_option(
    target, local_ssh_cmd=SSH_EXECUTABLE, tunnel_ssh_cmd="ssh"
):
    return f"{local_ssh_cmd} -A {target} {tunnel_ssh_cmd}"


def _make_remote_path(remote, path):
    return f'{remote}:"{os.fspath(path)}"'


def list_files_on_remote(
    path,
    *,
    target,
    tunnel=None,
    recursive=True,
    long=False,
    regex=None,
    image_id_json=None,
):
    """list files on the remote"""

    options = ["-avz", "--list-only"]
    if not recursive:
        options.append("--no-recursive")

    remote_shell = _make_remote_shell_option(tunnel) if tunnel else None
    remote_location = _make_remote_path(target, path)

    cmd = _make_rsync_cmd(*options, remote_shell=remote_shell)
    cmd.append(remote_location)

    if regex is not None:
        regex = re.compile(regex)

    if image_id_json is not None:
        _image_id_map = defaultdict(set)
        for record in image_id_json:
            iid = tuple(record["image_id"])
            _image_id_map[len(iid)].add(iid)
        image_id_json = dict(sorted(_image_id_map.items()))

    cmd_iter = _CommandIter(cmd)
    it = iter(cmd_iter)
    try:
        line0 = next(it)
    except StopIteration:
        print(cmd_iter.stderr, file=sys.stderr)
        raise RuntimeError(
            "rsync command failed with return_code:", cmd_iter.return_code
        )

    status_msgs = {"receiving file list ... done", "receiving incremental file list"}
    if line0 not in status_msgs:
        raise RuntimeError(f"received: '{line0!r}'")
    # parse files
    for line in it:
        try:
            permission, size, date, mtime, filename = line.split(maxsplit=4)
        except ValueError:
            break  # we reached the end of the file list

        if regex and not regex.search(filename):
            continue

        if image_id_json:
            fn_parts = Path(filename).parts
            for length, image_ids in image_id_json.items():
                if fn_parts[-length:] in image_ids:
                    break
            else:
                continue

        if long:
            print(line)
        else:
            print(filename)

    # parse summary
    _, _ = it  # ignore the two summary lines for now

    if cmd_iter.return_code != 0:
        raise RuntimeError(
            "rsync command failed with return_code:", cmd_iter.return_code
        )


def _make_path(path, *, base):
    if base is None:
        base = "/"
    return os.path.join(base, path)


def pull_files_from_remote(*, local_path, remote_base_path, files, target, tunnel=None):
    """pull files from the remote"""

    options = [
        "-avz",
        "--ignore-existing",
        "--partial",
        "--progress",
        f"--files-from={files}",
    ]

    remote_shell = _make_remote_shell_option(tunnel) if tunnel else None
    remote_location = _make_remote_path(target, remote_base_path)
    cmd = _make_rsync_cmd(*options, remote_shell=remote_shell)
    cmd.append(remote_location)
    cmd.append(local_path)

    proc = subprocess.run(cmd, env=os.environ)
    if proc.returncode != 0:
        raise RuntimeError("rsync command failed with return_code:", proc.returncode)


# -- commands ---------------------------------------------------------


def main(argv=None):
    global parser
    args = parser.parse_args(argv)

    if args.cmd is None:
        if args.version:
            # noinspection PyProtectedMember
            from pado import __version__

            print(f"{__version__}")
        else:
            parser.print_help()
        return 0

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARN)

    if args.target is None and args.cmd != "config":
        try:
            cfg = _get_default_config()
            target = cfg["target"]
            tunnel = cfg.get("tunnel", None)
        except (FileNotFoundError, KeyError):
            print("ERROR: please provide target or configure via `config` subcommand")
            return -1
        else:
            args.target = target
            args.tunnel = tunnel

    # base_path can only be set via config and disabled via --root
    if args.root and args.cmd != "config":
        args.base_path = None
    elif args.cmd != "config":
        try:
            cfg = _get_default_config()
            args.base_path = cfg["base_path"]
        except (FileNotFoundError, KeyError):
            print(
                "ERROR: please configure base_path via `config` subcommand or allow root"
            )
            return -1

    return args.cmd_func(args)


@subcommand(
    argument("--show", action="store_true", help="show configuration"),
    argument("--target", help="set default target host"),
    argument("--tunnel", help="set default tunnel host"),
    argument("--base-path", help="set default base path"),
)
def config(args, subparser):
    """configure default settings"""
    try:
        _config = _get_default_config()
    except FileNotFoundError:
        _config = {}

    def _show_config(cfg):
        if not cfg:
            print("no pado-transporter configuration set")
        else:
            print("pado-transporter configuration:")
            for key, value in sorted(cfg.items()):
                print(key, "=", value)

    if args.show:
        _show_config(_config)
        return 0

    new_config = _config.copy()
    if args.target:
        new_config["target"] = args.target
    if args.tunnel:
        new_config["tunnel"] = args.tunnel
    if args.base_path:
        if not os.path.isabs(args.base_path):
            logger.warning("--base-path requires an absolute path. please verify")
        new_config["base_path"] = os.path.abspath(args.base_path)

    if new_config != _config:
        _set_default_config(new_config)

    _show_config(new_config)
    return 0


@subcommand(
    argument("-r", "--recursive", action="store_true", help="recurse subdirectories"),
    argument("-l", "--long", action="store_true", help="list details"),
    argument("--match", help="match regex"),
    argument("--select-image-id-json", help="matches defined in json file"),
    argument("path", help="base directory to start ls"),
)
def ls(args, subparser):
    """list files on remote"""
    image_id_json = None
    if args.select_image_id_json:
        with open(args.select_image_id_json) as f:
            image_id_json = json.load(f)

    list_files_on_remote(
        path=_make_path(args.path, base=args.base_path),
        target=args.target,
        tunnel=args.tunnel,
        recursive=args.recursive,
        long=args.long,
        regex=args.match,
        image_id_json=image_id_json,
    )


@subcommand()
def remote_shell(args, subparser):
    """open a remote shell on the target"""
    cmd_args = [SSH_EXECUTABLE]
    if args.tunnel:
        cmd_args.extend(["-J", args.tunnel])
    cmd_args.append(args.target)
    if args.base_path:
        cmd_args.extend(["-t", f"cd '{args.base_path}'; exec $SHELL --login"])
    # drop to remote shell
    os.execvp(SSH_EXECUTABLE, cmd_args)


@subcommand(
    argument("--files-from", required=True, help="text file with files to copy"),
    argument("dest_local", help="local destination path"),
)
def pull(args, subparser):
    """pull files from remote"""
    pull_files_from_remote(
        local_path=args.dest_local,
        remote_base_path=args.base_path,
        files=args.files_from,
        target=args.target,
        tunnel=args.tunnel,
    )


def cli_main():
    # noinspection PyBroadException
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"{parser.prog} terminated on user request")
        sys.exit(-1)
    except Exception:
        traceback.print_exc()
        sys.exit(-1)


if __name__ == "__main__":
    cli_main()
