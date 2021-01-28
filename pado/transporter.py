import argparse
import functools
import logging
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from textwrap import dedent

from appdirs import user_config_dir
import toml

from pado._cli import argument, subcommand
from pado._logging import get_logger

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
parser.add_argument('--version', action='store_true', help="print version")
parser.add_argument('-v', '--verbose', action='store_true', help="print more info")

# logging objects
logger = get_logger(__name__)

# might need to refer to a specific ssh
SSH_EXECUTABLE = "ssh"
RSYNC_EXECUTABLE = "rsync"


def _get_default_target_and_tunnel():
    config_dir = user_config_dir("pado.transporter", version="0.1")
    config_file = Path(config_dir) / "pado-transporter-config.toml"
    with config_file.open("r") as f:
        config = toml.load(f)
    target = config['target_host']
    tunnel = config.get('tunnel_host', None)
    return target, tunnel


def _make_ssh_command(remote, *cmd):
    """creates the command list for testing passwordless login on a remote"""
    if not cmd:
        raise ValueError("command required")
    cmd_list = [
        SSH_EXECUTABLE,
        "-o", "PasswordAuthentication=no",
        "-o", "BatchMode=yes",
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
        msg = dedent("""\
            SSH ERROR: Could not access the requested host '{target}' without password via '{tunnel}'
            SUGGESTED FIX: add your public ssh key from '{tunnel}' to your remote machine '{target}'
        """)
        print(msg.format(target=target, tunnel=tunnel))

    else:
        msg = dedent("""\
            SSH ERROR: Could not access the requested host '{target}' without password
            SUGGESTED FIX: add your public ssh key to your remote machine '{target}'
        """)
        print(msg.format(target=tunnel))


class _CommandIter:
    """iterate over the stdout of a running subprocess"""

    def __init__(self, command, poll_timeout=0.5, max_timeout=10.0):
        assert isinstance(command, list)
        self.command = command
        self.return_code = None
        self.max_timeout_counter = max(1, int(max_timeout / poll_timeout))
        self.poll_timeout = float(poll_timeout)

    def __iter__(self):
        logger.info(f"rsync: {self.command!r}")
        process = subprocess.Popen(self.command, stdout=subprocess.PIPE, env=os.environ)
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
            process.wait()


def _make_rsync_cmd(*options, remote_shell=None):
    """creates the command list for running rsync commands"""
    cmd = [RSYNC_EXECUTABLE]
    if remote_shell:
        assert '"' not in remote_shell, "double quote character in the remote_shell?"
        cmd.extend(['-e', remote_shell])
    for option in options:
        assert option.startswith("-"), f"option '{option}' does not start with '-'"
        cmd.append(option)
    return cmd


def _make_remote_shell_option(target, local_ssh_cmd=SSH_EXECUTABLE, tunnel_ssh_cmd="ssh"):
    return f"{local_ssh_cmd} -A {target} {tunnel_ssh_cmd}"


def _make_remote_path(remote, path):
    return f'{remote}:"{os.fspath(path)}"'


def list_files_on_remote(path, *, target, tunnel=None, recursive=True):
    """list files on the remote"""

    options = ["-avz", "--list-only"]
    if not recursive:
        options.append("--no-recursive")

    remote_shell = _make_remote_shell_option(tunnel) if tunnel else None
    remote_location = _make_remote_path(target, path)

    cmd = _make_rsync_cmd(*options, remote_shell=remote_shell)
    cmd.append(remote_location)

    it = _CommandIter(cmd)
    for line in it:
        print(line)
    if it.return_code != 0:
        raise RuntimeError("rsync command failed with return_code:", it.return_code)


def main(argv=None):
    global parser
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARN)


    try:
        target, tunnel = _get_default_target_and_tunnel()
    except FileNotFoundError:
        print("ERROR: please provide target or configure via `config` subcommand")
        return -1


if __name__ == "__main__":
    # main(sys.argv)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"{parser.prog} terminated on user request")
        sys.exit(-1)
    except Exception:
        traceback.print_exc()
        sys.exit(-1)
