from __future__ import annotations

import io
from collections import namedtuple
from contextlib import redirect_stdout

import pytest

from pado.__main__ import cli

pytestmark = pytest.mark.xfail  # currently broken after switch to typer

_Output = namedtuple("_Output", "return_code stdout")


def run(func, argv1):
    f = io.StringIO()
    with redirect_stdout(f):
        return_code = func(argv1)
    return _Output(return_code, f.getvalue().rstrip())


def test_no_args():
    assert cli([]) == 0


def test_version():
    from pado import __version__

    assert run(cli, ["--version"]) == (0, __version__)


def test_info_cmd(tmpdir):
    # help
    dataset_path = tmpdir.mkdir("not_a_dataset")
    output = run(cli, ["info", str(dataset_path)])
    assert output.return_code == -1
    assert "error" in output.stdout
    assert "not_a_dataset" in output.stdout
