from __future__ import annotations

from collections import namedtuple

from typer.testing import CliRunner

from pado.__main__ import cli

_Output = namedtuple("_Output", "return_code stdout")

runner = CliRunner()


def test_no_args():
    result = runner.invoke(cli, [])
    assert result.exit_code == 0


def test_version():
    from pado import __version__

    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert result.stdout == f"{__version__}\n"


def test_info_cmd(tmpdir):
    # help
    dataset_path = tmpdir.mkdir("not_a_dataset")
    result = runner.invoke(cli, ["info", str(dataset_path)])
    assert result.exit_code == 1
    assert "ERROR" in result.stdout
    assert "not_a_dataset" in result.stdout
