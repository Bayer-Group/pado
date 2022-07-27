from __future__ import annotations

import os

from typer.testing import CliRunner

from pado.__main__ import cli
from pado.dataset import PadoDataset
from pado.mock import mock_dataset
from pado.settings import dataset_registry

runner = CliRunner()


def test_no_args():
    result = runner.invoke(cli, [])
    assert result.exit_code == 0


def test_version():
    from pado import __version__

    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert result.stdout == f"{__version__}\n"


def test_cmd_info_error(tmpdir):
    dataset_path = tmpdir.mkdir("not_a_dataset")
    result = runner.invoke(cli, ["info", str(dataset_path)])
    assert result.exit_code == 1
    assert "ERROR" in result.stdout
    assert "not_a_dataset" in result.stdout


def test_cmd_info(mock_dataset_path):
    result = runner.invoke(cli, ["info", mock_dataset_path])
    assert result.exit_code == 0
    assert "SUMMARY" in result.stdout


def test_cmd_stores(mock_dataset_path):
    result = runner.invoke(cli, ["stores", mock_dataset_path])
    assert result.exit_code == 0
    assert "Dataset Store" in result.stdout


def test_cmd_copy(registry, tmp_path):
    pth0 = tmp_path.joinpath("d0")
    pth1 = tmp_path.joinpath("d1")
    mock_dataset(pth0)
    with dataset_registry() as dct:
        dct["ds0"] = os.fspath(pth0)
        dct["ds1"] = os.fspath(pth1)

    result = runner.invoke(
        cli,
        [
            "copy",
            "--src",
            "ds0",
            "--dst",
            "ds1",
            "--image-providers",
            "--metadata-providers",
            "--annotation-providers",
        ],
    )
    assert "copying: ImageProvider" in result.stdout
    assert "copying: MetadataProvider" in result.stdout
    assert "copying: AnnotationProvider" in result.stdout
    assert result.exit_code == 0

    ds_dst = PadoDataset(pth1)
    assert len(ds_dst.index) > 0


def test_cmd_copy_error(registry, tmp_path):
    # src dataset not there
    result = runner.invoke(cli, ["copy", "--src", "not-there", "--dst", "ds1"])
    assert result.exit_code == 1
    assert "not-there" in result.stdout
    # dst dataset not there
    with dataset_registry() as dct:
        dct["ds0"] = "somepath"
    result = runner.invoke(cli, ["copy", "--src", "ds0", "--dst", "not-there"])
    assert result.exit_code == 1
    assert "not-there" in result.stdout


def test_cmd_ops_list_ids(mock_dataset_path):
    result = runner.invoke(cli, ["ops", "list-ids", mock_dataset_path])
    assert result.exit_code == 0
    assert "ImageId" in result.stdout
    result = runner.invoke(cli, ["ops", "list-ids", mock_dataset_path, "--as-path"])
    assert result.exit_code == 0
    assert "ImageId" not in result.stdout
