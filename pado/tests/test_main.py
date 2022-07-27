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


def test_cmd_info(mock_dataset_path):
    result = runner.invoke(cli, ["info", mock_dataset_path])
    assert result.exit_code == 0
    assert "SUMMARY" in result.stdout


def test_cmd_info_from_registry(mock_dataset_path, registry):
    with dataset_registry() as dct:
        dct["abc"] = mock_dataset_path
    result = runner.invoke(cli, ["info", "--name", "abc"])
    assert result.exit_code == 0
    assert "SUMMARY" in result.stdout


def test_cmd_info_error_no_dataset(tmpdir):
    dataset_path = tmpdir.mkdir("not_a_dataset")
    result = runner.invoke(cli, ["info", str(dataset_path)])
    assert result.exit_code == 1
    assert "ERROR" in result.stdout
    assert "not_a_dataset" in result.stdout


def test_cmd_info_error_unknown_dataset():
    result = runner.invoke(cli, ["info", "--name", "abc"])
    assert result.exit_code == 1


def test_cmd_info_error_missing_args():
    result = runner.invoke(cli, ["info", "--storage-options", "{}"])
    assert result.exit_code == 1


def test_cmd_info_error_too_many_args(tmp_path):
    result = runner.invoke(cli, ["info", "--name", "abc", str(tmp_path)])
    assert result.exit_code == 1


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


def test_cmd_ops_filter_ids(mock_dataset_path):
    ds = PadoDataset(mock_dataset_path)

    result = runner.invoke(
        cli, ["ops", "filter-ids", mock_dataset_path, "-i", ds.index[0].to_str()]
    )
    assert result.exit_code == 0
    assert len(result.stdout.splitlines()) == 1


def test_cmd_ops_filter_ids_provide_pathlikes(mock_dataset_path):
    ds = PadoDataset(mock_dataset_path)
    iid = os.path.join(*ds.index[0].parts)

    result = runner.invoke(cli, ["ops", "filter-ids", mock_dataset_path, "-i", iid])
    assert result.exit_code == 0
    assert len(result.stdout.splitlines()) == 1


def test_cmd_ops_filter_ids_provide_csv(mock_dataset_path, tmp_path):
    ds = PadoDataset(mock_dataset_path)
    # csv file with all columns
    csv_file = tmp_path.joinpath("iids.csv")
    csv_file.write_text(f"# header\n{os.path.join(*ds.index[0].parts)}\n")
    result = runner.invoke(
        cli, ["ops", "filter-ids", mock_dataset_path, "--csv", str(csv_file)]
    )
    assert result.exit_code == 0
    assert len(result.stdout.splitlines()) == 1


def test_cmd_ops_filter_ids_provide_csv_selected_column(mock_dataset_path, tmp_path):
    ds = PadoDataset(mock_dataset_path)
    # csv file with one selected column
    csv_file = tmp_path.joinpath("iids.csv")
    csv_file.write_text(f"c0,c1\nnono,{','.join(ds.index[0].parts)}\n")
    result = runner.invoke(
        cli, ["ops", "filter-ids", mock_dataset_path, "--csv", str(csv_file), "-c", "1"]
    )
    assert result.exit_code == 0
    assert len(result.stdout.splitlines()) == 1


def test_cmd_ops_filter_ids_provide_csv_selected_multi_column(
    mock_dataset_path, tmp_path
):
    ds = PadoDataset(mock_dataset_path)
    # csv file with multiple selected columns
    csv_file = tmp_path.joinpath("iids.csv")
    csv_file.write_text(f"c0,c1\nnono,{','.join(ds.index[0].parts)}\n")
    result = runner.invoke(
        cli,
        [
            "ops",
            "filter-ids",
            mock_dataset_path,
            "--csv",
            str(csv_file),
            "-c",
            "0",
            "-c",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert len(result.stdout.splitlines()) == 1


def test_cmd_ops_filter_ids_output_pathlikes(mock_dataset_path):
    ds = PadoDataset(mock_dataset_path)
    result = runner.invoke(
        cli,
        [
            "ops",
            "filter-ids",
            mock_dataset_path,
            "-i",
            ds.index[0].to_str(),
            "--as-path",
        ],
    )
    assert result.exit_code == 0
    assert "ImageId" not in result.stdout


def test_cmd_ops_filter_ids_write_output(mock_dataset_path, tmp_path):
    out_pth = tmp_path.joinpath("output")
    ds = PadoDataset(mock_dataset_path)

    result = runner.invoke(
        cli,
        [
            "ops",
            "filter-ids",
            mock_dataset_path,
            "-i",
            ds.index[0].to_str(),
            "--out",
            str(out_pth),
        ],
    )
    assert result.exit_code == 0

    dso = PadoDataset(out_pth)
    assert len(dso.index) == 1
    assert ds.index[0] in dso.index


def test_cmd_ops_filter_ids_error_noargs(mock_dataset_path):
    result = runner.invoke(cli, ["ops", "filter-ids", mock_dataset_path])
    assert result.exit_code == 1


def test_cmd_registry_add(registry, mock_dataset_path):
    result = runner.invoke(
        cli, ["registry", "add", "abc", mock_dataset_path, "--storage-options", "{}"]
    )
    assert result.exit_code == 0
    with dataset_registry() as dct:
        assert "abc" in dct


def test_cmd_registry_add_error_storage_options(registry, mock_dataset_path):
    result = runner.invoke(
        cli, ["registry", "add", "abc", mock_dataset_path, "--storage-options", "NONO"]
    )
    assert result.exit_code == 1


def test_cmd_registry_add_error_dataset(registry):
    result = runner.invoke(cli, ["registry", "add", "abc", "no-path-no-no"])
    assert result.exit_code == 1


def test_cmd_registry_list(registry, mock_dataset_path):
    with dataset_registry() as dct:
        dct["abc"] = mock_dataset_path
        dct["efg"] = "/blabla"

    result = runner.invoke(cli, ["registry", "list"])
    assert result.exit_code == 0
    assert "abc" in result.stdout
    assert "efg" in result.stdout


def test_cmd_registry_list_empty(registry):
    result = runner.invoke(cli, ["registry", "list"])
    assert result.exit_code == 0
    assert "No datasets registered" in result.stdout


def test_cmd_registry_list_check_readable(registry, mock_dataset_path):
    with dataset_registry() as dct:
        dct["abc"] = mock_dataset_path
        dct["efg"] = "/blabla"

    result = runner.invoke(cli, ["registry", "list", "--check-readable"])
    assert result.exit_code == 0


def test_cmd_registry_remove(registry):
    with dataset_registry() as dct:
        dct["efg"] = "/blabla"

    result = runner.invoke(cli, ["registry", "remove", "efg"])
    assert result.exit_code == 0
    with dataset_registry() as dct:
        assert "efg" not in dct


def test_cmd_registry_remove_error_missing(registry):
    result = runner.invoke(cli, ["registry", "remove", "abc"])
    assert result.exit_code == 1
    assert "not registered" in result.stdout
