from __future__ import annotations

import json
import os
from pathlib import Path

from typer.testing import CliRunner

from pado.__main__ import cli
from pado.dataset import PadoDataset
from pado.io.files import urlpathlike_get_path
from pado.mock import mock_dataset
from pado.registry import dataset_registry
from pado.types import UrlpathWithStorageOptions

runner = CliRunner(mix_stderr=False)


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
    assert "ERROR" in result.stderr
    assert "not_a_dataset" in result.stderr


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
    assert "not-there" in result.stderr
    # dst dataset not there
    with dataset_registry() as dct:
        dct["ds0"] = "somepath"
    result = runner.invoke(cli, ["copy", "--src", "ds0", "--dst", "not-there"])
    assert result.exit_code == 1
    assert "not-there" in result.stderr


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
    csv_file.write_text(f"custom_column\n{os.path.join(*ds.index[0].parts)}\n")
    result = runner.invoke(
        cli,
        ["ops", "filter-ids", mock_dataset_path, "--csv", str(csv_file)],
    )
    assert result.exit_code == 0
    assert len(result.stdout.splitlines()) == 1
    assert "custom_column" in result.stderr


def test_cmd_ops_filter_ids_provide_csv_without_headers(mock_dataset_path, tmp_path):
    ds = PadoDataset(mock_dataset_path)
    # csv file with all columns
    csv_file = tmp_path.joinpath("iids.csv")
    csv_file.write_text(f"{os.path.join(*ds.index[0].parts)}\n")
    result = runner.invoke(
        cli,
        ["ops", "filter-ids", mock_dataset_path, "--csv", str(csv_file), "--no-header"],
    )
    assert result.exit_code == 0
    assert len(result.stdout.splitlines()) == 1


def test_cmd_ops_filter_ids_provide_csv_without_headers_wrong_c(
    mock_dataset_path, tmp_path
):
    ds = PadoDataset(mock_dataset_path)
    # csv file with all columns
    csv_file = tmp_path.joinpath("iids.csv")
    csv_file.write_text(f"{os.path.join(*ds.index[0].parts)}\n")
    result = runner.invoke(
        cli,
        [
            "ops",
            "filter-ids",
            mock_dataset_path,
            "--csv",
            str(csv_file),
            "--no-header",
            "-c",
            "blah",
        ],
    )
    assert result.exit_code == 2
    assert "integer" in result.stderr


def test_cmd_ops_filter_ids_provide_csv_selected_column(mock_dataset_path, tmp_path):
    ds = PadoDataset(mock_dataset_path)
    # csv file with one selected column
    csv_file = tmp_path.joinpath("iids.csv")
    csv_file.write_text(f"c0,c1\nnono,{','.join(ds.index[0].parts)}\n")
    result = runner.invoke(
        cli,
        ["ops", "filter-ids", mock_dataset_path, "--csv", str(csv_file), "-c", "c1"],
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
            "c0",
            "-c",
            "c1",
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


def test_cmd_ops_remote_ids(mock_dataset_path):
    result = runner.invoke(cli, ["ops", "remote-images", mock_dataset_path])
    assert result.exit_code == 0
    assert result.stderr == ""
    assert result.stdout == ""


def test_cmd_ops_remote_ids_as_path(mock_dataset_path):
    result = runner.invoke(
        cli, ["ops", "remote-images", mock_dataset_path, "--as-path"]
    )
    assert result.exit_code == 0
    assert "ImageId" not in result.stdout


def test_cmd_ops_local_ids(mock_dataset_path):
    result = runner.invoke(cli, ["ops", "local-images", mock_dataset_path])
    assert result.exit_code == 0
    assert len(list(result.stdout.splitlines())) == 3


def test_cmd_ops_local_ids_as_path(mock_dataset_path):
    result = runner.invoke(cli, ["ops", "local-images", mock_dataset_path, "--as-path"])
    assert result.exit_code == 0
    assert "ImageId" not in result.stdout


def test_cmd_ops_local_ids_with_check_missing(mock_dataset_path):
    result = runner.invoke(
        cli, ["ops", "local-images", mock_dataset_path, "--check-missing"]
    )
    assert result.exit_code == 0
    assert len(list(result.stdout.splitlines())) == 3
    assert "missing" not in result.stdout


def test_cmd_registry_add(registry, mock_dataset_path):
    result = runner.invoke(cli, ["registry", "add", "abc", mock_dataset_path])
    assert result.exit_code == 0
    with dataset_registry() as dct:
        assert "abc" in dct


def test_cmd_registry_add_wrong_secret(registry, mock_dataset_path):
    result = runner.invoke(
        cli,
        [
            "registry",
            "add",
            "abc",
            mock_dataset_path,
            "--storage-options",
            "{}",
            "--secret",
            "blabla",
        ],
    )
    assert result.exit_code == 1
    assert "not a key in storage_options" in result.stderr


def test_cmd_registry_add_scramble_secret(registry, mock_dataset_path):
    result = runner.invoke(
        cli,
        [
            "registry",
            "add",
            "abc",
            mock_dataset_path,
            "--storage-options",
            '{"something": 1}',
            "--secret",
            "something",
            "--urlpath-is-secret",
        ],
    )
    assert result.exit_code == 0
    assert "scrambling: urlpath" in result.stdout
    assert "scrambling: something" in result.stdout


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
    assert "No datasets registered" in result.stderr


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
    assert "not registered" in result.stderr


def _make_secret_urlpath_with_storage_options(dataset_name, *so):
    return UrlpathWithStorageOptions(
        urlpath=f"@SECRET:__default__:{dataset_name}:urlpath",
        storage_options={s: f"@SECRET:__default__:{dataset_name}:{s}" for s in so},
    )


def test_cmd_registry_input_secrets(registry):
    with dataset_registry() as dct:
        dct["nosecrets"] = "/nono"
        dct["abc"] = _make_secret_urlpath_with_storage_options("abc", "blah", "skip")

    result = runner.invoke(
        cli, ["registry", "input-secrets"], input="urlpath-test\nblah-test\n \n"
    )
    assert result.exit_code == 0
    assert "nosecrets has no missing secrets" in result.stdout
    assert "@SECRET:__default__:abc:urlpath = urlpath-test" in result.stdout
    assert "@SECRET:__default__:abc:blah = blah-test" in result.stdout
    assert "skipped." in result.stdout


def test_cmd_options_show():
    result = runner.invoke(
        cli,
        ["config", "show"],
    )
    assert result.exit_code == 0
    dct = json.loads(result.stdout)
    assert {"CACHE_PATH", "CONFIG_PATH"}.issubset(dct)


def test_cmd_ops_update_image(dataset_and_images_path):
    ds_pth, img_pth = dataset_and_images_path
    result = runner.invoke(
        cli,
        [
            "ops",
            "update-images",
            ds_pth,
            "--search-urlpath",
            img_pth,
            "--glob",
            "*.svs",
        ],
    )
    assert result.exit_code == 0
    img_urlpath = PadoDataset(ds_pth).images.df.urlpath.iloc[0]
    path = urlpathlike_get_path(img_urlpath)
    assert Path(img_pth) == Path(path).parent


def test_cmd_ops_update_image_dry_run(dataset_and_images_path):
    ds_pth, img_pth = dataset_and_images_path
    result = runner.invoke(
        cli,
        [
            "ops",
            "update-images",
            ds_pth,
            "--search-urlpath",
            img_pth,
            "--glob",
            "*.svs",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert img_pth not in PadoDataset(ds_pth).images.df.urlpath.iloc[0]


def test_cmd_ops_update_image_no_pattern_missing_glob(mock_dataset_path, tmp_path):
    result = runner.invoke(
        cli,
        [
            "ops",
            "update-images",
            mock_dataset_path,
            "--search-urlpath",
            os.fspath(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "not a pattern: must provide --glob" in result.stdout


def test_cmd_ops_update_image_wrong_glob(mock_dataset_path, tmp_path):
    result = runner.invoke(
        cli,
        [
            "ops",
            "update-images",
            mock_dataset_path,
            "--search-urlpath",
            os.fspath(tmp_path),
            "--glob",
            "abc.tif",
        ],
    )
    assert result.exit_code == 1
    assert "does not contain wildcard '*'" in result.stdout


def test_cmd_ops_update_image_pattern_and_glob(mock_dataset_path, tmp_path):
    result = runner.invoke(
        cli,
        [
            "ops",
            "update-images",
            mock_dataset_path,
            "--search-urlpath",
            os.fspath(tmp_path / "*"),
            "--glob",
            "*",
        ],
    )
    assert result.exit_code == 1
    assert "`--search-urlpath` OR provide `--glob`" in result.stdout
