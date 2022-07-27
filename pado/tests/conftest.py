from __future__ import annotations

import os

import pytest

from pado.dataset import PadoDataset
from pado.mock import mock_dataset


@pytest.fixture(scope="function")
def datasource(tmp_path):
    ds = mock_dataset(tmp_path)
    yield ds


@pytest.fixture(scope="function")
def dataset(tmp_path):
    dataset_path = tmp_path / "my_dataset"
    ds = mock_dataset(dataset_path)
    yield ds


@pytest.fixture(scope="function")
def dataset_ro(tmp_path):
    dataset_path = tmp_path / "my_dataset"
    ds = mock_dataset(dataset_path)
    del ds
    yield PadoDataset(dataset_path, mode="r")


@pytest.fixture(scope="function")
def mock_dataset_path(tmp_path):
    dataset_path = tmp_path / "my_dataset"
    mock_dataset(dataset_path)
    yield os.fspath(dataset_path)


@pytest.fixture(scope="function")
def registry(monkeypatch, tmp_path):
    # mock configuration path
    conf_path = tmp_path.joinpath("mocked_pado_config")
    monkeypatch.setenv("PADO_CONFIG_PATH", str(conf_path))
    yield


@pytest.fixture(scope="function", autouse=True)
def mock_delete_pathlib_glob(monkeypatch):
    """pathlib.Path.glob suffers from a bug regarding symlinks:

    https://bugs.python.org/issue33428

    let's enforce that we do not use it in pado and use glob.glob instead.
    """
    with monkeypatch.context() as m:
        # remove pathlib.Path.glob so that tests fail if used unintentionally
        m.delattr("pathlib.Path.glob")
        yield
