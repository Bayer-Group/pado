from __future__ import annotations

import os
import uuid

import pytest

from pado.dataset import PadoDataset
from pado.mock import mock_dataset
from pado.mock import mock_images
from pado.settings import settings


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
def dataset_ip_only(tmp_path):
    dataset_path = tmp_path / "my_dataset"
    ds = mock_dataset(dataset_path, metadata_provider=False, annotation_provider=False)
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
def dataset_and_images_path(tmp_path):
    dataset_path = tmp_path / "my_dataset"
    mock_dataset(dataset_path, num_images=3)
    images_path = tmp_path / "other_images"
    images_path.mkdir()
    mock_images(images_path, number=3)
    yield os.fspath(dataset_path), os.fspath(images_path)


@pytest.fixture(scope="function")
def registry(tmp_path):
    # mock configuration path
    old_conf_path = settings.config_path
    new_conf_path = tmp_path.joinpath(f"mocked_pado_config_{uuid.uuid4()}")
    settings.configure(config_path=new_conf_path)
    try:
        yield
    finally:
        settings.configure(config_path=old_conf_path)


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
