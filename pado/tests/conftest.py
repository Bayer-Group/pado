import pytest

from pado._test_source import TestDataSource
from pado.dataset import PadoDataset


@pytest.fixture(scope="function")
def datasource():
    yield TestDataSource(num_images=1, num_findings=10)


@pytest.fixture(scope="function")
def dataset(datasource, tmp_path):
    dataset_path = tmp_path / "my_dataset"
    ds = PadoDataset(dataset_path, mode="x")
    ds.add_source(datasource)
    yield ds


@pytest.fixture(scope="function")
def dataset_ro(datasource, tmp_path):
    dataset_path = tmp_path / "my_dataset"
    ds = PadoDataset(dataset_path, mode="x")
    ds.add_source(datasource)
    del ds
    yield PadoDataset(dataset_path, mode="r")


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
