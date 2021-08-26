import pytest

from pado.mock import mock_dataset
from pado.dataset import PadoDataset


@pytest.fixture(scope="function")
def datasource(tmp_path):
    ds = mock_dataset(tmp_path)
    yield ds


@pytest.fixture(scope="function")
def dataset(tmp_path):
    dataset_path = tmp_path / "my_dataset"
    ds = mock_dataset(tmp_path)
    yield ds


@pytest.fixture(scope="function")
def dataset_ro(datasource, tmp_path):
    dataset_path = tmp_path / "my_dataset"
    ds = mock_dataset(dataset_path)
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
