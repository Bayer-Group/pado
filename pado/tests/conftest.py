import pytest

from pado.dataset import PadoDataset
from pado.ext.testsource import TestDataSource


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
