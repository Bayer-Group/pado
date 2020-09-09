from pathlib import Path

import pandas as pd
import pytest

from pado.dataset import PadoDataset
from pado.structure import PadoColumn, PadoReserved


def test_pado_test_datasource(datasource):
    with datasource:
        assert isinstance(datasource.metadata, pd.DataFrame)
        for image in datasource.images():
            assert image.id is not None
            assert image.path.is_file()


def test_write_pado_dataset(datasource, tmp_path):

    dataset_path = tmp_path / "my_dataset"

    ds = PadoDataset(dataset_path, mode="x")
    ds.add_source(datasource)

    assert len(list(filter(Path.is_file, (ds.path / "images").glob("**/*")))) == 1
    assert isinstance(ds.metadata, pd.DataFrame)
    assert len(ds.metadata) == 10


@pytest.fixture()
def dataset(datasource, tmp_path):
    dataset_path = tmp_path / "my_dataset"
    ds = PadoDataset(dataset_path, mode="x")
    ds.add_source(datasource)
    yield ds


@pytest.fixture(scope="module")
def dataset_ro(datasource, tmp_path):
    dataset_path = tmp_path / "my_dataset"
    ds = PadoDataset(dataset_path, mode="x")
    ds.add_source(datasource)
    del ds
    yield PadoDataset(dataset_path, mode="r")


def test_pado_dataframe_accessor(dataset):
    df_subset = dataset.metadata.pado.organs
    assert isinstance(df_subset, pd.DataFrame)
    assert len(df_subset) > 0
    assert all(map(lambda x: x.startswith(PadoColumn.ORGAN), df_subset.columns))
