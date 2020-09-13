from pathlib import Path

import pandas as pd
import pytest

from pado.dataset import PadoDataset
from pado.structure import PadoColumn


def test_pado_test_datasource(datasource):
    with datasource:
        assert isinstance(datasource.metadata, pd.DataFrame)
        for image in datasource.images:
            assert image.id is not None
            assert image.size > 0


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


_accessors = {
    "studies": PadoColumn.STUDY,
    "experiments": PadoColumn.EXPERIMENT,
    "groups": PadoColumn.GROUP,
    "animals": PadoColumn.ANIMAL,
    "compounds": PadoColumn.COMPOUND,
    "organs": PadoColumn.ORGAN,
    "slides": PadoColumn.SLIDE,
    "images": PadoColumn.IMAGE,
    "findings": PadoColumn.FINDING,
}


@pytest.mark.parametrize("accessor,column", _accessors.items(), ids=_accessors.keys())
def test_pado_dataframe_accessor(dataset, accessor, column):
    # this is testing, i.e.: dataset.metadata.pado.organs
    df_subset = getattr(dataset.metadata.pado, accessor)
    assert isinstance(df_subset, pd.DataFrame)
    assert len(df_subset) > 0
    assert all(map(lambda x: x.startswith(column), df_subset.columns))
