from pathlib import Path

import pandas as pd
import pytest

from pado.dataset import PadoDataset
from pado.resource import SerializableImageResourcesProvider
from pado.structure import PadoColumn


def count_images(ds: PadoDataset):
    """helper to count images in a pado dataset"""

    def is_image_file(p):
        return p.is_file() and p.name != SerializableImageResourcesProvider.STORAGE_FILE

    images = list(filter(is_image_file, (ds.path / "images").glob("**/*")))
    return len(images)


def test_pado_test_datasource(datasource):
    with pytest.raises(RuntimeError):
        _ = datasource.images
    with pytest.raises(RuntimeError):
        _ = datasource.metadata

    with datasource:
        assert isinstance(datasource.metadata, pd.DataFrame)
        for image in datasource.images:
            assert image.id is not None
            assert image.size > 0


def test_write_pado_dataset(datasource, tmp_path):

    dataset_path = tmp_path / "my_dataset"

    ds = PadoDataset(dataset_path, mode="x")
    ds.add_source(datasource)

    assert count_images(ds) == 1
    assert isinstance(ds.metadata, pd.DataFrame)
    assert len(ds.metadata) == 10


def test_add_multiple_datasets(tmp_path):
    from pado.ext.testsource import TestDataSource

    dataset_path = tmp_path / "my_dataset"
    ds = PadoDataset(dataset_path, mode="x")

    ds.add_source(TestDataSource(num_images=2, num_findings=12, identifier="s0"))
    ds.add_source(TestDataSource(num_images=1, num_findings=7, identifier="s1"))

    assert isinstance(ds.metadata, pd.DataFrame)
    assert len(ds.metadata) == 19
    assert count_images(ds) == 3


@pytest.fixture()
def dataset(datasource, tmp_path):
    dataset_path = tmp_path / "my_dataset"
    ds = PadoDataset(dataset_path, mode="x")
    ds.add_source(datasource)
    yield ds


@pytest.fixture()
def dataset_ro(datasource, tmp_path):
    dataset_path = tmp_path / "my_dataset"
    ds = PadoDataset(dataset_path, mode="x")
    ds.add_source(datasource)
    del ds
    yield PadoDataset(dataset_path, mode="r")


def test_use_dataset_as_datasource(dataset_ro, tmp_path):
    dataset_path = tmp_path / "new_dataset"
    ds = PadoDataset(dataset_path, mode="x")
    ds.add_source(dataset_ro, copy_images=True)


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
