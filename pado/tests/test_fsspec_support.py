import os

import fsspec
import pytest

from pado.dataset import PadoDataset


@pytest.fixture
def urlpath():
    fs: fsspec.AbstractFileSystem = fsspec.get_filesystem_class("memory")()
    try:
        yield "memory://testdataset"  # nonlocal
    finally:
        fs.rm("testdataset", recursive=True)


def test_dataset_add_source_non_local(datasource, urlpath):
    ds = PadoDataset(urlpath, mode="x")
    ds.add_source(datasource)


def test_dataset_non_local_access(datasource, urlpath):
    ds = PadoDataset(urlpath, mode="x")
    ds.add_source(datasource)

    image_id = next(iter(ds))
    dct = ds[image_id]
    assert set(dct) == {'annotations', 'image', 'metadata'}
    assert dct['annotations']
    assert dct['image']
    assert not dct['metadata'].empty
