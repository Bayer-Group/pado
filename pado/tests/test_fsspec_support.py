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
    ds.ingest_obj(datasource)


def test_dataset_non_local_access(datasource, urlpath):
    ds = PadoDataset(urlpath, mode="x")
    ds.ingest_obj(datasource)

    image_id = next(iter(ds.images))
    a = ds.annotations[image_id]
    m = ds.metadata[image_id]
    i = ds.images[image_id]
