import fsspec
import uuid
import pytest

from pado.dataset import PadoDataset

@pytest.fixture(scope='function')
def urlpath():
    # the fsspec memory filesystem is shared,
    # so provide a new path for each invocation
    unique = uuid.uuid4().hex
    yield f"memory://{unique}"

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
