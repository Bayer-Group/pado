
import os.path
import uuid

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pado._version import version as _pado_version
from pado.io.store import StoreType
from pado.metadata.store import MetadataStore


@pytest.fixture(scope='function')
def urlpath():
    # the fsspec memory filesystem is shared,
    # so provide a new path for each invocation
    unique = uuid.uuid4().hex
    yield f"memory://{unique}"


@pytest.fixture(scope='function')
def parquet_path(urlpath):
    yield os.path.join(urlpath, "test.parquet")


def test_meta_store_roundtrip(parquet_path):
    df = pd.DataFrame({'A': [1, 2, 3]})
    identifier = 'test-identifier'
    meta = {'abc': 1}

    store = MetadataStore()
    store.to_urlpath(parquet_path, df, identifier=identifier, **meta)
    df2, identifier2, meta2 = store.from_urlpath(parquet_path)

    # ensure version info is there
    assert meta2.pop(MetadataStore.METADATA_KEY_PADO_VERSION) == _pado_version
    assert meta2.pop(MetadataStore.METADATA_KEY_STORE_VERSION) == 1
    assert meta2.pop(MetadataStore.METADATA_KEY_STORE_TYPE) == StoreType.METADATA
    assert meta2.pop(MetadataStore.METADATA_KEY_DATASET_VERSION) == MetadataStore.DATASET_VERSION

    # ensure round trip
    assert_frame_equal(df, df2, check_column_type=True, check_index_type=True, check_exact=True)
    assert identifier == identifier2
    assert meta == meta2
