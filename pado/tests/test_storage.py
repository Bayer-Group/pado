
import itertools
import os.path
import uuid

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pado.meta.store import MetadataStore
from pado.meta.store import METADATA_KEY_PADO_VERSION
from pado.meta.store import METADATA_KEY_DATASET_VERSION


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
    _ = meta2.pop(METADATA_KEY_PADO_VERSION)
    _ = meta2.pop(METADATA_KEY_DATASET_VERSION)

    # ensure round trip
    assert_frame_equal(df, df2, check_column_type=True, check_index_type=True, check_exact=True)
    assert identifier == identifier2
    assert meta == meta2
