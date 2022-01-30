from __future__ import annotations

import os.path
import uuid

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pado._version import version as _pado_version
from pado.io.store import DataVersionTuple
from pado.io.store import StoreInfo
from pado.io.store import StoreMigrationInfo
from pado.io.store import StoreType
from pado.io.store import StoreVersionTuple
from pado.io.store import find_migration_path
from pado.metadata.providers import MetadataProviderStore


@pytest.fixture(scope="function")
def urlpath():
    # the fsspec memory filesystem is shared,
    # so provide a new path for each invocation
    unique = uuid.uuid4().hex
    yield f"memory://{unique}"


@pytest.fixture(scope="function")
def parquet_path(urlpath):
    yield os.path.join(urlpath, "test.parquet")


def test_meta_store_roundtrip(parquet_path):
    df = pd.DataFrame({"A": [1, 2, 3]})
    identifier = "test-identifier"
    meta = {"abc": 1}

    store = MetadataProviderStore()
    store.to_urlpath(df, parquet_path, identifier=identifier, **meta)
    df2, identifier2, meta2 = store.from_urlpath(parquet_path)

    # ensure version info is there
    assert meta2.pop(MetadataProviderStore.METADATA_KEY_PADO_VERSION) == _pado_version
    assert meta2.pop(MetadataProviderStore.METADATA_KEY_STORE_VERSION) == 1
    assert (
        meta2.pop(MetadataProviderStore.METADATA_KEY_STORE_TYPE) == StoreType.METADATA
    )
    assert (
        meta2.pop(MetadataProviderStore.METADATA_KEY_PROVIDER_VERSION)
        == MetadataProviderStore.DATASET_VERSION
    )
    assert meta2.pop(MetadataProviderStore.METADATA_KEY_CREATED_AT) is not None
    assert meta2.pop(MetadataProviderStore.METADATA_KEY_CREATED_BY) is not None

    # ensure round trip
    assert_frame_equal(
        df, df2, check_column_type=True, check_index_type=True, check_exact=True
    )
    assert identifier == identifier2
    assert meta == meta2


def test_migration_can_migrate():
    so = StoreInfo(
        StoreType.IMAGE, StoreVersionTuple(0, 0), DataVersionTuple("test", 0)
    )
    m = StoreMigrationInfo.create(StoreType.IMAGE, None, (0, 0, None), (0, 1, None))
    assert m.can_migrate(so)


def test_migration_resolution():

    store_info = StoreInfo(
        StoreType.IMAGE, StoreVersionTuple(0, 0), DataVersionTuple("test", 0)
    )

    migrations = list(
        map(
            lambda args: StoreMigrationInfo.create(*args),
            [
                (StoreType.IMAGE, None, (0, 0, None), (0, 1, None)),
                (StoreType.IMAGE, "test", (0, 1, 0), (0, 1, 1)),
                (StoreType.IMAGE, "test", (0, 1, 1), (0, 1, 2)),
                (StoreType.ANNOTATION, "test", (0, 1, 0), (0, 1, 1)),
                (StoreType.IMAGE, None, (0, 1, None), (0, 2, None)),
                (StoreType.IMAGE, "test", (0, 2, 2), (0, 2, 3)),
            ],
        )
    )

    ms = find_migration_path(store_info, migrations)
    assert len(ms) == 5
