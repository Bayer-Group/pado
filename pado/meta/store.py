import json
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import fsspec
import pandas as pd
import pyarrow
from pandas.io.common import is_fsspec_url
from pandas.io.parquet import BaseImpl

# noinspection PyProtectedMember,PyPep8Naming
from pado._version import version as _pado_version

# --- extra metadata stored in pado metadata files --------------------

# DEV_NOTE:
# - changing the following keys is strongly discouraged, to prevent
#   versioning and support nightmares later on.
# - adding is fine provided there's tests and fallbacks for conversion
METADATA_KEY_PADO_VERSION = 'pado_version'
METADATA_KEY_DATASET_VERSION = 'dataset_version'
METADATA_KEY_IDENTIFIER = 'identifier'
METADATA_KEY_USER_METADATA = 'user_metadata'

METADATA_PREFIX = 'pado.metadata.parquet'


def _val_set(dct, key, value):
    k = f'{METADATA_PREFIX}.{key}'.encode()  # parquet requires bytes keys
    dct[k] = json.dumps(value).encode()      # string encode value


def _val_get(dct, key, default):             # require providing a default
    k = f'{METADATA_PREFIX}.{key}'.encode()
    v = dct.get(k)
    if v is None:
        return default
    return json.loads(v)


# --- serialization ---------------------------------------------------

class _MetadataStore:
    """skeleton class for storing metadata as parquet

    DEV_NOTE: consider this as a seed for your mental model for metadata stores.
      We could get rid of this, since support for writing older versions of
      metadata is not really needed for now.
    """
    version: int

    def to_urlpath(self, urlpath: str, df: pd.DataFrame, *, identifier: Optional[str] = None, **extra_md):
        raise NotImplementedError('implement in subclass')

    @classmethod
    def from_urlpath(cls, urlpath: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        raise NotImplementedError('implement in subclass')


def get_store_instance(version: int = -1):
    """get a specific version of the metadata serializer (default: latest)"""
    stores = [
        _MetadataStoreV0,  # initial version of the store
        MetadataStore,  # latest version
    ]
    assert [s.version for s in stores] == list(range(len(stores)))
    return stores[version]()


class _MetadataStoreV0(_MetadataStore):
    version = 0

    def to_urlpath(self, urlpath: str, df: pd.DataFrame, *, identifier: Optional[str] = None, **extra_md):
        raise NotImplementedError("unsupported for old stores")

    @classmethod
    def from_urlpath(cls, urlpath: str) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
        # TODO: :)
        raise NotImplementedError("TODO")


class MetadataStore(_MetadataStore):
    version = 1

    # changing these constants is untested
    COMPRESSION = 'GZIP'
    USE_NULLABLE_DTYPES = False  # TODO: we should force switch to nullable dtypes

    def to_urlpath(self, urlpath: str, df: pd.DataFrame, *, identifier: Optional[str] = None, **user_metadata):
        """store """
        if not is_fsspec_url(urlpath):
            raise TypeError(f"requires a fsspec url, got: {urlpath!r}")

        BaseImpl.validate_dataframe(df)

        # noinspection PyArgumentList
        table = pyarrow.Table.from_pandas(df, schema=None, preserve_index=None)

        # prepare new schema
        dct = {}
        _val_set(dct, METADATA_KEY_IDENTIFIER, identifier)
        _val_set(dct, METADATA_KEY_PADO_VERSION, _pado_version)
        _val_set(dct, METADATA_KEY_DATASET_VERSION, self.version)
        if user_metadata:
            _val_set(dct, METADATA_KEY_USER_METADATA, user_metadata)
        dct.update(table.schema.metadata)

        # rewrite table schema
        table = table.replace_schema_metadata(dct)

        with fsspec.open(urlpath, mode="wb") as f:
            # write to single output file
            pyarrow.parquet.write_table(
                table, f, compression=self.COMPRESSION,
            )

    @classmethod
    def from_urlpath(cls, urlpath: str) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
        if not is_fsspec_url(urlpath):
            raise TypeError(f"requires a fsspec urlpath, got: {urlpath!r}")

        to_pandas_kwargs = {}
        if cls.USE_NULLABLE_DTYPES:
            mapping = {
                pyarrow.int8(): pd.Int8Dtype(),
                pyarrow.int16(): pd.Int16Dtype(),
                pyarrow.int32(): pd.Int32Dtype(),
                pyarrow.int64(): pd.Int64Dtype(),
                pyarrow.uint8(): pd.UInt8Dtype(),
                pyarrow.uint16(): pd.UInt16Dtype(),
                pyarrow.uint32(): pd.UInt32Dtype(),
                pyarrow.uint64(): pd.UInt64Dtype(),
                pyarrow.bool_(): pd.BooleanDtype(),
                pyarrow.string(): pd.StringDtype(),
            }
            to_pandas_kwargs["types_mapper"] = mapping.get

        _fs, _path = fsspec.core.url_to_fs(urlpath)
        table = pyarrow.parquet.read_table(_path, use_pandas_metadata=True, filesystem=_fs)

        # retrieve the additional metadata stored in the parquet
        _md = table.schema.metadata
        identifier = _val_get(_md, METADATA_KEY_IDENTIFIER, None)
        dataset_version = _val_get(_md, METADATA_KEY_DATASET_VERSION, 0)
        dataset_pado_version = _val_get(_md, METADATA_KEY_PADO_VERSION, '0.0.0')
        user_metadata = _val_get(_md, METADATA_KEY_USER_METADATA, {})

        if dataset_version < cls.version:
            raise RuntimeError(
                f"{urlpath} uses MetadataStore version={cls.version} "
                f"(created with pado=={dataset_pado_version}): "
                "please migrate the PadoDataset to a newer version"
            )
        elif dataset_version > cls.version:
            raise RuntimeError(
                f"{urlpath} uses MetadataStore version={cls.version} "
                f"(created with pado=={dataset_pado_version}): "
                "please update pado"
            )

        df = table.to_pandas(**to_pandas_kwargs)
        version_info = {
            METADATA_KEY_PADO_VERSION: dataset_pado_version,
            METADATA_KEY_DATASET_VERSION: cls.version,
        }
        user_metadata.update(version_info)
        return df, identifier, user_metadata


# --- version conversion ----------------------------------------------

def check_dataset_version(urlpath: str) -> int:
    """return the dataset version"""
    if not is_fsspec_url(urlpath):
        raise TypeError(f"requires a fsspec urlpath, got: {urlpath!r}")

    _fs, _path = fsspec.core.url_to_fs(urlpath)
    table = pyarrow.parquet.read_table(_path, use_pandas_metadata=True, filesystem=_fs)

    _md = table.schema.metadata
    dataset_version = _val_get(_md, METADATA_KEY_DATASET_VERSION, 0)
    return dataset_version


def update_dataset(df: pd.DataFrame, identifier: str, **extra_md) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    dataset_version = extra_md[METADATA_KEY_DATASET_VERSION]
    pado_version = extra_md[METADATA_KEY_PADO_VERSION]

    if dataset_version == MetadataStore.version:
        # all good
        return df, identifier, extra_md

    elif dataset_version > MetadataStore.version:
        raise NotImplementedError(
            f"dataset (version={dataset_version}) was created with pado=={pado_version} "
            f"please update your pado ({_pado_version})"
        )

    elif dataset_version < 0:
        raise RuntimeError(
            f"dataset has unsupported version {dataset_version!r} created with pado=={pado_version}"
        )

    df, identifier, extra_md = _get_dataset_migration(dataset_version)(df, identifier, **extra_md)
    # recurse
    return update_dataset(df, identifier, **extra_md)


def _get_dataset_migration(version: int):
    """return the matching conversion function to increase the version by one increment"""
    updates = {
        0: _update_v0_to_v1,
    }
    return updates[version]


# --- pado metadata migrations ----------------------------------------

# noinspection PyUnusedLocal
def _update_v0_to_v1(df: pd.DataFrame, identifier: str, **extra_md) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """convert from v0 to v1

    Notes
    -----
    v0 was still missing an enforced schema, so this might potentially be a lot of conversion code
    """
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    _df = pd.DataFrame({'A': [1, 2, 3], 'B': [2, 3, 5]})

    _store = MetadataStore()

    up = 'file:///tmp/test.parquet'

    _store.to_urlpath(up, _df, identifier='haha')
    _df2, id2, md = _store.from_urlpath(up)

    assert md['identifier'] == id2 == 'haha'
    assert _df2.columns.tolist() == _df.columns.tolist(), f"{_df2.columns} vs {_df.columns}"
    assert _df2.index.tolist() == _df.index.tolist(), f"{_df2.index} vs {_df.index}"
    assert _df2.dtypes.tolist() == _df.dtypes.tolist(), f"{_df2.dtypes} vs {_df.dtypes}"
    assert _df2.equals(_df)
