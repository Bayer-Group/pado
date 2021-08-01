import enum
import json
from abc import ABC
from typing import Any
from typing import Callable
from typing import Dict
from typing import MutableMapping
from typing import Optional
from typing import Tuple

import fsspec
import pandas as pd
import pyarrow
from pandas.io.common import is_fsspec_url
from pandas.io.parquet import BaseImpl

from pado._version import version as _pado_version


class StoreType(str, enum.Enum):
    IMAGE = "image"
    METADATA = "metadata"


class Store(ABC):
    METADATA_KEY_PADO_VERSION = 'pado_version'
    METADATA_KEY_STORE_VERSION = 'store_version'
    METADATA_KEY_STORE_TYPE = 'store_type'
    METADATA_KEY_IDENTIFIER = 'identifier'
    METADATA_KEY_USER_METADATA = 'user_metadata'

    USE_NULLABLE_DTYPES = False  # todo: switch to True?
    COMPRESSION = "GZIP"

    def __init__(self, version: int, store_type: StoreType):
        self.version = int(version)
        self.type = store_type

    @property
    def prefix(self):
        return f"pado.{self.type.value}.parquet"

    def _md_set(self, dct: MutableMapping[bytes, bytes], key: str, value: Any) -> None:
        k = f'{self.prefix}.{key}'.encode()  # parquet requires bytes keys
        dct[k] = json.dumps(value).encode()  # string encode value

    def _md_get(self, dct: MutableMapping[bytes, bytes], key: str, default: Any) -> Any:  # require providing a default
        k = f'{self.prefix}.{key}'.encode()
        if k not in dct:
            return default
        return json.loads(dct[k])

    def __metadata_set_hook__(self, dct: Dict[bytes, bytes], setter: Callable[[dict, str, Any], None]) -> None:
        """allows setting more metadata in subclasses"""

    def __metadata_get_hook__(self, dct: Dict[bytes, bytes], getter: Callable[[dict, str, Any], Any]) -> Optional[dict]:
        """allows getting more metadata in subclass or validate versioning"""

    def to_urlpath(self, df: pd.DataFrame, urlpath: str, *, identifier: Optional[str] = None, **user_metadata):
        """store a pandas dataframe with an identifier and user metadata"""
        if not is_fsspec_url(urlpath):
            raise TypeError(f"requires a fsspec url, got: {urlpath!r}")

        BaseImpl.validate_dataframe(df)

        # noinspection PyArgumentList
        table = pyarrow.Table.from_pandas(df, schema=None, preserve_index=None)

        # prepare new schema
        dct: Dict[bytes, bytes] = {}
        self._md_set(dct, self.METADATA_KEY_IDENTIFIER, identifier)
        self._md_set(dct, self.METADATA_KEY_PADO_VERSION, _pado_version)
        self._md_set(dct, self.METADATA_KEY_STORE_VERSION, self.version)
        self._md_set(dct, self.METADATA_KEY_STORE_TYPE, self.type.value)
        if user_metadata:
            self._md_set(dct, self.METADATA_KEY_USER_METADATA, user_metadata)
        dct.update(table.schema.metadata)

        # rewrite table schema
        table = table.replace_schema_metadata(dct)

        with fsspec.open(urlpath, mode="wb") as f:
            # write to single output file
            pyarrow.parquet.write_table(
                table, f, compression=self.COMPRESSION,
            )

    def from_urlpath(self, urlpath: str) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
        """load dataframe and info from urlpath"""
        if not is_fsspec_url(urlpath):
            raise TypeError(f"requires a fsspec urlpath, got: {urlpath!r}")

        to_pandas_kwargs = {}
        if self.USE_NULLABLE_DTYPES:
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
        identifier = self._md_get(_md, self.METADATA_KEY_IDENTIFIER, None)
        store_version = self._md_get(_md, self.METADATA_KEY_STORE_VERSION, 0)
        store_type = self._md_get(_md, self.METADATA_KEY_STORE_TYPE, None)
        dataset_pado_version = self._md_get(_md, self.METADATA_KEY_PADO_VERSION, '0.0.0')
        user_metadata = self._md_get(_md, self.METADATA_KEY_USER_METADATA, {})

        if store_version < self.version:
            raise RuntimeError(
                f"{urlpath} uses Store version={self.version} "
                f"(created with pado=={dataset_pado_version}): "
                "please migrate the PadoDataset to a newer version"
            )
        elif store_version > self.version:
            raise RuntimeError(
                f"{urlpath} uses Store version={self.version} "
                f"(created with pado=={dataset_pado_version}): "
                "please update pado"
            )

        df = table.to_pandas(**to_pandas_kwargs)
        version_info = {
            self.METADATA_KEY_PADO_VERSION: dataset_pado_version,
            self.METADATA_KEY_STORE_VERSION: self.version,
            self.METADATA_KEY_STORE_TYPE: StoreType(store_type),
        }
        user_metadata.update(version_info)
        return df, identifier, user_metadata
