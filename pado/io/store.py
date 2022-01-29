from __future__ import annotations

import enum
import json
import platform
from abc import ABC
from datetime import datetime
from getpass import getuser
from typing import Any
from typing import Callable
from typing import Dict
from typing import MutableMapping
from typing import Optional
from typing import Tuple

import pandas as pd
import pyarrow
import pyarrow.parquet
from pandas.io.parquet import BaseImpl

from pado._version import version as _pado_version
from pado.io.files import urlpathlike_to_fsspec
from pado.types import UrlpathLike


class StoreType(str, enum.Enum):
    ANNOTATION = "annotation"
    IMAGE = "image"
    METADATA = "metadata"


class Store(ABC):
    METADATA_PREFIX = "pado.store.parquet"
    METADATA_KEY_PADO_VERSION = "pado_version"
    METADATA_KEY_STORE_VERSION = "store_version"
    METADATA_KEY_STORE_TYPE = "store_type"
    METADATA_KEY_IDENTIFIER = "identifier"
    METADATA_KEY_CREATED_AT = "created_at"
    METADATA_KEY_CREATED_BY = "created_by"
    METADATA_KEY_USER_METADATA = "user_metadata"
    METADATA_KEY_PROVIDER_VERSION: str | None = None

    USE_NULLABLE_DTYPES = False  # todo: switch to True?
    COMPRESSION = "GZIP"

    def __init__(self, version: int, store_type: StoreType):
        self.version = int(version)
        self.type = store_type

    def _md_set(self, dct: MutableMapping[bytes, bytes], key: str, value: Any) -> None:
        k = f"{self.METADATA_PREFIX}.{key}".encode()  # parquet requires bytes keys
        dct[k] = json.dumps(value).encode()  # string encode value

    def _md_get(
        self, dct: MutableMapping[bytes, bytes], key: str, default: Any
    ) -> Any:  # require providing a default
        k = f"{self.METADATA_PREFIX}.{key}".encode()
        if k not in dct:
            return default
        return json.loads(dct[k])

    def __metadata_set_hook__(
        self, dct: Dict[bytes, bytes], setter: Callable[[dict, str, Any], None]
    ) -> None:
        """allows setting more metadata in subclasses"""

    def __metadata_get_hook__(
        self, dct: Dict[bytes, bytes], getter: Callable[[dict, str, Any], Any]
    ) -> Optional[dict]:
        """allows getting more metadata in subclass or validate versioning"""

    def to_urlpath(
        self,
        df: pd.DataFrame,
        urlpath: UrlpathLike,
        *,
        identifier: Optional[str] = None,
        **user_metadata,
    ):
        """store a pandas dataframe with an identifier and user metadata"""
        open_file = urlpathlike_to_fsspec(urlpath, mode="wb")

        BaseImpl.validate_dataframe(df)

        # noinspection PyArgumentList
        table = pyarrow.Table.from_pandas(df, schema=None, preserve_index=None)

        # prepare new schema
        dct: Dict[bytes, bytes] = {}
        self._md_set(dct, self.METADATA_KEY_IDENTIFIER, identifier)
        self._md_set(dct, self.METADATA_KEY_PADO_VERSION, _pado_version)
        self._md_set(dct, self.METADATA_KEY_STORE_VERSION, self.version)
        self._md_set(dct, self.METADATA_KEY_STORE_TYPE, self.type.value)
        self._md_set(dct, self.METADATA_KEY_CREATED_AT, datetime.utcnow().isoformat())
        self._md_set(dct, self.METADATA_KEY_CREATED_BY, _get_user_host())
        if user_metadata:
            self._md_set(dct, self.METADATA_KEY_USER_METADATA, user_metadata)
        dct.update(table.schema.metadata)

        # for subclasses
        self.__metadata_set_hook__(dct, self._md_set)

        # rewrite table schema
        table = table.replace_schema_metadata(dct)

        with open_file as f:
            # write to single output file
            pyarrow.parquet.write_table(
                table,
                f,
                compression=self.COMPRESSION,
            )

    def from_urlpath(
        self, urlpath: UrlpathLike
    ) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
        """load dataframe and info from urlpath"""
        open_file = urlpathlike_to_fsspec(urlpath, mode="rb")

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

        table = pyarrow.parquet.read_table(
            open_file.path, use_pandas_metadata=True, filesystem=open_file.fs
        )

        # retrieve the additional metadata stored in the parquet
        _md = table.schema.metadata
        identifier = self._md_get(_md, self.METADATA_KEY_IDENTIFIER, None)
        store_version = self._md_get(_md, self.METADATA_KEY_STORE_VERSION, 0)
        store_type = self._md_get(_md, self.METADATA_KEY_STORE_TYPE, None)
        pado_version = self._md_get(_md, self.METADATA_KEY_PADO_VERSION, "0.0.0")
        created_at = self._md_get(_md, self.METADATA_KEY_CREATED_AT, None)
        created_by = self._md_get(_md, self.METADATA_KEY_CREATED_BY, None)
        user_metadata = self._md_get(_md, self.METADATA_KEY_USER_METADATA, {})

        # for subclasses
        get_hook_data = self.__metadata_get_hook__(_md, self._md_get)

        if store_version < self.version:
            raise RuntimeError(
                f"{urlpath} uses Store version={self.version} "
                f"(created with pado=={pado_version}): "
                "please migrate the PadoDataset to a newer version"
            )
        elif store_version > self.version:
            raise RuntimeError(
                f"{urlpath} uses Store version={self.version} "
                f"(created with pado=={pado_version}): "
                "please update pado"
            )

        df = table.to_pandas(**to_pandas_kwargs)
        version_info = {
            self.METADATA_KEY_PADO_VERSION: pado_version,
            self.METADATA_KEY_STORE_VERSION: self.version,
            self.METADATA_KEY_STORE_TYPE: StoreType(store_type),
            self.METADATA_KEY_CREATED_AT: created_at,
            self.METADATA_KEY_CREATED_BY: created_by,
        }
        user_metadata.update(version_info)
        if get_hook_data is not None:
            user_metadata.update(get_hook_data)
        return df, identifier, user_metadata


def get_store_type(urlpath: UrlpathLike) -> Optional[StoreType]:
    """return the store type from an urlpath"""
    open_file = urlpathlike_to_fsspec(urlpath, mode="rb")
    table = pyarrow.parquet.read_table(
        open_file.path, use_pandas_metadata=True, filesystem=open_file.fs
    )
    key_store_type = f"{Store.METADATA_PREFIX}.{Store.METADATA_KEY_STORE_TYPE}".encode()
    try:
        store_type = json.loads(table.schema.metadata[key_store_type])
    except (KeyError, json.JSONDecodeError):
        return None
    return StoreType(store_type)


def get_store_metadata(urlpath: UrlpathLike) -> Dict[str, Any]:
    """return the store metadata from an urlpath"""
    open_file = urlpathlike_to_fsspec(urlpath, mode="rb")
    table = pyarrow.parquet.read_table(
        open_file.path, use_pandas_metadata=True, filesystem=open_file.fs
    )
    md = {}
    for k, v in dict(table.schema.metadata).items():
        k = k.decode()
        if not k.startswith(Store.METADATA_PREFIX):
            continue
        else:
            k = k[len(Store.METADATA_PREFIX) + 1 :]
        try:
            v = json.loads(v)
        except json.JSONDecodeError as err:
            v = err
        md[k] = v
    return md


def _get_user_host() -> Optional[str]:
    import pado.settings

    try:
        return pado.settings.settings.override_user_host
    except AttributeError:
        pass
    try:
        return f"{getuser()!s}@{platform.uname().node!s}"
    except (OSError, KeyError, ValueError):
        return None
