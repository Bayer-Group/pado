from __future__ import annotations

import enum
import importlib
import json
import platform
from abc import ABC
from datetime import datetime
from getpass import getuser
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar

import pandas as pd
import pyarrow
import pyarrow.parquet
from pandas.io.parquet import BaseImpl

from pado._version import version as _pado_version
from pado.io.files import fsopen
from pado.io.files import urlpathlike_to_fs_and_path
from pado.io.files import urlpathlike_to_fsspec
from pado.types import UrlpathLike

if TYPE_CHECKING:
    from pado.dataset import PadoDataset


class StoreType(str, enum.Enum):
    ANNOTATION = "annotation"
    IMAGE = "image"
    METADATA = "metadata"
    IMAGE_PREDICTIONS = "image_predictions"
    METADATA_PREDICTIONS = "metadata_predictions"


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
    METADATA_KEY_DATA_VERSION: str | None = None

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
        storage_options: dict[str, Any] | None = None,
        **user_metadata,
    ):
        """store a pandas dataframe with an identifier and user metadata"""
        open_file = urlpathlike_to_fsspec(
            urlpath, mode="wb", storage_options=storage_options
        )

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
        self, urlpath: UrlpathLike, *, storage_options: dict[str, Any] | None = None
    ) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
        """load dataframe and info from urlpath"""
        open_file = urlpathlike_to_fsspec(
            urlpath, mode="rb", storage_options=storage_options
        )

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


def get_store_type(
    urlpath: UrlpathLike, *, storage_options: dict[str, Any] | None = None
) -> Optional[StoreType]:
    """return the store type from an urlpath"""
    open_file = urlpathlike_to_fsspec(
        urlpath, mode="rb", storage_options=storage_options
    )
    table = pyarrow.parquet.read_table(
        open_file.path, use_pandas_metadata=True, filesystem=open_file.fs
    )
    key_store_type = f"{Store.METADATA_PREFIX}.{Store.METADATA_KEY_STORE_TYPE}".encode()
    try:
        store_type = json.loads(table.schema.metadata[key_store_type])
    except (KeyError, json.JSONDecodeError):
        return None
    return StoreType(store_type)


def get_store_metadata(
    urlpath: UrlpathLike, *, storage_options: dict[str, Any] | None = None
) -> Dict[str, Any]:
    """return the store metadata from an urlpath"""
    open_file = urlpathlike_to_fsspec(
        urlpath, mode="rb", storage_options=storage_options
    )
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


# === store migration =================================================


class StoreVersionTuple(NamedTuple):
    """describing storage implementation"""

    store: int
    provider: int


class DataVersionTuple(NamedTuple):
    """describing stored data"""

    identifier: str
    version: int


class StoreInfo(NamedTuple):
    """version information for storage and data"""

    store_type: StoreType
    store_version: StoreVersionTuple | None
    data_version: DataVersionTuple | None

    def is_explicit(self):
        """provides all version information explicitly"""
        return self.store_version is not None and self.data_version is not None

    def to_string(self):
        """provide a short version-like string"""
        s, p = (0, 0) if self.store_version is None else self.store_version
        id_, d = ("no-data", "?") if self.data_version is None else self.data_version
        return f"{self.store_type.value}-s{s}p{p}-{id_}@{d}"


class StoreMigrationInfo(NamedTuple):
    """upgrade migration information for stores"""

    store_info: StoreInfo
    target_store_version: int | None = None
    target_provider_version: int | None = None
    target_data_version: int | None = None

    def can_migrate(self, store_info: StoreInfo) -> StoreInfo | None:
        """return the new store info if migration is possible or None otherwise"""
        if not store_info.is_explicit():
            raise ValueError("must provide an explicit store and data version")

        m_si, s_si = self.store_info, store_info

        if not (
            m_si.store_type == s_si.store_type
            and (m_si.store_version is None or m_si.store_version == s_si.store_version)
            and (m_si.data_version is None or m_si.data_version == s_si.data_version)
        ):
            return None
        else:
            # return version as if updated
            sv = self.target_store_version
            if sv is None:
                sv = store_info.store_version.store
            pv = self.target_provider_version
            if pv is None:
                pv = store_info.store_version.provider
            dv = self.target_data_version
            if dv is None:
                dv = store_info.data_version.version
            return StoreInfo(
                store_type=store_info.store_type,
                store_version=StoreVersionTuple(sv, pv),
                data_version=DataVersionTuple(store_info.data_version.identifier, dv),
            )

    @classmethod
    def create(
        cls,
        store_type: StoreType,
        data_identifier: str | None,
        origin: tuple[int | None, int | None, int | None],
        target: tuple[int | None, int | None, int | None],
    ) -> StoreMigrationInfo:
        """convenience constructor"""
        # checks
        _pairs = [
            (data_identifier, origin[2], "data_identifier and version"),
            (origin[0], origin[1], "store and provider version"),
        ]
        for a, b, name in _pairs:
            if (a is None and b is not None) or (b is None and a is not None):
                raise ValueError(f"either both of {name} must be None or none")

        if any(x is None for x in origin[:2]):
            raise NotImplementedError("fixme: currently assumed to be provided...")

        for a, b, name in zip(origin, target, ["store", "provider", "data"]):
            if b is not None:
                if a is None:
                    raise ValueError(
                        f"{name} target versions must be based on specific origin version"
                    )
                if b < a:
                    raise ValueError(
                        f"{name} target version must be equal or larger than origin version"
                    )

        if origin == target:
            raise ValueError("the target version must increase")

        return StoreMigrationInfo(
            store_info=StoreInfo(
                store_type=store_type,
                store_version=StoreVersionTuple(origin[0], origin[1]),
                data_version=DataVersionTuple(data_identifier, origin[2])
                if data_identifier is not None
                else None,
            ),
            target_store_version=target[0],
            target_provider_version=target[1],
            target_data_version=target[2],
        )

    def __repr__(self):
        def fill_none(x, value):
            return value if x is None else x

        _osv, _opv, _odv, _tsv, _tpv, _tdv = map(
            lambda x: fill_none(x, "X"),
            [
                self.store_info.store_version.store,
                self.store_info.store_version.provider,
                self.store_info.data_version.version
                if self.store_info.data_version
                else None,
                self.target_store_version,
                self.target_provider_version,
                self.target_data_version,
            ],
        )
        ov = f"s{_osv}p{_opv}d{_odv}"
        tv = f"s{_tsv}p{_tpv}d{_tdv}"
        type_ = self.store_info.store_type.value

        if self.store_info.data_version is None:
            d_id = None
        else:
            d_id = self.store_info.data_version.identifier
        return f"<StoreMigrationInfo type={type_!r} data_id={d_id!r} migration='{ov}->{tv}'>"


StoreMigrationFunc = Callable[
    [pd.DataFrame, str, Dict[str, Any]], Tuple[pd.DataFrame, str, Dict[str, Any]]
]
_STORE_MIGRATION_REGISTRY: Dict[StoreMigrationInfo, StoreMigrationFunc] = {}


def register_store_migration(
    info: StoreMigrationInfo, func: StoreMigrationFunc
) -> None:
    """register an upgrade migration function for a pado data provider"""
    if not isinstance(info, StoreMigrationInfo):
        raise TypeError(info)
    if not callable(func):
        raise TypeError(f"{func} not callable")
    if info in _STORE_MIGRATION_REGISTRY:
        raise ValueError(f"{info!r} already in registry")
    _STORE_MIGRATION_REGISTRY[info] = func


def get_migration_registry() -> Mapping[StoreMigrationInfo, StoreMigrationFunc]:
    """return the upgrade migration registry"""
    return _STORE_MIGRATION_REGISTRY


def find_migration_path(
    store_info: StoreInfo,
    migrations: Sequence[StoreMigrationInfo],
) -> list[StoreMigrationInfo]:
    """return a store/data migration path"""
    pth = []

    si = store_info
    while True:
        # then try store / provider migrations
        for m_info in migrations:  # fixme: inefficient
            _si = m_info.can_migrate(si)
            if _si is not None:
                pth.append(m_info)
                si = _si
                break
        else:
            break

    return pth


S = TypeVar("S", bound=Store)


def _get_store_subclass(store_type: StoreType) -> Type[S]:
    """return the corresponding Store subclass"""
    # get the corresponding store class
    _module, _clsname = {
        StoreType.IMAGE: ("pado.images.providers", "ImageProviderStore"),
        StoreType.METADATA: ("pado.metadata.providers", "MetadataProviderStore"),
        StoreType.ANNOTATION: ("pado.annotations.providers", "AnnotationProviderStore"),
        StoreType.IMAGE_PREDICTIONS: (
            "pado.predictions.providers",
            "ImagePredictionsProviderStore",
        ),
    }[store_type]
    store_cls: Type[Store] = getattr(importlib.import_module(_module), _clsname)
    return store_cls


def get_store_info(
    urlpath: UrlpathLike,
    *,
    storage_options: dict[str, Any] | None = None,
) -> StoreInfo:
    """return the store information for a pado store"""
    md = get_store_metadata(urlpath, storage_options=storage_options)
    store_type = StoreType(md["store_type"])
    store_cls = _get_store_subclass(store_type)

    # get all versions
    store_version = int(md.get(store_cls.METADATA_KEY_STORE_VERSION, 0))
    provider_version = int(md.get(store_cls.METADATA_KEY_PROVIDER_VERSION, 0))
    data_version = int(md.get(store_cls.METADATA_KEY_DATA_VERSION, 0))
    data_identifier = md[store_cls.METADATA_KEY_IDENTIFIER]

    return StoreInfo(
        store_type=store_type,
        store_version=StoreVersionTuple(store_version, provider_version),
        data_version=DataVersionTuple(data_identifier, data_version),
    )


def migrate_store(
    urlpath: UrlpathLike,
    *,
    storage_options: dict[str, Any] | None = None,
    dry_run: bool = True,
) -> tuple[bool, StoreInfo, UrlpathLike]:
    """try migrating a store to a newer version"""
    store_info = get_store_info(urlpath, storage_options=storage_options)
    store_type = store_info.store_type
    store_cls = _get_store_subclass(store_type)

    _up = urlpath
    _so = storage_options

    migrations = find_migration_path(store_info, _STORE_MIGRATION_REGISTRY)
    if not migrations:
        return False, store_info, urlpath

    for info in migrations:
        m_func = _STORE_MIGRATION_REGISTRY[info]

        # we need to load the data with the correct store_version
        current_store_info = get_store_info(_up, storage_options=storage_options)
        target_store_info = info.can_migrate(current_store_info)
        if target_store_info is None:
            raise RuntimeError("can_migrate should not have returned None")
        load_store = store_cls(
            version=current_store_info.store_version.store, store_type=store_type
        )

        # load and migrate
        df, identifier, user_md = load_store.from_urlpath(_up, storage_options=_so)
        df, identifier, user_md = m_func(df, identifier, user_md)

        # we need to store the data with the correct store_version
        if (
            current_store_info.store_version.store
            == target_store_info.store_version.store
        ):
            save_store = load_store
        else:
            save_store = store_cls(
                version=info.target_store_version, store_type=store_type
            )

        # store temporary
        _up = f"memory://migrated-{target_store_info.to_string()}"
        _so = None
        save_store.to_urlpath(
            df,
            _up,
            identifier=identifier,
            storage_options=_so,
            **user_md,
        )

    if dry_run:
        return _up

    else:
        src_fs, src_pth = urlpathlike_to_fs_and_path(
            urlpath, storage_options=storage_options
        )
        if not src_fs.isfile(src_pth):
            raise NotImplementedError("only implemented for single file stores for now")
        else:
            # make a backup
            src_fs.copy(src_pth, f"{src_pth}.backup")
        m_of = urlpathlike_to_fsspec(_up, storage_options=_so)
        with m_of as m_f:
            with src_fs.open(src_pth, "wb") as f:
                for chunk in iter(lambda: m_f.read(2**20), ""):
                    f.write(chunk)
        return urlpath


def get_dataset_store_infos(ds: PadoDataset) -> dict[str, StoreInfo]:
    """gather store information for all stores in dataset"""
    # noinspection PyProtectedMember
    fs, get_fspath = ds._fs, ds._get_fspath

    return {
        path: get_store_info(fsopen(fs, path))
        for path in fs.glob(get_fspath("*.parquet"))
    }
