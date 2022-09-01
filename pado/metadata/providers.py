"""pado.meta.store

provides a single file parquet store for pd.DataFrames with per store metadata
"""
from __future__ import annotations

import uuid
from abc import ABC
from reprlib import Repr
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Iterator
from typing import MutableMapping
from typing import Optional
from typing import cast

import pandas as pd

from pado._compat import cached_property
from pado.collections import validate_dataframe_index
from pado.images.ids import GetImageIdFunc
from pado.images.ids import ImageId
from pado.io.store import Store
from pado.io.store import StoreType
from pado.types import UrlpathLike

# === storage =================================================================


class MetadataProviderStore(Store):
    """stores the metadata in a single file with per store metadata"""

    METADATA_KEY_PROVIDER_VERSION = "dataset_version"
    DATASET_VERSION = 1

    def __init__(self, version: int = 1, store_type: StoreType = StoreType.METADATA):
        if store_type != StoreType.METADATA:
            raise ValueError("changing store_type in subclasses unsupported")
        super().__init__(version=version, store_type=store_type)

    def __metadata_set_hook__(
        self, dct: Dict[bytes, bytes], setter: Callable[[dict, str, Any], None]
    ) -> None:
        setter(dct, self.METADATA_KEY_PROVIDER_VERSION, self.DATASET_VERSION)

    def __metadata_get_hook__(
        self, dct: Dict[bytes, bytes], getter: Callable[[dict, str, Any], Any]
    ) -> Optional[dict]:
        dataset_version = getter(dct, self.METADATA_KEY_PROVIDER_VERSION, None)
        if dataset_version is None or dataset_version < self.DATASET_VERSION:
            raise RuntimeError("Please migrate MetadataProviderStore to newer version.")
        elif dataset_version > self.DATASET_VERSION:
            raise RuntimeError(
                "MetadataProviderStore is newer. Please upgrade pado to newer version."
            )
        return {self.METADATA_KEY_PROVIDER_VERSION: dataset_version}


# === provider ================================================================


class BaseMetadataProvider(MutableMapping[ImageId, pd.DataFrame], ABC):
    """base class for metadata providers"""


_r = Repr()
_r.maxdict = 4


class MetadataProvider(BaseMetadataProvider):
    df: pd.DataFrame
    identifier: str

    def __init__(
        self,
        provider: BaseMetadataProvider | pd.DataFrame | dict,
        *,
        identifier: Optional[str] = None,
    ) -> None:
        if isinstance(provider, MetadataProvider):
            self.df = provider.df.copy()
            self.identifier = str(identifier) if identifier else provider.identifier
        elif isinstance(provider, pd.DataFrame):
            validate_dataframe_index(provider)
            self.df = provider.copy()
            self.identifier = str(identifier) if identifier else str(uuid.uuid4())
        elif isinstance(provider, (BaseMetadataProvider, dict)):
            if not provider:
                self.df = pd.DataFrame(index=[], data={})
                self.identifier = str(identifier) if identifier else str(uuid.uuid4())
            else:
                columns = set()
                dfs = []
                for image_id, df in provider.items():
                    if df.empty:
                        continue
                    ids = set(df.index.unique())
                    if len(ids) > 2:
                        raise ValueError(f"image_ids in provider not unique: {ids!r}")
                    image_id_str = image_id.to_str()
                    if {image_id_str} == ids:
                        pass
                    elif {None, image_id_str}.issuperset(ids):
                        index = df.index.fillna(image_id_str)
                        df = df.set_index(index)
                    else:
                        raise AssertionError(f"{image_id_str} with Index: {ids!r}")
                    dfs.append(df)
                    columns.add(frozenset(df.columns))
                if len(columns) != 1:
                    raise RuntimeError(
                        f"dataframe columns in provider don't match {columns!r}"
                    )
                self.df = pd.concat(dfs)
            self.identifier = str(identifier) if identifier else str(uuid.uuid4())
        else:
            raise TypeError(
                f"expected `BaseMetadataProvider`, got: {type(provider).__name__!r}"
            )

    def __getitem__(self, image_id: ImageId) -> pd.DataFrame:
        if not isinstance(image_id, ImageId):
            raise TypeError(
                f"keys must be ImageId instances, got {type(image_id).__name__!r}"
            )
        return self.df.loc[[image_id.to_str()]]

    def __setitem__(self, image_id: ImageId, value: pd.DataFrame) -> None:
        if not isinstance(image_id, ImageId):
            raise TypeError(
                f"keys must be ImageId instances, got {type(image_id).__name__!r}"
            )
        if not value.columns == self.df.columns:
            raise ValueError("dataframe columns do not match")
        self.df = pd.concat(
            [
                self.df.drop(image_id.to_str()),
                value.set_index(pd.Index([image_id.to_str()] * len(value))),
            ]
        )

    def __delitem__(self, image_id: ImageId) -> None:
        if not isinstance(image_id, ImageId):
            raise TypeError(
                f"keys must be ImageId instances, got {type(image_id).__name__!r}"
            )
        self.df.drop(image_id.to_str(), inplace=True)

    def __repr__(self):
        _akw = [_r.repr_dict(cast(dict, self), 0)]
        if self.identifier is not None:
            _akw.append(f"identifier={self.identifier!r}")
        return f"{type(self).__name__}({', '.join(_akw)})"

    def __len__(self) -> int:
        return self.df.index.nunique(dropna=True)

    def __iter__(self) -> Iterator[ImageId]:
        return map(ImageId.from_str, self.df.index.unique())

    def to_parquet(
        self, urlpath: UrlpathLike, *, storage_options: dict[str, Any] | None = None
    ) -> None:
        store = MetadataProviderStore()
        store.to_urlpath(
            self.df,
            urlpath,
            identifier=self.identifier,
            storage_options=storage_options,
        )

    @classmethod
    def from_parquet(cls, urlpath: UrlpathLike) -> MetadataProvider:
        store = MetadataProviderStore()
        df, identifier, user_metadata = store.from_urlpath(urlpath)
        if {
            store.METADATA_KEY_STORE_TYPE,
            store.METADATA_KEY_STORE_VERSION,
            store.METADATA_KEY_PADO_VERSION,
            store.METADATA_KEY_PROVIDER_VERSION,
            store.METADATA_KEY_CREATED_AT,
            store.METADATA_KEY_CREATED_BY,
        } != set(user_metadata):
            raise NotImplementedError(f"currently unused {user_metadata!r}")
        inst = cls.__new__(cls)
        inst.df = df
        inst.identifier = identifier
        return inst


class GroupedMetadataProvider(MetadataProvider):
    # todo: deduplicate

    # noinspection PyMissingConstructor
    def __init__(self, *providers: BaseMetadataProvider):
        # super().__init__() ... violating Liskov anyways ...
        self.providers = []
        for p in providers:
            if not isinstance(p, MetadataProvider):
                p = MetadataProvider(p)
            if isinstance(p, GroupedMetadataProvider):
                self.providers.extend(p.providers)
            else:
                self.providers.append(p)
        self.is_standardized = len({tuple(p.df.columns) for p in self.providers}) == 1
        self.identifier = "-".join(["grouped", *(p.identifier for p in self.providers)])

    @cached_property
    def df(self):
        if not self.is_standardized:
            raise RuntimeError(
                "can't access a combined pd.DataFrame on a non standardized "
            )
        return pd.concat([p.df for p in self.providers])

    def __getitem__(self, image_id: ImageId) -> pd.DataFrame:
        for ap in self.providers:
            try:
                return ap[image_id]
            except KeyError:
                pass
        raise KeyError(image_id)

    def __setitem__(self, image_id: ImageId, value: pd.DataFrame) -> None:
        raise RuntimeError("can't add new item to GroupedImageProvider")

    def __delitem__(self, image_id: ImageId) -> None:
        raise RuntimeError("can't delete from {type(self).__name__}")

    def __len__(self) -> int:
        return len(set().union(*self.providers))

    def __iter__(self) -> Iterator[ImageId]:
        d = {}
        for provider in reversed(self.providers):
            d.update(dict.fromkeys(provider))
        return iter(d)

    def __repr__(self):
        return f'{type(self).__name__}({", ".join(map(repr, self.providers))})'

    def to_parquet(
        self, urlpath: UrlpathLike, *, storage_options: dict[str, Any] | None = None
    ) -> None:
        super().to_parquet(urlpath, storage_options=storage_options)

    @classmethod
    def from_parquet(cls, urlpath: UrlpathLike) -> MetadataProvider:
        raise NotImplementedError(f"unsupported operation for {cls.__name__!r}()")


# === manipulation ============================================================


MetadataFromFileFunc = Callable[[UrlpathLike], Optional[pd.DataFrame]]


def create_metadata_provider(
    search_urlpath: UrlpathLike,
    search_glob: str,
    *,
    output_urlpath: Optional[UrlpathLike],
    image_id_func: GetImageIdFunc,
    metadata_func: MetadataFromFileFunc,
    identifier: Optional[str] = None,
    resume: bool = False,
    valid_image_ids: Optional[Collection[ImageId]] = None,
    progress: bool = False,
) -> MetadataProvider:
    """create an metadata provider from a directory containing metadata"""
    raise NotImplementedError("todo")
