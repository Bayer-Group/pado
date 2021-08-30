"""pado.meta.store

provides a single file parquet store for pd.DataFrames with per store metadata
"""
from __future__ import annotations

import uuid
from abc import ABC
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Iterator
from typing import MutableMapping
from typing import Optional
from typing import Union

import pandas as pd

from pado.images import ImageId
from pado.images.ids import GetImageIdFunc
from pado.io.store import Store
from pado.io.store import StoreType
from pado.types import UrlpathLike


# === storage =================================================================

class MetadataStore(Store):
    """stores the metadata in a single file with per store metadata"""
    METADATA_KEY_DATASET_VERSION = "dataset_version"
    DATASET_VERSION = 1

    def __init__(self):
        super().__init__(version=1, store_type=StoreType.METADATA)

    def __metadata_set_hook__(self, dct: Dict[bytes, bytes], setter: Callable[[dict, str, Any], None]) -> None:
        setter(dct, self.METADATA_KEY_DATASET_VERSION, self.DATASET_VERSION)

    def __metadata_get_hook__(self, dct: Dict[bytes, bytes], getter: Callable[[dict, str, Any], Any]) -> Optional[dict]:
        dataset_version = getter(dct, self.METADATA_KEY_DATASET_VERSION, None)
        if dataset_version is None or dataset_version < self.DATASET_VERSION:
            raise RuntimeError("Please migrate MetadataStore to newer version.")
        elif dataset_version > self.DATASET_VERSION:
            raise RuntimeError("MetadataStore is newer. Please upgrade pado to newer version.")
        return {
            self.METADATA_KEY_DATASET_VERSION: dataset_version
        }


# === provider ================================================================

class BaseMetadataProvider(MutableMapping[ImageId, pd.DataFrame], ABC):
    """base class for metadata providers"""


class MetadataProvider(BaseMetadataProvider):
    df: pd.DataFrame
    identifier: str

    def __init__(
        self,
        provider: BaseMetadataProvider | pd.DataFrame | dict | None = None,
        *,
        identifier: Optional[str] = None,
    ) -> None:
        if provider is None:
            provider = {}

        if isinstance(provider, MetadataProvider):
            self.df = provider.df.copy()
            self.identifier = str(identifier) if identifier else provider.identifier
        elif isinstance(provider, pd.DataFrame):
            try:
                _ = map(ImageId.from_str, provider.index)
            except (TypeError, ValueError):
                raise ValueError("provider dataframe index has non ImageId indices")
            self.df = provider.copy()
            self.identifier = str(identifier) if identifier else str(uuid.uuid4())
        elif isinstance(provider, (BaseMetadataProvider, dict)):
            if not provider:
                raise ValueError("can't create from an empty MetadataProvider")
            else:
                columns = set()
                dfs = []
                for image_id, df in provider.items():
                    if df.empty:
                        continue
                    ids = set(df.index.unique())
                    assert len(ids) <= 2
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
                assert len(columns) == 1, f"dataframe columns in provider don't match {columns!r}"
                self.df = pd.concat(dfs)
            self.identifier = str(identifier) if identifier else str(uuid.uuid4())
        else:
            raise TypeError(f"expected `BaseMetadataProvider`, got: {type(provider).__name__!r}")

    def __getitem__(self, image_id: ImageId) -> pd.DataFrame:
        if not isinstance(image_id, ImageId):
            raise TypeError(f"keys must be ImageId instances, got {type(image_id).__name__!r}")
        return self.df.loc[[image_id.to_str()]]

    def __setitem__(self, image_id: ImageId, value: pd.DataFrame) -> None:
        if not isinstance(image_id, ImageId):
            raise TypeError(f"keys must be ImageId instances, got {type(image_id).__name__!r}")
        if not value.columns == self.df.columns:
            raise ValueError("dataframe columns do not match")
        self.df = pd.concat([
            self.df.drop(image_id.to_str()),
            value.set_index(pd.Index([image_id.to_str()] * len(value))),
        ])

    def __delitem__(self, image_id: ImageId) -> None:
        if not isinstance(image_id, ImageId):
            raise TypeError(f"keys must be ImageId instances, got {type(image_id).__name__!r}")
        self.df.drop(image_id.to_str(), inplace=True)

    def __repr__(self):
        return f'{type(self).__name__}({self.identifier!r})'

    def __len__(self) -> int:
        return self.df.index.nunique(dropna=True)

    def __iter__(self) -> Iterator[ImageId]:
        return map(ImageId.from_str, self.df.index.unique())

    def to_parquet(self, urlpath: UrlpathLike) -> None:
        store = MetadataStore()
        store.to_urlpath(self.df, urlpath, identifier=self.identifier)

    @classmethod
    def from_parquet(cls, urlpath: UrlpathLike) -> MetadataProvider:
        store = MetadataStore()
        df, identifier, user_metadata = store.from_urlpath(urlpath)
        assert {
                   store.METADATA_KEY_STORE_TYPE,
                   store.METADATA_KEY_STORE_VERSION,
                   store.METADATA_KEY_PADO_VERSION,
                   store.METADATA_KEY_DATASET_VERSION,
                   store.METADATA_KEY_CREATED_AT,
                   store.METADATA_KEY_CREATED_BY,
               } == set(user_metadata), f"currently unused {user_metadata!r}"
        inst = cls.__new__(cls)
        inst.df = df
        inst.identifier = identifier
        return inst


class GroupedMetadataProvider(MetadataProvider):
    # todo: deduplicate

    def __init__(self, *providers: BaseMetadataProvider):
        super().__init__()
        self.providers = []
        for p in providers:
            if not isinstance(p, MetadataProvider):
                p = MetadataProvider(p)
            if isinstance(p, GroupedMetadataProvider):
                self.providers.extend(p.providers)
            else:
                self.providers.append(p)
        self.is_standardized = len({tuple(p.df.columns) for p in self.providers}) == 1

    def df(self):
        if not self.is_standardized:
            raise RuntimeError("can't access a combined pd.DataFrame on a non standardized ")
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

    def to_parquet(self, urlpath: UrlpathLike) -> None:
        super().to_parquet(urlpath)

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