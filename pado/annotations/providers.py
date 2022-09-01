"""annotation provider"""
from __future__ import annotations

import uuid
from abc import ABC
from collections.abc import Collection
from itertools import repeat
from reprlib import Repr
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import MutableMapping
from typing import Optional
from typing import cast

import pandas as pd
from tqdm import tqdm

from pado._compat import cached_property
from pado.annotations.annotation import Annotations
from pado.annotations.formats import AnnotationModel
from pado.annotations.loaders import AnnotationsFromFileFunc
from pado.collections import validate_dataframe_index
from pado.images.ids import GetImageIdFunc
from pado.images.ids import ImageId
from pado.images.ids import match_partial_image_ids_reversed
from pado.io.files import find_files
from pado.io.store import Store
from pado.io.store import StoreType
from pado.types import UrlpathLike

# === storage =================================================================


class AnnotationProviderStore(Store):
    """stores the annotation provider in a single file with metadata"""

    METADATA_KEY_PROVIDER_VERSION = "annotation_version"
    ANNOTATION_VERSION = 1

    def __init__(self, version: int = 1, store_type: StoreType = StoreType.ANNOTATION):
        if store_type != StoreType.ANNOTATION:
            raise ValueError("changing store_type in subclasses unsupported")
        super().__init__(version=version, store_type=store_type)

    def __metadata_set_hook__(
        self, dct: Dict[bytes, bytes], setter: Callable[[dict, str, Any], None]
    ) -> None:
        setter(dct, self.METADATA_KEY_PROVIDER_VERSION, self.ANNOTATION_VERSION)

    def __metadata_get_hook__(
        self, dct: Dict[bytes, bytes], getter: Callable[[dict, str, Any], Any]
    ) -> Optional[dict]:
        image_provider_version = getter(dct, self.METADATA_KEY_PROVIDER_VERSION, None)
        if (
            image_provider_version is None
            or image_provider_version < self.ANNOTATION_VERSION
        ):
            raise RuntimeError("Please migrate AnnotationProvider to newer version.")
        elif image_provider_version > self.ANNOTATION_VERSION:
            raise RuntimeError(
                "AnnotationProvider is newer. Please upgrade pado to newer version."
            )
        return {self.METADATA_KEY_PROVIDER_VERSION: image_provider_version}


# === providers ===============================================================


class BaseAnnotationProvider(MutableMapping[ImageId, Annotations], ABC):
    """base class for annotation providers"""


_r = Repr()
_r.maxdict = 4


class AnnotationProvider(BaseAnnotationProvider):
    df: pd.DataFrame
    identifier: str

    def __init__(
        self,
        provider: BaseAnnotationProvider | pd.DataFrame | dict,
        *,
        identifier: Optional[str] = None,
    ):
        if isinstance(provider, AnnotationProvider):
            self._df = provider.df.copy()
            self.identifier = str(identifier) if identifier else provider.identifier
        elif isinstance(provider, pd.DataFrame):
            validate_dataframe_index(provider)
            self._df = provider.copy()
            self.identifier = str(identifier) if identifier else str(uuid.uuid4())
        elif isinstance(provider, (BaseAnnotationProvider, dict)):
            if not provider:
                self._df = pd.DataFrame(columns=AnnotationModel.__fields__)
            else:
                indices = []
                data = []
                for key, value in provider.items():
                    if value is None:
                        continue
                    indices.extend(repeat(ImageId.to_str(key), len(value)))
                    data.extend(a.to_record() for a in value)
                self._df = pd.DataFrame.from_records(
                    index=indices,
                    data=data,
                    columns=AnnotationModel.__fields__,
                )
            self.identifier = str(identifier) if identifier else str(uuid.uuid4())
        else:
            raise TypeError(
                f"expected `BaseAnnotationProvider`, got: {type(provider).__name__!r}"
            )

        self._store = {}

    @property
    def df(self):
        if not self._store:
            return self._df
        else:
            # fixme: for now the implementation does the right thing.
            #  but we should try to cache the dataframe based on some
            #  indicator if the Annotations have been modified or not.
            changed_iids = list(set(self._store))
            changed_labels = list(map(str, changed_iids))
            if not changed_iids:
                dfs = [self._df]
            else:
                dfs = [self._df.drop(index=changed_labels, errors="ignore")]
            for iid in changed_iids:
                df = self._store[iid].df
                df = df.set_index(pd.Index([iid.to_str()] * len(df)))
                dfs.append(df)
            return pd.concat(dfs)

    def __getitem__(self, image_id: ImageId) -> Annotations:
        if not isinstance(image_id, ImageId):
            raise TypeError(
                f"keys must be ImageId instances, got {type(image_id).__name__!r}"
            )
        try:
            return self._store[image_id]
        except KeyError:
            df = self.df.loc[
                [image_id.to_str()], :
            ]  # list: return DataFrame even if length == 1
            df = df.reset_index(drop=True)
            a = self._store[image_id] = Annotations(df, image_id=image_id)
            return a

    def __setitem__(self, image_id: ImageId, v: Annotations) -> None:
        if not isinstance(image_id, ImageId):
            raise TypeError(
                f"keys must be ImageId instances, got {type(image_id).__name__!r}"
            )
        if not isinstance(v, Annotations):
            raise TypeError(f"requires Annotations, got {type(v).__name__}")
        if v.image_id is None:
            v.image_id = image_id
        elif v.image_id != image_id:
            raise ValueError(f"image_ids don't match: {image_id!r} vs {v.image_id!r}")
        self._store[image_id] = v

    def __delitem__(self, image_id: ImageId) -> None:
        if not isinstance(image_id, ImageId):
            raise TypeError(
                f"keys must be ImageId instances, got {type(image_id).__name__!r}"
            )
        try:
            del self._store[image_id]
        except KeyError:
            had_store = False
        else:
            had_store = True
        try:
            self.df.drop(image_id.to_str(), inplace=True)
        except KeyError:
            had_df = False
        else:
            had_df = True
        if not had_store and not had_df:
            raise KeyError(image_id)

    def __len__(self) -> int:
        return self.df.index.nunique()

    def __iter__(self) -> Iterator[ImageId]:
        return iter(
            set(map(ImageId.from_str, self.df.index.unique())).union(self._store)
        )

    def __repr__(self):
        _akw = [_r.repr_dict(cast(dict, self), 0)]
        if self.identifier is not None:
            _akw.append(f"identifier={self.identifier!r}")
        return f"{type(self).__name__}({', '.join(_akw)})"

    def to_parquet(
        self, urlpath: UrlpathLike, *, storage_options: dict[str, Any] | None = None
    ) -> None:
        store = AnnotationProviderStore()
        dfs = []
        for image_id, annos in self.items():
            df = annos.df
            df = df.set_index(pd.Index([image_id.to_str()] * len(df)))
            dfs.append(df)
        if len(dfs) == 0:
            self._df = pd.DataFrame(columns=AnnotationModel.__fields__)
        else:
            self._df = pd.concat(dfs)
        store.to_urlpath(
            self.df,
            urlpath,
            identifier=self.identifier,
            storage_options=storage_options,
        )
        self._store.clear()

    @classmethod
    def from_parquet(cls, urlpath: UrlpathLike) -> AnnotationProvider:
        store = AnnotationProviderStore()
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
        inst = cls({}, identifier=identifier)
        inst._df = df
        return inst


class GroupedAnnotationProvider(AnnotationProvider):
    # todo: deduplicate

    # noinspection PyMissingConstructor
    def __init__(self, *providers: BaseAnnotationProvider):
        # super().__init__() ... violating liskov anyways ...
        self.providers = []
        for p in providers:
            if not isinstance(p, AnnotationProvider):
                p = AnnotationProvider(p)
            if isinstance(p, GroupedAnnotationProvider):
                self.providers.extend(p.providers)
            else:
                self.providers.append(p)
        self.identifier = "-".join(["grouped", *(p.identifier for p in self.providers)])

    @cached_property
    def df(self):
        return pd.concat([p.df for p in self.providers])

    def __getitem__(self, image_id: ImageId) -> Annotations:
        for ap in self.providers:
            try:
                return ap[image_id]
            except KeyError:
                pass
        raise KeyError(image_id)

    def __setitem__(self, image_id: ImageId, value: Annotations) -> None:
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
    def from_parquet(cls, urlpath: UrlpathLike) -> AnnotationProvider:
        raise NotImplementedError(f"unsupported operation for {cls.__name__!r}()")


# === manipulation ============================================================


def create_annotation_provider(
    search_urlpath: UrlpathLike,
    search_glob: str,
    *,
    output_urlpath: Optional[UrlpathLike],
    image_id_func: GetImageIdFunc,
    annotations_func: AnnotationsFromFileFunc,
    identifier: Optional[str] = None,
    resume: bool = False,
    valid_image_ids: Optional[Collection[ImageId]] = None,
    progress: bool = False,
) -> AnnotationProvider:
    """create an annotation provider from a directory containing annotations"""
    files_and_parts = find_files(search_urlpath, glob=search_glob)

    if resume:
        ap = AnnotationProvider.from_parquet(urlpath=output_urlpath)
    else:
        ap = AnnotationProvider({}, identifier=identifier)

    if progress:
        files_and_parts = tqdm(files_and_parts)

    try:
        for fp in files_and_parts:
            image_id = image_id_func(fp.file, fp.parts, ap.identifier)
            if image_id is None:
                continue  # skip if no image_id is returned
            if valid_image_ids is not None:
                if image_id not in valid_image_ids:
                    # try matching partially
                    image_id = match_partial_image_ids_reversed(
                        valid_image_ids, image_id
                    )
                    if image_id is None:
                        continue  # skip if image_id not in image_id_filter
            if resume and image_id in ap:
                continue  # skip if we resume and already have the annotations
            anno = annotations_func(fp.file)
            if anno is None:
                continue  # skip id no annotations are returned
            ap[image_id] = anno

    finally:
        if output_urlpath is not None:
            ap.to_parquet(output_urlpath)

    return ap
