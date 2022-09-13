from __future__ import annotations

import os.path
import sys
import uuid
import warnings
from abc import ABC
from reprlib import Repr
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import MutableMapping
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import cast

import numpy as np
import pandas as pd
from fsspec.core import OpenFile
from fsspec.implementations.cached import CachingFileSystem
from fsspec.implementations.cached import SimpleCacheFileSystem
from fsspec.implementations.cached import WholeFileCacheFileSystem
from fsspec.implementations.local import LocalFileSystem
from tqdm import tqdm

from pado._compat import cached_property
from pado.collections import validate_dataframe_index
from pado.images.ids import ImageId
from pado.images.image import Image
from pado.io.files import find_files
from pado.io.files import urlpathlike_to_fs_and_path
from pado.io.files import urlpathlike_to_fsspec
from pado.io.files import urlpathlike_to_string
from pado.io.paths import match_partial_paths_reversed
from pado.io.store import Store
from pado.io.store import StoreType
from pado.types import UrlpathLike


def __getattr__(name):
    """compatibility"""
    if name == "create_image_provider":
        from pado.create import create_image_provider

        warnings.warn(
            "moved: `from pado.create import create_image_provider`",
            DeprecationWarning,
            stacklevel=2,
        )

        return create_image_provider
    else:
        raise AttributeError(name)


# === storage =================================================================


class ImageProviderStore(Store):
    """stores the image provider in a single file with metadata"""

    METADATA_KEY_PROVIDER_VERSION = "image_provider_version"
    PROVIDER_VERSION = 1

    def __init__(self, version: int = 1, store_type: StoreType = StoreType.IMAGE):
        if store_type != StoreType.IMAGE:
            raise ValueError("changing store_type in subclasses unsupported")
        super().__init__(version=version, store_type=store_type)

    def __metadata_set_hook__(
        self, dct: Dict[bytes, bytes], setter: Callable[[dict, str, Any], None]
    ) -> None:
        setter(dct, self.METADATA_KEY_PROVIDER_VERSION, self.PROVIDER_VERSION)

    def __metadata_get_hook__(
        self, dct: Dict[bytes, bytes], getter: Callable[[dict, str, Any], Any]
    ) -> Optional[dict]:
        image_provider_version = getter(dct, self.METADATA_KEY_PROVIDER_VERSION, None)
        if (
            image_provider_version is None
            or image_provider_version < self.PROVIDER_VERSION
        ):
            raise RuntimeError("Please migrate ImageProvider to newer version.")
        elif image_provider_version > self.PROVIDER_VERSION:
            raise RuntimeError(
                "ImageProvider is newer. Please upgrade pado to newer version."
            )
        return {self.METADATA_KEY_PROVIDER_VERSION: image_provider_version}


# === providers ===============================================================


class BaseImageProvider(MutableMapping[ImageId, Image], ABC):
    """base class for image providers"""


# noinspection PyUnresolvedReferences
BaseImageProvider.register(dict)

_r = Repr()
_r.maxdict = 4


class ImageProvider(BaseImageProvider):
    df: pd.DataFrame
    identifier: str

    def __init__(
        self,
        provider: BaseImageProvider | pd.DataFrame | dict | None = None,
        *,
        identifier: Optional[str] = None,
    ):
        if provider is None:
            provider = {}

        if isinstance(provider, ImageProvider):
            self.df = provider.df.copy()
            self.identifier = str(identifier) if identifier else provider.identifier
        elif isinstance(provider, pd.DataFrame):
            validate_dataframe_index(provider, unique_index=True)
            self.df = provider.copy()
            self.identifier = str(identifier) if identifier else str(uuid.uuid4())
        elif isinstance(provider, BaseImageProvider):
            if not provider:
                self.df = pd.DataFrame(columns=Image.__fields__)
            else:
                self.df = pd.DataFrame.from_records(
                    index=list(map(ImageId.to_str, provider.keys())),
                    data=list(map(lambda x: x.to_record(), provider.values())),
                    columns=Image.__fields__,
                )
            self.identifier = str(identifier) if identifier else str(uuid.uuid4())
        else:
            raise TypeError(
                f"expected `BaseImageProvider`, got: {type(provider).__name__!r}"
            )

    def __getitem__(self, image_id: ImageId) -> Image:
        if not isinstance(image_id, ImageId):
            raise TypeError(
                f"keys must be ImageId instances, got {type(image_id).__name__!r}"
            )
        row = self.df.loc[image_id.to_str()]
        return Image.from_obj(row)

    def __setitem__(self, image_id: ImageId, image: Image) -> None:
        if not isinstance(image_id, ImageId):
            raise TypeError(
                f"keys must be ImageId instances, got {type(image_id).__name__!r}"
            )
        if not isinstance(image, Image):
            raise TypeError(
                f"values must be Image instances, got {type(image).__name__!r}"
            )
        dct = image.to_record()
        self.df.loc[image_id.to_str()] = pd.Series(dct)

    def __delitem__(self, image_id: ImageId) -> None:
        if not isinstance(image_id, ImageId):
            raise TypeError(
                f"keys must be ImageId instances, got {type(image_id).__name__!r}"
            )
        self.df.drop(image_id.to_str(), inplace=True)

    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self) -> Iterator[ImageId]:
        return iter(map(ImageId.from_str, self.df.index))

    def items(self) -> Iterator[Tuple[ImageId, Image]]:
        for row in self.df.itertuples(index=True, name="ImageAsRow"):
            # noinspection PyProtectedMember
            x = row._asdict()
            i = x.pop("Index")
            yield ImageId.from_str(i), Image.from_obj(x)

    def __repr__(self):
        _akw = [_r.repr_dict(cast(dict, self), 0)]
        if self.identifier is not None:
            _akw.append(f"identifier={self.identifier!r}")
        return f"{type(self).__name__}({', '.join(_akw)})"

    def to_parquet(
        self, urlpath: UrlpathLike, *, storage_options: dict[str, Any] | None = None
    ) -> None:
        store = ImageProviderStore()
        store.to_urlpath(
            self.df,
            urlpath,
            identifier=self.identifier,
            storage_options=storage_options,
        )

    @classmethod
    def from_parquet(cls, urlpath: UrlpathLike) -> ImageProvider:
        store = ImageProviderStore()
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


class GroupedImageProvider(ImageProvider):
    # noinspection PyMissingConstructor
    def __init__(self, *providers: BaseImageProvider):
        # super().__init__() ... violating Liskov anyways...
        self.providers = []
        for p in providers:
            if not isinstance(p, ImageProvider):
                p = ImageProvider(p)
            if isinstance(p, GroupedImageProvider):
                self.providers.extend(p.providers)
            else:
                self.providers.append(p)
        self.identifier = "-".join(["grouped", *(p.identifier for p in self.providers)])

    @cached_property
    def df(self):
        return pd.concat([p.df for p in self.providers])

    def __getitem__(self, image_id: ImageId) -> Image:
        for ip in self.providers:
            try:
                return ip[image_id]
            except KeyError:
                pass
        raise KeyError(image_id)

    def __setitem__(self, image_id: ImageId, image: Image) -> None:
        for ip in self.providers:
            if image_id in ip:
                ip[image_id] = image
                break
        raise RuntimeError("can't add new item to GroupedImageProvider")

    def __delitem__(self, image_id: ImageId) -> None:
        raise RuntimeError("can't delete from GroupedImageProvider")

    def __len__(self) -> int:
        return len(set().union(*self.providers))

    def __iter__(self) -> Iterator[ImageId]:
        d = {}
        for provider in reversed(self.providers):
            d.update(dict.fromkeys(provider))
        return iter(d)

    def items(self) -> Iterator[Tuple[ImageId, Image]]:
        return super().items()

    def __repr__(self):
        return f'{type(self).__name__}({", ".join(map(repr, self.providers))})'

    def to_parquet(
        self, urlpath: UrlpathLike, *, storage_options: dict[str, Any] | None = None
    ) -> None:
        super().to_parquet(urlpath, storage_options=storage_options)

    @classmethod
    def from_parquet(cls, urlpath: UrlpathLike) -> ImageProvider:
        raise NotImplementedError(f"unsupported operation for {cls.__name__!r}()")


class FilteredImageProvider(ImageProvider):
    def __init__(
        self,
        provider: BaseImageProvider,
        *,
        valid_keys: Optional[Iterable[ImageId]] = None,
    ):
        super().__init__()
        self._provider = ImageProvider(provider)
        self._vk = set(self._provider) if valid_keys is None else set(valid_keys)

    @cached_property
    def df(self):
        return self._provider.df.filter(items=self._vk, axis="index")

    @property
    def valid_keys(self) -> Set[ImageId]:
        return self._vk

    def __getitem__(self, image_id: ImageId) -> Image:
        if image_id not in self._vk:
            raise KeyError(image_id)
        return self._provider[image_id]

    def __setitem__(self, image_id: ImageId, image: Image) -> None:
        raise NotImplementedError("can't add to FilteredImageProvider")

    def __delitem__(self, image_id: ImageId) -> None:
        raise NotImplementedError("can't delete from FilteredImageProvider")

    def __len__(self) -> int:
        return len(self.valid_keys.intersection(self._provider))

    def __iter__(self) -> Iterator[ImageId]:
        return iter(map(ImageId.from_str, self.df.index))  # fixme

    def items(self) -> Iterator[Tuple[ImageId, Image]]:
        return super().items()

    def __repr__(self):
        return f"{type(self).__name__}({self._provider!r})"

    def to_parquet(
        self, urlpath: UrlpathLike, *, storage_options: dict[str, Any] | None = None
    ) -> None:
        super().to_parquet(urlpath, storage_options=storage_options)

    @classmethod
    def from_parquet(cls, urlpath: UrlpathLike) -> ImageProvider:
        raise NotImplementedError(f"unsupported operation for {cls.__name__!r}()")


class LocallyCachedImageProvider(ImageProvider):
    """image provider that prepends a fsspec CachingFileSystem

    use to route a normal ImageProvider through a local cache
    """

    def __init__(
        self,
        provider: BaseImageProvider | pd.DataFrame | dict | None = None,
        *,
        identifier: Optional[str] = None,
        cache_cls: Type[CachingFileSystem] = WholeFileCacheFileSystem,
        **cache_kwargs,
    ):
        super().__init__(provider, identifier=identifier)
        self._cache_cls = cache_cls
        self._cache_kw = cache_kwargs

    def _prepend_cache(self, urlpath: UrlpathLike) -> UrlpathLike:
        """prepend the cache to the urlpath"""
        fs, path = urlpathlike_to_fs_and_path(urlpath)
        cached_fs = self._cache_cls(**self._cache_kw, fs=fs)
        return OpenFile(fs=cached_fs, path=path)

    def __getitem__(self, image_id: ImageId) -> Image:
        image = super().__getitem__(image_id)
        image.urlpath = self._prepend_cache(image.urlpath)
        return image

    def items(self) -> Iterator[Tuple[ImageId, Image]]:
        for row in self.df.itertuples(index=True, name="ImageAsRow"):
            # noinspection PyProtectedMember
            x = row._asdict()
            i = x.pop("Index")
            image = Image.from_obj(x)
            image.urlpath = self._prepend_cache(image.urlpath)
            yield ImageId.from_str(i),

    def __setitem__(self, image_id: ImageId, image: Image) -> None:
        raise NotImplementedError(f"can't add to {type(self).__name__}")

    def __delitem__(self, image_id: ImageId) -> None:
        raise NotImplementedError(f"can't delete from {type(self).__name__}")

    def to_parquet(
        self, urlpath: UrlpathLike, *, storage_options: dict[str, Any] | None = None
    ) -> None:
        raise NotImplementedError


def image_is_cached_or_local(image: Image) -> bool:
    """check if an image can be accessed locally"""
    fs, path = urlpathlike_to_fs_and_path(image.urlpath)
    if isinstance(fs, LocalFileSystem):
        return os.path.isfile(path)
    elif isinstance(fs, SimpleCacheFileSystem):
        # noinspection PyProtectedMember
        return fs._check_file(path) is not None
    elif isinstance(fs, CachingFileSystem):
        # noinspection PyProtectedMember
        return fs._check_file(path) is not False
    else:
        return False


def image_cached_percentage(image: Image) -> float:
    """return how much of an image is currently cached"""
    fs, path = urlpathlike_to_fs_and_path(image.urlpath)
    if isinstance(fs, LocalFileSystem):
        return 100.0
    elif isinstance(fs, CachingFileSystem):
        # noinspection PyProtectedMember
        sha = fs.hash_name(path, fs.same_names)
        fn = os.path.join(fs.storage[-1], sha)
        if not os.path.exists(fn):
            return 0.0
        else:
            cached_bytes = os.stat(fn).st_size
            image_bytes = image.file_info.size_bytes.to("b")
            return min(100.0 * cached_bytes / image_bytes, 100.0)
    else:
        return 0.0


# === manipulation ============================================================


_PT = TypeVar("_PT", bound="ImageProvider")


def update_image_provider_urlpaths(
    search_urlpath: UrlpathLike,
    search_glob: str,
    *,
    provider: _PT | UrlpathLike,
    inplace: bool = False,
    ignore_ambiguous: bool = False,
    progress: bool = False,
    provider_cls: Type[_PT] = ImageProvider,
    storage_options: dict[str, Any] | None = None,
) -> _PT:
    """search a path and re-associate image urlpaths by filename"""
    files_and_parts = find_files(
        search_urlpath, glob=search_glob, storage_options=storage_options
    )
    if progress:
        print(
            f"[info] found {len(files_and_parts)} new files matching the pattern",
            file=sys.stderr,
            flush=True,
        )

    if len(files_and_parts) == 0:
        raise FileNotFoundError("no files found")

    if isinstance(provider, provider_cls):
        ip = provider
    else:
        ip = provider_cls.from_parquet(urlpath=provider)

    if progress:
        print(
            f"[info] provider has {len(ip)} images",
            file=sys.stderr,
            flush=True,
        )

    new_urlpaths = match_partial_paths_reversed(
        current_urlpaths=ip.df.urlpath,
        new_urlpaths=list(x.file for x in files_and_parts),
        ignore_ambiguous=ignore_ambiguous,
        progress=progress,
    )

    old = ip.df.urlpath.copy()
    ip.df.loc[:, "urlpath"] = [urlpathlike_to_string(p) for p in new_urlpaths]

    num_updated = np.sum(old.values != ip.df.urlpath.values)
    if progress:
        print(
            f"[info] re-associated {num_updated} images",
            file=sys.stderr,
            flush=True,
        )

    if inplace and not isinstance(provider, provider_cls) and num_updated:
        ip.to_parquet(provider)
    return ip


def copy_image(
    provider: ImageProvider,
    image_id: ImageId | Iterable[ImageId],
    dest: UrlpathLike,
    *,
    update_provider: bool = True,
    progress: bool = False,
    chunk_size: int = 2**20,
    **update_kwargs,
) -> None:
    """copy image data to a new location"""
    if not isinstance(provider, ImageProvider):
        raise TypeError(
            f"expected ImageProvider instance, got {type(provider).__name__}"
        )

    # prepare image ids
    if isinstance(image_id, ImageId):
        image_ids = [image_id]
    else:
        image_ids = list(image_id)

    if progress and len(image_ids) > 1:
        image_ids = tqdm(image_ids, desc="images", disable=not progress)

    # prepare destination dir
    dst_fs, dst_root = urlpathlike_to_fs_and_path(dest)
    if not dst_fs.isdir(dst_root):
        dst_fs.mkdir(dst_root, create_parents=True)

    # copy images
    for image_id in image_ids:
        image = provider[image_id]

        dst_fn = os.path.join(dst_root, image_id.site, *image_id.parts)
        dst_dir = os.path.dirname(dst_fn)

        with urlpathlike_to_fsspec(image.urlpath, mode="rb") as src:
            dst_fs.mkdir(dst_dir, create_parents=True)
            try:
                with dst_fs.open(dst_fn, mode="wb") as dst:

                    prg = tqdm(
                        desc="copy",
                        total=image.file_info.size_bytes,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        disable=not progress,
                    )

                    # iterate in chunks
                    for chunk in iter(lambda: src.read(chunk_size), b""):
                        dst.write(chunk)
                        prg.update(chunk_size)

            except Exception:
                dst_fs.mv(dst_fn, f"{dst_fn}.broken.partial")
                raise

    if update_provider:
        update_image_provider_urlpaths(
            dest, "**/*.svs", provider=provider, progress=progress, **update_kwargs
        )
