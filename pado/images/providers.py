from __future__ import annotations

import os.path
import uuid
from abc import ABC
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type

import numpy as np
import pandas as pd
from fsspec.core import OpenFile
from fsspec.implementations.cached import CachingFileSystem
from fsspec.implementations.cached import SimpleCacheFileSystem
from fsspec.implementations.cached import WholeFileCacheFileSystem
from fsspec.implementations.local import LocalFileSystem
from tqdm import tqdm

from pado._compat import cached_property
from pado.images.ids import GetImageIdFunc
from pado.images.ids import ImageId
from pado.images.ids import image_id_from_parts
from pado.images.image import Image
from pado.io.files import find_files
from pado.io.files import urlpathlike_to_fs_and_path
from pado.io.files import urlpathlike_to_fsspec
from pado.io.files import urlpathlike_to_string
from pado.io.paths import match_partial_paths_reversed
from pado.io.store import Store
from pado.io.store import StoreType
from pado.types import UrlpathLike


# === storage =================================================================

class ImageProviderStore(Store):
    """stores the image provider in a single file with metadata"""
    METADATA_KEY_PROVIDER_VERSION = "image_provider_version"
    PROVIDER_VERSION = 1

    def __init__(self):
        super().__init__(version=1, store_type=StoreType.IMAGE)

    def __metadata_set_hook__(self, dct: Dict[bytes, bytes], setter: Callable[[dict, str, Any], None]) -> None:
        setter(dct, self.METADATA_KEY_PROVIDER_VERSION, self.PROVIDER_VERSION)

    def __metadata_get_hook__(self, dct: Dict[bytes, bytes], getter: Callable[[dict, str, Any], Any]) -> Optional[dict]:
        image_provider_version = getter(dct, self.METADATA_KEY_PROVIDER_VERSION, None)
        if image_provider_version is None or image_provider_version < self.PROVIDER_VERSION:
            raise RuntimeError("Please migrate ImageProvider to newer version.")
        elif image_provider_version > self.PROVIDER_VERSION:
            raise RuntimeError("ImageProvider is newer. Please upgrade pado to newer version.")
        return {
            self.METADATA_KEY_PROVIDER_VERSION: image_provider_version
        }


# === providers ===============================================================

class BaseImageProvider(MutableMapping[ImageId, Image], ABC):
    """base class for image providers"""


# noinspection PyUnresolvedReferences
BaseImageProvider.register(dict)


class ImageProvider(BaseImageProvider):
    df: pd.DataFrame
    identifier: str

    def __init__(self, provider: BaseImageProvider | pd.DataFrame | dict | None = None, *, identifier: Optional[str] = None):
        if provider is None:
            provider = {}

        if isinstance(provider, ImageProvider):
            self.df = provider.df.copy()
            self.identifier = str(identifier) if identifier else provider.identifier
        elif isinstance(provider, pd.DataFrame):
            try:
                _ = map(ImageId.from_str, provider.index)
            except (TypeError, ValueError):
                raise ValueError("provider dataframe index has non ImageId indices")
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
            raise TypeError(f"expected `BaseImageProvider`, got: {type(provider).__name__!r}")

    def __getitem__(self, image_id: ImageId) -> Image:
        if not isinstance(image_id, ImageId):
            raise TypeError(f"keys must be ImageId instances, got {type(image_id).__name__!r}")
        row = self.df.loc[image_id.to_str()]
        return Image.from_obj(row)

    def __setitem__(self, image_id: ImageId, image: Image) -> None:
        if not isinstance(image_id, ImageId):
            raise TypeError(f"keys must be ImageId instances, got {type(image_id).__name__!r}")
        if not isinstance(image, Image):
            raise TypeError(f"values must be Image instances, got {type(image).__name__!r}")
        dct = image.to_record()
        self.df.loc[image_id.to_str()] = pd.Series(dct)

    def __delitem__(self, image_id: ImageId) -> None:
        if not isinstance(image_id, ImageId):
            raise TypeError(f"keys must be ImageId instances, got {type(image_id).__name__!r}")
        self.df.drop(image_id.to_str(), inplace=True)

    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self) -> Iterator[ImageId]:
        return iter(map(ImageId.from_str, self.df.index))

    def items(self) -> Iterator[Tuple[ImageId, Image]]:
        for row in self.df.itertuples(index=True, name='ImageAsRow'):
            # noinspection PyProtectedMember
            x = row._asdict()
            i = x.pop("Index")
            yield ImageId.from_str(i), Image.from_obj(x)

    def __repr__(self):
        return f'{type(self).__name__}({self.identifier!r})'

    def to_parquet(self, urlpath: UrlpathLike) -> None:
        store = ImageProviderStore()
        store.to_urlpath(self.df, urlpath, identifier=self.identifier)

    @classmethod
    def from_parquet(cls, urlpath: UrlpathLike) -> ImageProvider:
        store = ImageProviderStore()
        df, identifier, user_metadata = store.from_urlpath(urlpath)
        assert {
            store.METADATA_KEY_STORE_TYPE,
            store.METADATA_KEY_STORE_VERSION,
            store.METADATA_KEY_PADO_VERSION,
            store.METADATA_KEY_PROVIDER_VERSION,
            store.METADATA_KEY_CREATED_AT,
            store.METADATA_KEY_CREATED_BY,
        } == set(user_metadata), f"currently unused {user_metadata!r}"
        inst = cls.__new__(cls)
        inst.df = df
        inst.identifier = identifier
        return inst


class GroupedImageProvider(ImageProvider):

    def __init__(self, *providers: BaseImageProvider):
        super().__init__()
        self.providers = []
        for p in providers:
            if not isinstance(p, ImageProvider):
                p = ImageProvider(p)
            if isinstance(p, GroupedImageProvider):
                self.providers.extend(p.providers)
            else:
                self.providers.append(p)

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

    def to_parquet(self, urlpath: UrlpathLike) -> None:
        super().to_parquet(urlpath)

    @classmethod
    def from_parquet(cls, urlpath: UrlpathLike) -> ImageProvider:
        raise NotImplementedError(f"unsupported operation for {cls.__name__!r}()")


class FilteredImageProvider(ImageProvider):

    def __init__(self, provider: BaseImageProvider, *, valid_keys: Optional[Iterable[ImageId]] = None):
        super().__init__()
        self._provider = ImageProvider(provider)
        self._vk = set(self._provider) if valid_keys is None else set(valid_keys)

    @cached_property
    def df(self):
        return self._provider.df.filter(items=self._vk, axis='index')

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
        return f'{type(self).__name__}({self._provider!r})'

    def to_parquet(self, urlpath: UrlpathLike) -> None:
        super().to_parquet(urlpath)

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
        for row in self.df.itertuples(index=True, name='ImageAsRow'):
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

    def to_parquet(self, urlpath: UrlpathLike) -> None:
        raise NotImplementedError(f"")


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
        if fs._check_file(path):
            return 100.0
        else:
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

def create_image_provider(
    search_urlpath: UrlpathLike,
    search_glob: str,
    *,
    output_urlpath: Optional[UrlpathLike],
    identifier: Optional[str] = None,
    checksum: bool | Mapping[ImageId, str] = True,
    resume: bool = False,
    ignore_broken: bool = True,
    image_id_func: GetImageIdFunc = image_id_from_parts,
    progress: bool = False,
) -> ImageProvider:
    """create an image provider from a directory containing images"""
    files_and_parts = find_files(search_urlpath, glob=search_glob)

    if resume:
        try:
            ip = ImageProvider.from_parquet(urlpath=output_urlpath)
        except FileNotFoundError:
            ip = ImageProvider(identifier=identifier)
    else:
        ip = ImageProvider(identifier=identifier)

    if progress:
        files_and_parts = tqdm(files_and_parts)

    try:
        for fp in files_and_parts:
            image_id = image_id_func(fp.file, fp.parts, ip.identifier)
            if resume and image_id in ip:
                continue
            if isinstance(checksum, Mapping):
                chk = checksum[image_id]
            else:
                chk = checksum
            try:
                image = Image(fp.file, load_metadata=True, load_file_info=True, checksum=chk)
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                if not ignore_broken:
                    raise e
            else:
                ip[image_id] = image

    finally:
        if output_urlpath is not None:
            ip.to_parquet(output_urlpath)

    return ip


def update_image_provider_urlpaths(
    search_urlpath: UrlpathLike,
    search_glob: str,
    *,
    provider: ImageProvider | UrlpathLike,
    inplace: bool = False,
    ignore_ambiguous: bool = False,
    progress: bool = False,
) -> ImageProvider:
    """search a path and re-associate image urlpaths by filename"""
    files_and_parts = find_files(search_urlpath, glob=search_glob)
    if isinstance(provider, ImageProvider):
        ip = provider
    else:
        ip = ImageProvider.from_parquet(urlpath=provider)

    new_urlpaths = match_partial_paths_reversed(
        current_urlpaths=ip.df.urlpath,
        new_urlpaths=list(x.file for x in files_and_parts),
        ignore_ambiguous=ignore_ambiguous,
        progress=progress,
    )

    old = ip.df.urlpath.copy()
    ip.df.loc[:, 'urlpath'] = [urlpathlike_to_string(p) for p in new_urlpaths]

    if progress:
        print(f"re-associated {np.sum(old.values != ip.df.urlpath.values)} files")

    if inplace and not isinstance(provider, ImageProvider):
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
    **update_kwargs
) -> None:
    """copy image data to a new location"""
    if not isinstance(provider, ImageProvider):
        raise TypeError(f"expected ImageProvider instance, got {type(provider).__name__}")

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
            dest, "**/*.svs",
            provider=provider,
            progress=progress,
            **update_kwargs
        )
