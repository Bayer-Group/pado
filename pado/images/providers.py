from __future__ import annotations

import uuid
from abc import ABC
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import MutableMapping
from typing import Optional
from typing import Set
from typing import Tuple

import pandas as pd

from pado._compat import cached_property
from pado.images.ids import ImageId
from pado.images.image import Image
from pado.types import UrlpathLike
from pado.io.files import find_files
from pado.io.store import Store
from pado.io.store import StoreType


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

    def __init__(self, provider: Optional[BaseImageProvider] = None, identifier: Optional[str] = None):
        if provider is None:
            provider = {}

        if isinstance(provider, ImageProvider):
            self.df = provider.df.copy()
            self.identifier = str(identifier) if identifier else provider.identifier
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


# === manipulation ============================================================

def create_image_provider(
    search_urlpath: UrlpathLike,
    search_glob: str,
    *,
    output_urlpath: Optional[UrlpathLike],
    identifier: Optional[str] = None,
    checksum: bool = True,
    resume: bool = False,
) -> ImageProvider:
    """create an image provider from a directory containing images"""
    files_and_parts = find_files(search_urlpath, glob=search_glob)

    if resume:
        ip = ImageProvider.from_parquet(urlpath=output_urlpath)
    else:
        ip = ImageProvider(identifier=identifier)

    try:
        for fp in files_and_parts:
            image_id = ImageId(*fp.parts, site=ip.identifier)
            if resume and image_id in ip:
                continue
            image = Image(fp.file, load_metadata=True, load_file_info=True, checksum=checksum)
            ip[image_id] = image

    finally:
        if output_urlpath is not None:
            ip.to_parquet(output_urlpath)

    return ip


def reassociate_images(provider: BaseImageProvider, search_path, search_pattern="**/*.svs"):
    """search a path and re-associate resources by filename"""
    '''
    def _fn(x):
        pth = ImageResource.deserialize(x).local_path
        if pth is None:
            return None
        return pth.name

    _local_path_name = self._df.apply(_fn, axis=1)

    idx = 0
    total = len(_local_path_name)
    for p in glob.iglob(f"{search_path}/{search_pattern}", recursive=True):
        p = Path(p)
        select = _local_path_name == p.name
        num_select = select.sum()
        if num_select.sum() != 1:
            if num_select > 1:
                warnings.warn(f"can't reassociate {p.name} due to multiple matches")
            continue
        idx += 1
        print(self._identifier, idx, total, "reassociating", p.name)
        row = self._df.loc[select].iloc[0]
        resource = ImageResource.deserialize(row)
        p = p.expanduser().absolute().resolve()
        new_resource = LocalImageResource(resource.id, p, resource.checksum)
        self[new_resource.id] = new_resource
    '''
    raise NotImplementedError("todo")
