import contextlib
import hashlib
import platform
import tempfile
import warnings
from abc import ABC, abstractmethod
from itertools import tee
from pathlib import Path
from typing import Callable, Iterable, List, NamedTuple, Optional, Tuple, Union
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd
from tqdm import tqdm

ImageId = Tuple[str, ...]
SEPARATOR = "__"


class _SerializedImageResource(NamedTuple):
    type: str
    image_id: str
    uri: str
    md5: Optional[str]


class ImageResource(ABC):
    __slots__ = ("_image_id", "_resource", "_md5sum")
    registry = {}
    resource_type = None

    def __init_subclass__(cls, **kwargs):
        cls.resource_type = kwargs.pop("resource_type")
        super().__init_subclass__(**kwargs)
        cls.registry[cls.resource_type] = cls

    def __init__(self, image_id, resource, md5sum=None):
        if isinstance(image_id, str):
            image_id = tuple(image_id.split(SEPARATOR))
        if not isinstance(image_id, (tuple, list, pd.Series)):
            raise TypeError(f"image_id not str or tuple, got {type(image_id)}")
        self._image_id = tuple(image_id)
        self._resource = resource
        self._md5sum = md5sum

    @property
    def id(self) -> ImageId:
        return self._image_id

    @property
    def md5(self) -> str:
        """return the md5 of the resource"""
        return self._md5sum

    @property
    @abstractmethod
    def uri(self) -> str:
        """return an uri for the resource"""
        ...

    @abstractmethod
    def open(self):
        """return a file like object"""
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        """return the size of the file"""
        ...

    def serialize(self):
        """serialize the object"""
        return _SerializedImageResource(
            self.resource_type, SEPARATOR.join(self.id), self.uri, self.md5,
        )

    @classmethod
    def deserialize(cls, data: Union[_SerializedImageResource, pd.Series]):
        impl = cls.registry[data.type]
        return impl(data.image_id, data.uri, data.md5)


class ImageResourcesProvider(ABC):
    def ids(self) -> Iterable[ImageId]:
        return (resource.id for resource in self)

    @abstractmethod
    def __getitem__(self, item: int) -> ImageResource:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class LocalImageResource(ImageResource, resource_type="local"):
    __slots__ = ("_path",)

    def __init__(self, image_id, resource, md5sum=None):
        if not isinstance(resource, (Path, str)):
            raise TypeError(f"resource not str or pathlib.Path, got {type(resource)}")
        super().__init__(image_id, resource, md5sum)
        self._path = Path(resource)
        if not self._path.is_absolute():
            raise ValueError(
                f"LocalImageResource requires absolute path, got '{resource}'"
            )

    def open(self):
        return self._path.open("rb")

    @property
    def size(self):
        return self._path.stat().st_size

    @property
    def uri(self) -> str:
        return self._path.as_uri()


class RemoteImageResource(ImageResource, resource_type="remote"):
    __slots__ = ("_url", "_fp")

    def __init__(self, image_id, resource, md5sum=None):
        super().__init__(image_id, resource, md5sum)
        if not isinstance(resource, str):
            raise TypeError(f"url not str, got {type(resource)}")
        if urlparse(resource).scheme not in {"http", "https", "ftp"}:
            warnings.warn(f"untested scheme for url: '{resource}'")
        self._url = resource
        self._fp = None

    @contextlib.contextmanager
    def open(self):
        try:
            self._fp = urlopen(self._url)
            yield self._fp
        finally:
            self._fp.close()
            self._fp = None

    @property
    def size(self):
        try:
            return int(self._fp.info()["Content-length"])
        except (AttributeError, KeyError):
            return -1

    @property
    def uri(self) -> str:
        return self._url


class InternalImageResource(ImageResource, resource_type="internal"):
    def __init__(self, image_id, resource, md5sum=None):
        if not isinstance(resource, (Path, str)):
            raise TypeError(f"resource not str or pathlib.Path, got {type(resource)}")
        super().__init__(image_id, resource, md5sum)
        self._path = Path(resource)
        if self._path.is_absolute():
            raise ValueError(
                f"LocalImageResource requires relative path, got '{resource}'"
            )
        self._base_path = None
        self._identifier = None

    def open(self):
        try:
            path = self._base_path / self._identifier / self._path
        except TypeError:
            raise RuntimeError(
                "InternalImageResource has to be attached to dataset for usage"
            )
        return path.open("rb")

    @property
    def size(self) -> int:
        try:
            path = self._base_path / self._identifier / self._path
        except TypeError:
            raise RuntimeError(
                "InternalImageResource has to be attached to dataset for usage"
            )
        return path.stat().st_size

    @property
    def uri(self) -> str:
        if self._identifier is None:
            raise RuntimeError(
                "InternalImageResource has to be attached to dataset for usage"
            )
        return f"pado+internal://{self._identifier}/{self._path}"

    def attach(self, identifier: str, base_path: Path):
        self._identifier = identifier
        self._base_path = base_path
        return self


_WINDOWS = platform.system() == "Windows"
_BLOCK_SIZE = {
    LocalImageResource.__name__: 1024 * 1024 if _WINDOWS else 1024 * 64,
    RemoteImageResource.__name__: 1024 * 8,
}


def copy_resource(
    resource: ImageResource,
    path: Path,
    progress_hook: Optional[Callable[[int, int], None]],
):
    """copy an image resource to a local path"""
    md5hash = None
    # in case we provide an md5 build the hash incrementally
    if resource.md5:
        md5hash = hashlib.md5()

    with resource.open() as src, path.open("wb") as dst:
        src_size = resource.size
        bs = _BLOCK_SIZE[resource.__class__.__name__]
        src_read = src.read
        dst_write = dst.write
        read = 0

        if progress_hook:
            progress_hook(read, src_size)

        while True:
            buf = src_read(bs)
            if not buf:
                break
            read += len(buf)
            dst_write(buf)
            if md5hash:
                md5hash.update(buf)
            if progress_hook:
                progress_hook(read, src_size)

    if src_size >= 0 and read < src_size:
        raise RuntimeError(f"{resource.id}: could only copy {read} of {src_size} bytes")

    if md5hash and md5hash.hexdigest() != resource.md5:
        raise ValueError(f"{resource.id}: md5sum does not match provided md5")


class MergedImageResourcesProvider(ImageResourcesProvider):
    def __init__(self, image_providers: List[ImageResourcesProvider]):
        self._providers = image_providers
        self._cumulative_lengths = [0]
        for provider in self._providers:
            p_len = len(provider)
            self._cumulative_lengths.append(self._cumulative_lengths[-1] + p_len)
        self._total_len = self._cumulative_lengths[-1]

    def __getitem__(self, item: int) -> ImageResource:
        if not isinstance(item, int):
            raise TypeError("expects integer type")
        if item < 0 or item >= self._total_len:
            raise ValueError(f"item index {item} not in range({len(self)})")
        it0, it1 = tee(self._cumulative_lengths)
        next(it1, None)

        for low, high, provider in zip(it0, it1, self._providers):
            if item < high:
                return provider[item - low]
        else:
            raise IndexError(f"item index {item} not in range({len(self)})")

    def __len__(self) -> int:
        return self._total_len


class SerializableImageResourcesProvider(ImageResourcesProvider):
    STORAGE_FILE = "image_provider.parquet.gzip"

    def __init__(self, identifier, base_path):
        self._identifier = identifier
        self._base_path = base_path
        self._df_filename = self._base_path / self._identifier / self.STORAGE_FILE
        if self._df_filename.is_file():
            self._df = pd.read_parquet(self._df_filename)
        else:
            self._df = pd.DataFrame(columns=_SerializedImageResource._fields)

    @property
    def data(self):
        return self._df

    def __getitem__(self, item: int) -> ImageResource:
        row = self._df.iloc[item]
        resource = ImageResource.deserialize(row)
        if isinstance(resource, InternalImageResource):
            resource.attach(self._identifier, self._base_path)

        return resource

    def __setitem__(self, item: int, resource: ImageResource) -> None:
        self._df.iloc[item] = resource.serialize()

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self):
        for _, row in self._df.itertuples():
            resource = ImageResource.deserialize(row)
            if isinstance(resource, InternalImageResource):
                resource.attach(self._identifier, self._base_path)
            yield resource

    def ids(self):
        return (resource.id for resource in iter(self))

    def save(self):
        self._df_filename.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_parquet(self._df_filename, compression="gzip")

    @classmethod
    def from_provider(cls, identifier, base_path, provider):
        df = [resource.serialize() for resource in provider]
        inst = cls(identifier, base_path)
        inst.data.append(df)
        inst.save()
        return inst


class _ProgressCB(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, size, total):
        if total >= 0:
            self.total = total
        self.update(size - self.n)


class ImageResourceCopier:
    def __init__(self, identifier: str, base_path: Path):
        self.identifier = identifier
        self.base_path = Path(base_path)

    def __call__(self, images: SerializableImageResourcesProvider):
        try:
            for idx, image in tqdm(enumerate(images)):
                if isinstance(image, InternalImageResource):
                    continue  # image already available

                # copy image to dataset
                # vvv tqdm responsiveness
                miniters = 1 if isinstance(image, RemoteImageResource) else None
                # create folder structure
                internal_path = Path(*image.id)
                new_path = self.base_path / self.identifier / internal_path
                new_path.parent.mkdir(parents=True, exist_ok=True)

                with _ProgressCB(miniters=miniters) as t:
                    try:
                        copy_resource(image, new_path, t.update_to)
                    except Exception:
                        # todo: remove file?
                        raise
                    else:
                        images[idx] = InternalImageResource(
                            image.id, internal_path, image.md5
                        ).serialize()
        finally:
            images.save()