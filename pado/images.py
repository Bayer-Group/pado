import contextlib
import hashlib
import platform
import re
import warnings
from abc import ABC, abstractmethod
from ast import literal_eval
from itertools import tee
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Iterable, List, NamedTuple, Optional, Tuple, Union
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

import pandas as pd
from tqdm import tqdm

ImageId = Tuple[str, ...]


class _SerializedImageResource(NamedTuple):
    type: str
    image_id: str
    uri: str
    md5: Optional[str]


class ImageResource(ABC):
    __slots__ = "_image_id", "_str_image_id", "_resource", "_checksum"
    registry = {}
    resource_type = None

    def __init_subclass__(cls, **kwargs):
        cls.resource_type = kwargs.pop("resource_type")
        super().__init_subclass__(**kwargs)
        cls.registry[cls.resource_type] = cls

    def __init__(self, image_id, resource, checksum=None):
        if isinstance(image_id, str):
            str_image_id = image_id
            image_id = literal_eval(image_id)
        elif isinstance(image_id, (tuple, list, pd.Series)):
            image_id = tuple(image_id)
            str_image_id = repr(image_id)
        else:
            raise TypeError(f"image_id not str or tuple, got {type(image_id)}")
        self._image_id = image_id
        self._str_image_id = str_image_id
        self._resource = resource
        self._checksum = checksum

    @property
    def id(self) -> ImageId:
        return self._image_id

    @property
    def id_str(self) -> str:
        return self._str_image_id

    @property
    def checksum(self) -> str:
        """return the checksum of the resource"""
        return self._checksum

    @property
    @abstractmethod
    def uri(self) -> str:
        """return an uri for the resource"""
        ...

    @property
    @abstractmethod
    def local_path(self) -> Optional[Path]:
        """if possible return a local path"""
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
        return _SerializedImageResource(self.resource_type, self.id_str, self.uri, self.checksum)

    @classmethod
    def deserialize(cls, data: Union[_SerializedImageResource, pd.Series]):
        return cls.registry[data.type](data.image_id, data.uri, data.checksum)

    def __repr__(self):
        return f"{self.__class__.__name__}(image_id={self.id})"


class LocalImageResource(ImageResource, resource_type="local"):
    __slots__ = "_path",
    supported_schemes = {"file"}

    def __init__(self, image_id, resource, checksum=None):
        super().__init__(image_id, resource, checksum)
        if isinstance(resource, Path):
            p = resource

        elif isinstance(resource, str):
            # URIs need to be parsed
            _parsed = urlparse(resource)
            if _parsed.scheme not in self.supported_schemes:
                raise ValueError(f"'{_parsed.scheme}' scheme unsupported")
            path_str = unquote(_parsed.path)
            # check if we encode a windows path
            if re.match(r"/[A-Z]:/[^/]", path_str):
                p = PureWindowsPath(path_str[1:])
            elif re.match(r"//(?P<share>[^/]+)/(?P<directory>[^/]+)/", path_str):
                p = PureWindowsPath(path_str)
            else:
                p = PurePosixPath(path_str)

        else:
            raise TypeError(f"resource not str or pathlib.Path, got {type(resource)}")

        self._path = Path(p)
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

    @property
    def local_path(self) -> Optional[Path]:
        return self._path


class RemoteImageResource(ImageResource, resource_type="remote"):
    __slots__ = "_url", "_fp"
    supported_schemes = {"http", "https", "ftp"}

    def __init__(self, image_id, resource, checksum=None):
        super().__init__(image_id, resource, checksum)
        if not isinstance(resource, str):
            raise TypeError(f"url not str, got {type(resource)}")
        if urlparse(resource).scheme not in self.supported_schemes:
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

    @property
    def local_path(self) -> Optional[Path]:
        return None


class InternalImageResource(ImageResource, resource_type="internal"):
    __slots__ = ("_path", "_base_path", "_identifier")
    supported_schemes = {"pado+internal"}

    def __init__(self, image_id, resource, checksum=None):
        super().__init__(image_id, resource, checksum)
        if isinstance(resource, Path):
            # Paths can directly pass through
            ident, p = None, Path(resource)

        elif isinstance(resource, str):
            # URIs need to be parsed
            _parsed = urlparse(resource)
            if _parsed.scheme not in InternalImageResource.supported_schemes:
                raise ValueError(f"'{_parsed.scheme}' scheme unsupported")
            ident = _parsed.netloc
            p = Path(unquote(_parsed.path)).relative_to("/")

        else:
            raise TypeError(f"resource not str or pathlib.Path, got {type(resource)}")

        self._path = p
        if self._path.is_absolute():
            raise ValueError(
                f"{self.__class__.__name__} requires relative path, got '{resource}'"
            )
        self._base_path = None
        self._identifier = ident

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

    @property
    def local_path(self) -> Optional[Path]:
        try:
            path = self._base_path / self._identifier / self._path
        except TypeError:
            raise RuntimeError(
                "InternalImageResource has to be attached to dataset for usage"
            )
        return path

    def attach(self, identifier: str, base_path: Path):
        self._identifier = identifier
        self._base_path = base_path
        return self


class ImageResourcesProvider(ABC):
    def ids(self) -> Iterable[ImageId]:
        return (resource.id for resource in self)

    @abstractmethod
    def __getitem__(self, item: int) -> ImageResource:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


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
            raise IndexError(f"item index {item} not in range({len(self)})")
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
        for row in self._df.itertuples(index=False):
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
        inst = cls(identifier, base_path)
        inst._df = pd.DataFrame(
            [resource.serialize() for resource in provider],
            columns=_SerializedImageResource._fields,
        )
        inst.save()
        return inst


_BLOCK_SIZE = {
    LocalImageResource.__name__: 1024 * 1024 if platform.system() == "Windows" else 1024 * 64,
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
    if resource.checksum:
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

    if md5hash and md5hash.hexdigest() != resource.checksum:
        raise ValueError(f"{resource.id}: md5sum does not match provided md5")


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
                            image.id, internal_path, image.checksum
                        ).attach(self.identifier, self.base_path)
        finally:
            images.save()
