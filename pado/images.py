import contextlib
import glob
import hashlib
import platform
import re
import warnings
from abc import ABC, abstractmethod
from ast import literal_eval
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Callable, Mapping, NamedTuple, Optional, Union
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

import pandas as pd
from tqdm import tqdm


class ImageId(tuple):
    """ImageId for images in pado datasets"""

    _regex = re.compile(r"ImageId(\(.*\))")

    def __new__(cls, *args):
        if not args:
            raise ValueError("can not create an empty ImageId()")

        if not all(isinstance(x, str) for x in args):
            item_types = [type(x).__name__ for x in args]
            raise ValueError(f"all image_id elements must be of type str, received: {item_types!r}")

        if any(cls._regex.match(x) for x in args):
            if len(args) == 1:
                raise ValueError("use ImageId.from_str() to convert a serialized ImageId()")
            raise ValueError("a provided element matches a serialized image id")

        return super().__new__(cls, args)

    def __repr__(self):
        return f"{type(self).__name__}{super().__repr__()}"

    to_str = __str__ = __repr__

    @classmethod
    def from_str(cls, image_id: str):
        """create a new ImageId instance from an image id string

        >>> ImageId.from_str("ImageId('abc', '123.svs')")
        ImageId('abc', '123.svs')

        """
        if not isinstance(image_id, str):
            raise TypeError(f"image_id must be of type 'str', got: '{type(image_id)}'")

        m = cls._regex.match(image_id)
        if not m:
            raise ValueError(f"provided image_id str is not an ImageId(), got: '{image_id}'")

        try:
            image_id_elements = literal_eval(m.group(1))
        except (ValueError, SyntaxError):
            raise ValueError(f"provided image_id is unparsable: '{image_id}'")

        if not isinstance(image_id_elements, tuple):
            raise ValueError(f"not a ImageId(): '{image_id}'")

        return cls(*image_id_elements)

    def __fspath__(self) -> str:
        return str(self.to_path())

    def to_path(self):
        """return the ImageId as a relative path"""
        image_path = PurePath(*self)
        assert not image_path.is_absolute()
        return image_path

    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()


class _SerializedImageResource(NamedTuple):
    type: str
    image_id: str
    uri: str
    md5: Optional[str]


class ImageResource(ABC):
    _registry = {}
    resource_type = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.resource_type] = cls

    def __init__(self, image_id, resource, md5sum=None):
        # NOTE: only subclasses of ImageResource can be instantiated directly
        if isinstance(image_id, str):
            image_id = ImageId.from_str(image_id)
        elif isinstance(image_id, (tuple, list, pd.Series)):
            image_id = ImageId(*image_id)
        else:
            raise TypeError(f"can not convert to ImageId, got {type(image_id)}")
        self._image_id = image_id
        self._str_image_id = str(image_id)
        self._resource = resource
        self._md5sum = md5sum

    @property
    def id(self) -> ImageId:
        return self._image_id

    @property
    def id_str(self) -> str:
        return self._str_image_id

    @property
    def md5(self) -> str:
        """return the md5 of the resource"""
        return self._md5sum

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
        """serialize the object"""
        return _SerializedImageResource(
            self.resource_type, self.id_str, self.uri, self.md5,
        )

    @classmethod
    def deserialize(cls, data: Union[_SerializedImageResource, pd.Series]):
        impl = cls._registry[data.type]
        return impl(data.image_id, data.uri, data.md5)

    def __repr__(self):
        return f"{self.__class__.__name__}(image_id={self.id})"


class LocalImageResource(ImageResource):
    supported_schemes = {"file"}
    resource_type = "local"

    def __init__(self, image_id, resource, md5sum=None):
        super().__init__(image_id, resource, md5sum)
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


class RemoteImageResource(ImageResource):
    supported_schemes = {"http", "https", "ftp"}
    resource_type = "remote"

    def __init__(self, image_id, resource, md5sum=None):
        super().__init__(image_id, resource, md5sum)
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


class InternalImageResource(ImageResource):
    supported_schemes = {"pado+internal"}
    resource_type = "internal"

    def __init__(self, image_id, resource, md5sum=None):
        super().__init__(image_id, resource, md5sum)
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


ImageResourcesProvider = Mapping[str, ImageResource]


class SerializableImageResourcesProvider(ImageResourcesProvider):
    STORAGE_FILE = "image_provider.parquet.gzip"

    def __init__(self, identifier, base_path):
        self._identifier = identifier
        self._base_path = base_path
        self._df_filename = self._base_path / self._identifier / self.STORAGE_FILE
        if self._df_filename.is_file():
            df = pd.read_parquet(self._df_filename)
        else:
            df = pd.DataFrame(columns=_SerializedImageResource._fields)
        self._df = df.set_index(df["image_id"])

    def __getitem__(self, item: ImageId) -> ImageResource:
        if not isinstance(item, ImageId):
            raise TypeError(f"requires ImageId. got `{type(item)}`")
        row = self._df.loc[item.to_str()]
        resource = ImageResource.deserialize(row)
        if isinstance(resource, InternalImageResource):
            resource.attach(self._identifier, self._base_path)
        return resource

    def __setitem__(self, item: ImageId, resource: ImageResource) -> None:
        if not isinstance(item, ImageId):
            raise TypeError(f"requires ImageId. got `{type(item)}`")
        self._df.loc[item.to_str()] = resource.serialize()

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self):
        yield from (ImageId.from_str(x) for x in self._df["image_id"])

    def save(self):
        self._df_filename.parent.mkdir(parents=True, exist_ok=True)
        df = self._df.reset_index(drop=True)
        df.to_parquet(self._df_filename, compression="gzip")

    @classmethod
    def from_provider(cls, identifier, base_path, provider):
        inst = cls(identifier, base_path)
        df = pd.DataFrame(
            [resource.serialize() for resource in provider.values()],
            columns=_SerializedImageResource._fields,
        )
        df = df.set_index(df["image_id"])
        inst._df = df
        inst.save()
        return inst

    def __repr__(self):
        return f'{type(self).__name__}(identifier={self._identifier!r}, base_path={self._base_path!r})'

    def reassociate_resources(self, search_path, search_pattern="**/*.svs"):
        """search a path and re-associate resources by filename"""

        def _fn(x):
            pth = ImageResource.deserialize(x).local_path
            if pth is None:
                return None
            return pth.name

        _local_path_name = self._df.apply(_fn, axis=1)

        for p in glob.glob(f"{search_path}/{search_pattern}", recursive=True):
            p = Path(p)
            select = _local_path_name == p.name
            num_select = select.sum()
            if num_select.sum() != 1:
                if num_select > 1:
                    warnings.warn(f"can't reassociate {p.name} due to multiple matches")
                continue
            row = self._df.loc[select].iloc[0]
            resource = ImageResource.deserialize(row)
            p = p.expanduser().absolute().resolve()
            new_resource = LocalImageResource(resource.id, p, resource.md5)
            self[new_resource.id_str] = new_resource


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
            for idx, (image_id, image) in tqdm(enumerate(images.items())):
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
                        images[image_id] = InternalImageResource(
                            image.id, internal_path, image.md5
                        ).attach(self.identifier, self.base_path)
        finally:
            images.save()


def get_common_local_paths(image_provider: ImageResourcesProvider):
    """return common base paths in an image provider"""
    bases = set()
    for resource in image_provider.values():
        if isinstance(resource, RemoteImageResource):
            continue
        id_parts = resource.id
        parts = resource.local_path.parts
        assert len(parts) > len(id_parts), f"parts={parts!r}, id_parts={id_parts!r}"
        base, fn_parts = parts[:-len(id_parts)], parts[-len(id_parts):]
        assert id_parts == fn_parts, f"{id_parts!r} != {fn_parts!r}"
        bases.add(Path().joinpath(*base))  # resource.local_paths are guaranteed to be absolute
    return bases
