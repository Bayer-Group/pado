from __future__ import annotations

import contextlib
import glob
import hashlib
import os
import os.path as op
import platform
import re
import sys
import warnings
from abc import ABC, abstractmethod
from operator import itemgetter

from orjson import loads as orjson_loads
from orjson import dumps as orjson_dumps
from orjson import JSONDecodeError, OPT_SORT_KEYS
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple, Union, TYPE_CHECKING
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

import pandas as pd
from tqdm import tqdm

from pado.fileutils import hash_str


class FilenamePartsMapper:
    """a plain mapper for ImageId instances based on filename only"""
    # Used as a fallback when site is None and we only know the filename
    id_field_names = ('filename',)
    def fs_parts(self, parts: Tuple[str, ...]): return parts


class ImageId(tuple):
    """Unique identifier for images in pado datasets"""

    # string matching rather than regex for speedup `ImageId('1', '2')`
    _prefix, _suffix = f"{__qualname__}(", ")"

    def __new__(cls, *parts: str, site: Optional[str] = None):
        """Create a new ImageId instance"""
        try:
            part, *more_parts = parts
        except ValueError:
            raise ValueError(f"can not create an empty {cls.__name__}()")

        if isinstance(part, ImageId):
            return super().__new__(cls, [part.site, *part.parts])

        if any(not isinstance(x, str) for x in parts):
            if not more_parts and isinstance(part, Iterable):
                raise ValueError(f"all parts must be of type str. Did you mean `{cls.__name__}.make({part!r})`?")
            else:
                item_types = [type(x).__name__ for x in parts]
                raise ValueError(f"all parts must be of type str. Received: {item_types!r}")

        if part.startswith(cls._prefix) and part.endswith(cls._suffix):
            raise ValueError(f"use {cls.__name__}.from_str() to convert a serialized object")
        elif part[0] == "{" and part[-1] == "}" and '"image_id":' in part:
            raise ValueError(f"use {cls.__name__}.from_json() to convert a serialized json object")

        return super().__new__(cls, [site, *parts])

    if TYPE_CHECKING:
        # Pycharm doesn't understand cls in classmethods otherwise...
        __init__ = tuple.__init__

    @classmethod
    def make(cls, parts: Iterable[str], site: Optional[str] = None):
        """Create a new ImageId instance from an iterable"""
        if isinstance(parts, str):
            raise TypeError(
                f"{cls.__name__}.make() requires a Sequence[str]. Did you mean `{cls.__name__}({repr(parts)})`?"
            )
        return cls(*parts, site=site)

    def __repr__(self):
        """Return a nicely formatted representation string"""
        site, *parts = self
        args = [repr(p) for p in parts]
        if site is not None:
            args.append(f"site={site!r}")
        return f"{type(self).__name__}({', '.join(args)})"

    # --- pickling ----------------------------------------------------

    def __getnewargs_ex__(self):
        return self[1:], {"site": self[0]}

    # --- namedtuple style property access ----------------------------

    # note PyCharm doesn't recognize these: https://youtrack.jetbrains.com/issue/PY-47192
    site: Optional[str] = property(itemgetter(0), doc="return site of the image id")
    parts: Tuple[str, ...] = property(itemgetter(slice(1, None)), doc="return the parts of the image id")
    last: str = property(itemgetter(-1), doc="return the last part of the image id")

    # --- string serialization methods --------------------------------

    to_str = __str__ = __repr__
    to_str.__doc__ = """serialize the ImageId instance to str"""

    @classmethod
    def from_str(cls, image_id_str: str):
        """create a new ImageId instance from an image id string

        >>> ImageId.from_str("ImageId('abc', '123.svs')")
        ImageId('abc', '123.svs')

        >>> ImageId.from_str('ImageId("123.svs", site="somewhere")')
        ImageId('123.svs', site='somewhere')

        >>> img_id = ImageId('123.svs', site="somewhere")
        >>> img_id == ImageId.from_str(img_id.to_str())
        True

        """
        if not isinstance(image_id_str, str):
            raise TypeError(f"image_id must be of type 'str', got: '{type(image_id_str)}'")

        # let's verify the input a tiny little bit
        if not (image_id_str.startswith(cls._prefix) and image_id_str.endswith(cls._suffix)):
            raise ValueError(f"provided image_id str is not an ImageId(), got: '{image_id_str}'")

        try:
            # ... i know it's bad, but it's the easiest way right now to support
            #   kwargs in the calling interface
            # fixme: revisit in case we consider this a security problem
            image_id = eval(image_id_str, {cls.__name__: cls})
        except (ValueError, SyntaxError):
            raise ValueError(f"provided image_id is not parsable: '{image_id_str}'")
        except NameError:
            # note: We want to guarantee that it's the same class. This could
            #   happen if a subclass of ImageId tries to deserialize a ImageId str
            raise ValueError(f"not a {cls.__name__}(): {image_id_str!r}")

        return image_id

    # --- json serialization methods ----------------------------------

    def to_json(self):
        """Serialize the ImageId instance to a json object"""
        d = {'image_id': tuple(self[1:])}
        if self[0] is not None:
            d['site'] = self[0]
        return orjson_dumps(d, option=OPT_SORT_KEYS).decode()

    @classmethod
    def from_json(cls, image_id_json: str):
        """create a new ImageId instance from an image id json string

        >>> ImageId.from_json('{"image_id":["abc","123.svs"]}')
        ImageId('abc', '123.svs')

        >>> ImageId.from_json('{"image_id":["abc","123.svs"],"site":"somewhere"}')
        ImageId('123.svs', site='somewhere')

        >>> img_id = ImageId('123.svs', site="somewhere")
        >>> img_id == ImageId.from_json(img_id.to_json())
        True

        """
        try:
            data = orjson_loads(image_id_json)
        except (ValueError, TypeError, JSONDecodeError):
            if not isinstance(image_id_json, str):
                raise TypeError(f"image_id must be of type 'str', got: '{type(image_id_json)}'")
            else:
                raise ValueError(f"provided image_id is not parsable: '{image_id_json}'")

        image_id_list = data['image_id']
        if isinstance(image_id_list, str):
            raise ValueError("Incorrectly formatted json: `image_id` not a List[str]")

        return cls(*image_id_list, site=data.get('site'))

    # --- hashing and comparison methods ------------------------------

    def __hash__(self):
        """carefully handle hashing!

        hashing is just based on the filename as a fallback!

        BUT: __eq__ is actually based on the full id in case both
             ids specify a site (which will be the default, but is
             not really while we are still refactoring...)
        """
        return tuple.__hash__(self[-1:])  # (self.last,)

    def __eq__(self, other):
        """carefully handle equality!

        equality is based on filename only in case site is not
        specified. Otherwise it's tuple.__eq__

        """
        if not isinstance(other, ImageId):
            return False  # we don't coerce tuples

        if self[0] is None or other[0] is None:  # self.site
            return self[1:] == other[1:]
        else:
            return tuple.__eq__(self, other)

    def to_url_hash(self, *, full: bool = False) -> str:
        """return a one way hash of the image_id"""
        if not full:
            return hash_str(self.last)
        else:
            return hash_str(self.to_str())

    # --- path methods ------------------------------------------------

    site_mapper: Mapping[Optional[str]:ImageResourcesProvider] = {
        None: FilenamePartsMapper(),
    }

    # noinspection PyPropertyAccess
    @property
    def id_field_names(self):
        try:
            id_field_names = self.site_mapper[self.site].id_field_names
        except KeyError:
            raise KeyError(f"site '{self.site}' has no registered ImageProvider instance")
        return tuple([self.site, *id_field_names])

    # noinspection PyPropertyAccess
    def __fspath__(self) -> str:
        """return the ImageId as a relative path"""
        try:
            fs_parts = self.site_mapper[self.site].fs_parts(self.parts)
        except KeyError:
            raise KeyError(f"site '{self.site}' has no registered ImageProvider instance")
        return op.join(*fs_parts)

    # noinspection PyPropertyAccess
    def to_path(self) -> PurePath:
        """return the ImageId as a relative path"""
        try:
            fs_parts = self.site_mapper[self.site].fs_parts(self.parts)
        except KeyError:
            raise KeyError(f"site '{self.site}' has no registered ImageProvider instance")
        return PurePath(*fs_parts)


class _SerializedImageResource(NamedTuple):
    type: str
    image_id: str
    uri: str
    checksum: Optional[str]


class ImageResource(ABC):
    _registry = {}
    resource_type = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.resource_type] = cls

    def __init__(self, image_id, resource, checksum=None):
        # NOTE: only subclasses of ImageResource can be instantiated directly
        if isinstance(image_id, str):
            image_id = ImageId.from_str(image_id)
        if isinstance(image_id, ImageId):
            pass  # required to check this before
        elif isinstance(image_id, (tuple, list, pd.Series)):
            image_id = ImageId(*image_id)
        else:
            raise TypeError(f"can not convert to ImageId, got {type(image_id)}")
        self._image_id = image_id
        self._str_image_id = str(image_id)
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
        """serialize the object"""
        return _SerializedImageResource(
            self.resource_type, self.id_str, self.uri, self.checksum,
        )

    @classmethod
    def deserialize(cls, data: Union[_SerializedImageResource, pd.Series]):
        impl = cls._registry[data.type]
        return impl(data.image_id, data.uri, data.checksum)

    def __repr__(self):
        return f"{self.__class__.__name__}(image_id={self.id})"


class LocalImageResource(ImageResource):
    supported_schemes = {"file"}
    resource_type = "local"

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


class RemoteImageResource(ImageResource):
    supported_schemes = {"http", "https", "ftp"}
    resource_type = "remote"

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


class InternalImageResource(ImageResource):
    supported_schemes = {"pado+internal"}
    resource_type = "internal"

    def __init__(self, image_id, resource, checksum=None):
        super().__init__(image_id, resource, checksum)
        if isinstance(resource, PurePath):
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


ImageResourcesProvider = Mapping[ImageId, ImageResource]


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
        # support older format
        if 'md5' in df.columns:
            df = df.rename(columns={'md5': 'checksum'})
            df['image_id'] = df['image_id'].apply(lambda x: f'ImageId{tuple(x.split("__"))!r}')

        self._loc = df["image_id"].apply(ImageId.from_str)
        self._df = df

    def __getitem__(self, item: ImageId) -> ImageResource:
        if not isinstance(item, ImageId):
            raise TypeError(f"requires ImageId. got `{type(item)}`")
        try:
            row = self._df.loc[self._loc == item]
        except pd.core.indexing.IndexingError:
            self._loc = self._df["image_id"].apply(ImageId.from_str)
            return self.__getitem__(item)
        if len(row) == 0:
            raise KeyError(item)
        assert len(row) == 1, "there should only be one row per image in the image provider"
        resource = ImageResource.deserialize(row.squeeze())
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

    def values(self):
        yield from (ImageResource.deserialize(row) for _, row in self._df.iterrows())

    def items(self):
        yield from zip(iter(self), self.values())

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
            print(self._identifier, "reassociating", p.name)
            row = self._df.loc[select].iloc[0]
            resource = ImageResource.deserialize(row)
            p = p.expanduser().absolute().resolve()
            new_resource = LocalImageResource(resource.id, p, resource.checksum)
            self[new_resource.id] = new_resource


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
            for idx, (image_id, image) in tqdm(enumerate(images.items())):
                if isinstance(image, InternalImageResource):
                    continue  # image already available

                # copy image to dataset
                # vvv tqdm responsiveness
                miniters = 1 if isinstance(image, RemoteImageResource) else None
                # create folder structure
                internal_path = image.id.to_path()
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
                            image.id, internal_path, image.checksum
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
