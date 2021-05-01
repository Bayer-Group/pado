from __future__ import annotations

import os
import os.path as op
from abc import ABC
from functools import cached_property
from operator import itemgetter
from pathlib import Path
from pathlib import PurePath
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import Set
from typing import TYPE_CHECKING
from typing import Tuple
from typing import TypeVar
from typing import Union
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import fsspec
import pandas as pd
from orjson import JSONDecodeError
from orjson import OPT_SORT_KEYS
from orjson import dumps as orjson_dumps
from orjson import loads as orjson_loads

from pado.fileutils import hash_str
from pado.img import Image


class FilenamePartsMapper:
    """a plain mapper for ImageId instances based on filename only"""
    # Used as a fallback when site is None and we only know the filename
    id_field_names = ('filename',)
    def fs_parts(self, parts: Tuple[str, ...]): return parts


def register_filename_mapper(site, mapper):
    """used internally to register new mappers for ImageIds"""
    assert isinstance(mapper.id_field_names, tuple) and len(mapper.id_field_names) >= 1
    assert callable(mapper.fs_parts)
    if site in ImageId.site_mapper:
        raise RuntimeError(f"mapper: {site} -> {repr(mapper)} already registered")
    ImageId.site_mapper[site] = mapper()


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

    def __ne__(self, other):
        """need to overwrite tuple.__ne__"""
        return not self.__eq__(other)

    def to_url_hash(self, *, full: bool = False) -> str:
        """return a one way hash of the image_id"""
        if not full:
            return hash_str(self.last)
        else:
            return hash_str(self.to_str())

    # --- path methods ------------------------------------------------

    _SM = TypeVar("_SM", bound="FilenamePartsMapper")

    site_mapper: Mapping[Optional[str]:_SM] = {
        None: FilenamePartsMapper(),
    }

    # noinspection PyPropertyAccess
    @property
    def id_field_names(self):
        try:
            id_field_names = self.site_mapper[self.site].id_field_names
        except KeyError:
            raise KeyError(f"site '{self.site}' has no registered ImageProvider instance")
        return tuple(["site", *id_field_names])

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


def _ensure_image_id(image_id: Any) -> ImageId:
    if not isinstance(image_id, ImageId):
        raise TypeError(f"keys must be ImageId instances, got {type(image_id).__name__!r}")
    return image_id


def _ensure_image(image: Any) -> Image:
    if not isinstance(image, Image):
        raise TypeError(f"values must be Image instances, got {type(image).__name__!r}")
    return image


class BaseImageProvider(MutableMapping[ImageId, Image], ABC):
    pass

BaseImageProvider.register(dict)


def _identifier_from_path(parquet_path: str) -> str:
    """Basically pathlib.Path().stem for the fsspec path"""
    _, _, [path] = fsspec.get_fs_token_paths(parquet_path)
    return op.basename(path).partition(".")[0]


class ImageProvider(BaseImageProvider):

    def __init__(self, provider: BaseImageProvider):
        self.identifier = None
        if isinstance(provider, ImageProvider):
            self.df = provider.df.copy()
            self.identifier = provider.identifier
        elif provider is not None:
            if not provider:
                self.df = pd.DataFrame(columns=Image._fields)
            else:
                self.df = pd.DataFrame.from_records(
                    index=list(map(ImageId.to_str, provider.keys())),
                    data=list(map(lambda x: x.to_dict(), provider.values()))
                )

    def __getitem__(self, image_id: ImageId) -> Image:
        image_id = _ensure_image_id(image_id)
        row = self.df.loc[image_id.to_str()]
        return Image.from_dict(row)

    def __setitem__(self, image_id: ImageId, image: Image) -> None:
        image_id = _ensure_image_id(image_id)
        image = _ensure_image(image)
        dct = image.to_dict()
        self.df.loc[image_id.to_str()] = pd.Series(dct)

    def __delitem__(self, image_id: ImageId) -> None:
        image_id = _ensure_image_id(image_id)
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
            yield ImageId.from_str(i), Image.from_dict(x)

    def __repr__(self):
        return f'{type(self).__name__}({self.identifier!r})'

    def to_parquet(self, fspath: Union[Path, str]) -> None:
        self.df.to_parquet(fspath, compression="gzip")

    @classmethod
    def from_parquet(cls, fspath: Union[Path, str], identifier: Optional[str] = None):
        inst = cls.__new__(cls)
        inst.identifier = _identifier_from_path(fspath) if identifier is None else str(identifier)
        inst.df = pd.read_parquet(fspath)  # this already supports fsspec
        return inst


class GroupedImageProvider(ImageProvider):

    def __init__(self, *providers: ImageProvider):
        # noinspection PyTypeChecker
        super().__init__(None)
        self.providers = []
        for p in providers:
            if not isinstance(p, ImageProvider):
                p = ImageProvider(p)
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

    def to_parquet(self, fspath: Optional[Union[Path, str]] = None) -> None:
        super().to_parquet(fspath)

    @classmethod
    def from_parquet(cls, fspath: Union[Path, str], identifier: Optional[str] = None):
        raise NotImplementedError(f"unsupported operation for {cls.__name__!r}()")


class FilteredImageProvider(ImageProvider):

    def __init__(self, provider: BaseImageProvider, *, valid_keys: Optional[Iterable[ImageId]] = None):
        # noinspection PyTypeChecker
        super().__init__(None)
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
        return iter()

    def items(self) -> Iterator[Tuple[ImageId, Image]]:
        return super().items()

    def __repr__(self):
        return f'{type(self).__name__}({self._provider!r})'

    def to_parquet(self, fspath: Union[Path, str]) -> None:
        super().to_parquet(fspath)

    @classmethod
    def from_parquet(cls, fspath: Union[Path, str], identifier: Optional[str] = None):
        raise NotImplementedError(f"unsupported operation for {cls.__name__!r}()")


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
