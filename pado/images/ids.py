from __future__ import annotations

import hashlib
import io
import os.path as op
import token
import warnings
from ast import literal_eval
from operator import itemgetter
from pathlib import PurePath
from tokenize import generate_tokens
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple

from itsdangerous import base64_encode
from orjson import OPT_SORT_KEYS
from orjson import JSONDecodeError
from orjson import dumps as orjson_dumps
from orjson import loads as orjson_loads

from pado.types import FilterMissing
from pado.types import OpenFileLike

if TYPE_CHECKING:
    from pathlib import Path

_PADO_BLOCK_IMAGE_ID_EVAL = None


def _pado_image_id_from_str(cls: type[ImageId], image_id_str: str):
    """parse an image id string"""
    global _PADO_BLOCK_IMAGE_ID_EVAL

    if _PADO_BLOCK_IMAGE_ID_EVAL is None:
        from pado.settings import settings

        _PADO_BLOCK_IMAGE_ID_EVAL = settings.block_image_id_eval

    if not _PADO_BLOCK_IMAGE_ID_EVAL:
        # let's verify the input a tiny little bit
        if not (
            image_id_str.startswith(cls._prefix) and image_id_str.endswith(cls._suffix)
        ):
            raise ValueError(
                f"provided image_id str is not an ImageId(), got: '{image_id_str}'"
            )

        try:
            # fastest way to create the ImageId's from str.
            image_id = eval(image_id_str, {}, {cls.__name__: cls})  # nosec B307
        except (ValueError, SyntaxError):
            raise ValueError(f"provided image_id is not parsable: '{image_id_str}'")
        except NameError:
            # note: We want to guarantee that it's the same class. This could
            #   happen if a subclass of ImageId tries to deserialize a ImageId str
            raise ValueError(f"not a {cls.__name__}(): {image_id_str!r}")
        return image_id

    else:
        tokens = generate_tokens(io.StringIO(image_id_str).read)
        t_cls = next(tokens)
        if t_cls.type != token.NAME or t_cls.string != "ImageId":
            raise ValueError(f"provided image_id is not parsable: '{image_id_str}'")
        t_tuple_start = next(tokens)
        if t_tuple_start.type != token.OP or t_tuple_start.string != "(":
            raise ValueError(f"provided image_id is not parsable: '{image_id_str}'")
        parts = []
        site = None
        while True:
            t_item = next(tokens)
            t_sep = next(tokens)
            if t_item.type == token.STRING:
                parts.append(literal_eval(t_item.string))
                if t_sep.type == token.OP:
                    if t_sep.string == ")":
                        break
                    elif t_sep.string == ",":
                        continue
            elif t_item.type == token.NAME and t_item.string == "site":
                if t_sep.type == token.OP and t_sep.string == "=":
                    t_site = next(tokens)
                    t_end = next(tokens)
                    if (
                        t_site.type == token.STRING
                        and t_end.type == token.OP
                        and t_end.string == ")"
                    ):
                        site = literal_eval(t_site.string)
                        break
            raise ValueError(f"provided image_id is not parsable: '{image_id_str}'")
        return ImageId.make(parts, site=site)


# noinspection PyMethodMayBeStatic
class FilenamePartsMapper:
    """a plain mapper for ImageId instances based on filename only"""

    # Used as a fallback when site is None and we only know the filename
    id_field_names = ("filename",)

    def fs_parts(self, parts: Tuple[str, ...]):
        return parts


def register_filename_mapper(site, mapper):
    """used internally to register new mappers for ImageIds"""
    if not (
        isinstance(mapper.id_field_names, tuple) and len(mapper.id_field_names) >= 1
    ):
        raise RuntimeError("called on non-internal mapper?")
    if not callable(mapper.fs_parts):
        raise TypeError("expected a callable")
    if site in ImageId.site_mapper:
        raise RuntimeError(f"mapper: {site} -> {repr(mapper)} already registered")
    ImageId.site_mapper[site] = mapper()


class ImageId(Tuple[Optional[str], ...]):
    """Unique identifier for images in pado datasets"""

    # string matching rather than regex for speedup `ImageId('1', '2')`
    _prefix, _suffix = f"{__qualname__}(", ")"  # type: ignore

    def __new__(cls, *parts: str, site: Optional[str] = None):
        """Create a new ImageId instance"""
        try:
            part, *more_parts = parts
        except ValueError:
            raise ValueError(f"can not create an empty {cls.__name__}()")

        if isinstance(part, ImageId):
            # noinspection PyPropertyAccess
            return super().__new__(cls, [part.site, *part.parts])

        if any(not isinstance(x, str) for x in parts):
            if not more_parts and isinstance(part, Iterable):
                raise ValueError(
                    f"all parts must be of type str. Did you mean `{cls.__name__}.make({part!r})`?"
                )
            else:
                item_types = [type(x).__name__ for x in parts]
                raise ValueError(
                    f"all parts must be of type str. Received: {item_types!r}"
                )

        if part.startswith(cls._prefix) and part.endswith(cls._suffix):
            raise ValueError(
                f"use {cls.__name__}.from_str() to convert a serialized object"
            )
        elif part[0] == "{" and part[-1] == "}" and '"image_id":' in part:
            raise ValueError(
                f"use {cls.__name__}.from_json() to convert a serialized json object"
            )

        return super().__new__(cls, [site, *parts])  # type: ignore

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
    parts: Tuple[str, ...] = property(
        itemgetter(slice(1, None)), doc="return the parts of the image id"
    )
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
            raise TypeError(
                f"image_id must be of type 'str', got: '{type(image_id_str)}'"
            )
        return _pado_image_id_from_str(cls, image_id_str)

    # --- json serialization methods ----------------------------------

    def to_json(self):
        """Serialize the ImageId instance to a json object"""
        d = {"image_id": tuple(self[1:])}
        if self[0] is not None:
            d["site"] = self[0]
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
                raise TypeError(
                    f"image_id must be of type 'str', got: '{type(image_id_json)}'"
                )
            else:
                raise ValueError(
                    f"provided image_id is not parsable: '{image_id_json}'"
                )

        image_id_list = data["image_id"]
        if isinstance(image_id_list, str):
            raise ValueError("Incorrectly formatted json: `image_id` not a List[str]")

        return cls(*image_id_list, site=data.get("site"))

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

    def to_url_id(self) -> str:
        """return a base64 encoded string representation of imageid"""
        return base64_encode(self.to_str().encode()).decode()

    # --- path methods ------------------------------------------------

    site_mapper: Dict[Optional[str], FilenamePartsMapper] = {
        None: FilenamePartsMapper(),
    }

    # noinspection PyPropertyAccess
    @property
    def id_field_names(self):
        try:
            id_field_names = self.site_mapper[self.site].id_field_names
        except KeyError:
            raise KeyError(
                f"site '{self.site}' has no registered ImageProvider instance"
            )
        return tuple(["site", *id_field_names])

    # noinspection PyPropertyAccess
    def __fspath__(self) -> str:
        """return the ImageId as a relative path"""
        try:
            fs_parts = self.site_mapper[self.site].fs_parts(self.parts)
        except KeyError:
            raise KeyError(
                f"site '{self.site}' has no registered ImageProvider instance"
            )
        return op.join(*fs_parts)

    # noinspection PyPropertyAccess
    def to_path(self, *, ignore_site: bool = False) -> PurePath:
        """return the ImageId as a relative path

        Parameters
        ----------
        ignore_site:
            ignore the site for mapping to a PurePath and just rely on parts
        """
        if ignore_site:
            return PurePath(*self.parts)
        try:
            fs_parts = self.site_mapper[self.site].fs_parts(self.parts)
        except KeyError:
            raise KeyError(
                f"site '{self.site}' has no registered ImageProvider instance"
            )
        return PurePath(*fs_parts)


def _hash_str(string: str, hasher=hashlib.sha256) -> str:
    """calculate the hash of a string"""
    return hasher(string.encode()).hexdigest()


# === image id helpers ===

GetImageIdFunc = Callable[
    [OpenFileLike, Tuple[str, ...], Optional[str]], Optional[ImageId]
]


def image_id_from_parts(
    file: OpenFileLike, parts: Tuple[str, ...], identifier: Optional[str]
) -> Optional[ImageId]:
    return ImageId(*parts, site=identifier)


def image_id_from_parts_without_extension(
    file: OpenFileLike, parts: Tuple[str, ...], identifier: Optional[str]
) -> Optional[ImageId]:
    parts = PurePath(*parts).with_suffix("").parts
    return ImageId(*parts, site=identifier)


def image_id_from_json_file(
    file: OpenFileLike, parts: Tuple[str, ...], identifier: Optional[str]
) -> Optional[ImageId]:
    import json

    from pado.io.files import uncompressed

    try:
        with uncompressed(file) as f:
            data = json.load(f)
        fn = data["scan_name"]
        sd = data["scan_date"]
    except (json.JSONDecodeError, KeyError):
        return None

    if sd.lower() == "fixme":
        return ImageId(fn)
    else:
        return ImageId(sd, fn)


def match_partial_image_ids_reversed(
    ids: Iterable[ImageId], image_id: ImageId | tuple[str]
) -> Optional[ImageId]:
    """match image_ids from back to front

    returns None if no match
    raises ValueError in case match is ambiguous

    """
    if isinstance(ids, set):
        match_set = ids
    else:
        match_set = set(ids)

    def match(x: ImageId, s: Set[ImageId], idx: int) -> Optional[ImageId]:
        try:
            xi = x[idx]  # raises index error when out of parts to match
        except IndexError:
            raise ValueError(f"ambiguous: {x!r} -> {s!r}")
        sj = {sx for sx in s if sx[idx] == xi or xi is None}
        if len(sj) == 1:
            return sj.pop()
        elif len(sj) == 0:
            return None
        else:
            return match(x, sj, idx - 1)

    return match(image_id, match_set, -1)


def filter_image_ids(
    ids: Iterable[ImageId],
    target: Iterable[ImageId | tuple[str]],
    *,
    missing: FilterMissing | str = FilterMissing.WARN,
) -> set[ImageId]:
    """filter the provided ids to return the subset matching target"""
    is_image_id = isinstance(target, ImageId)
    is_parts_tuple = (
        isinstance(target, (tuple, list)) and target and isinstance(target[0], str)
    )

    if is_image_id or is_parts_tuple:
        raise TypeError(
            "please provide a Iterable[ImageId | tuple[str]] not a plain ImageId | tuple[str]"
        )

    if isinstance(missing, str):
        missing = FilterMissing(missing)

    # todo: improve speed
    s = set(ids)
    o = set()
    for t in target:
        m = match_partial_image_ids_reversed(s, t)
        if m is None:
            if missing == FilterMissing.WARN:
                warnings.warn(f"no match for: {t}")
            elif missing == FilterMissing.ERROR:
                raise KeyError(t)
            continue
        o.add(m)
    return o


def ensure_image_id(maybe_image_id: Any) -> ImageId:
    """guarantees that ImageId types get cast to ImageId"""
    if isinstance(maybe_image_id, ImageId):
        return maybe_image_id
    elif isinstance(maybe_image_id, list) or maybe_image_id.__class__ is tuple:
        site, *parts = maybe_image_id
        return ImageId.make(parts, site=site)
    elif isinstance(maybe_image_id, str):
        try:
            return ImageId.from_str(image_id_str=maybe_image_id)
        except ValueError:
            pass
        try:
            return ImageId.from_json(image_id_json=maybe_image_id)
        except ValueError:
            pass
        raise ValueError(f"can't cast string {maybe_image_id!r} to ImageId")
    raise TypeError(f"{maybe_image_id!r} of type {type(maybe_image_id).__name__!r}")


def load_image_ids_from_csv(
    csv_file: Path,
    *,
    csv_columns: list[int] | list[str] | None = None,
    no_header: bool = False,
) -> tuple[list[tuple[str, ...]], list[str] | None]:
    """load tuples from csv file

    Parameters
    ----------
    csv_file:
        path to your csv file
    csv_columns:
        a list of column names or column indices (`None` or `[]` means all)
    no_header:
        if enabled assume csv file has no header (requires int indices in csv_columns)

    Returns
    -------
    image_ids:
        tuple of selected cells for each row
    fieldnames:
        None if no_header=True else a list of column names

    """
    from pandas import read_csv

    if csv_columns is not None:
        is_seq = isinstance(csv_columns, (list, tuple))
        if not is_seq or any(not isinstance(c, (int, str)) for c in csv_columns):
            stype = type(csv_columns).__name__
            if is_seq and csv_columns:
                etype = type(csv_columns[0]).__name__
                stype += f"[{etype}]" if stype != "tuple" else f"[{etype},...]"
            raise TypeError("csv_columns must be a list[int] or list[str], got:", stype)

    if no_header and csv_columns is not None:
        csv_columns = [int(c) for c in csv_columns]

    csv_columns = csv_columns or []
    if len(csv_columns) == 0:
        csv_columns = slice(None)

    kw = {"header": None} if no_header else {}
    df = read_csv(csv_file, **kw)

    if no_header:
        fieldnames = None
    else:
        fieldnames = list(df.columns)

    df = df.loc[:, csv_columns]
    rows = list(df.itertuples(index=False, name="PadoImageIdTuple"))
    return rows, fieldnames
