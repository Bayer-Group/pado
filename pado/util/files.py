"""file utility functions"""
import json
import logging
import os
import sys
from typing import Any
from typing import Iterator
from typing import NamedTuple
from typing import Tuple

if sys.version_info[:2] >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

import fsspec
from fsspec.core import OpenFile

from pado.types import OpenFileLike
from pado.types import UrlpathLike

_logger = logging.getLogger(__name__)


class _OpenFileAndParts(NamedTuple):
    file: OpenFile
    parts: Tuple[str, ...]


def find_files(urlpath: UrlpathLike, *, glob: str = "**/*") -> Iterator[_OpenFileAndParts]:
    """iterate over the files with matching at all paths"""
    ofile = urlpathlike_to_fsspec(urlpath)
    fs = ofile.fs
    pth = ofile.path
    if not fs.isdir(pth):
        # todo: check if this makes sense for all fsspec...
        raise NotADirectoryError(pth)

    def split_into_parts(base, p):
        rpth = os.path.relpath(p, base)
        return tuple(os.path.normpath(rpth).split(os.sep))

    yield from (
        _OpenFileAndParts(
            file=OpenFile(fs=fs, path=opth),
            parts=split_into_parts(pth, opth),
        )
        for opth in fs.glob(os.path.join(pth, glob))
    )


def is_fsspec_open_file_like(obj: Any) -> TypeGuard[OpenFileLike]:
    """test if an object is like a fsspec.core.OpenFile instance"""
    # if isinstance(obj, fsspec.core.OpenFile) doesn't cut it...
    # ... fsspec filesystems just need to quack OpenFile.
    return (
        isinstance(obj, OpenFileLike)
        and isinstance(obj.fs, fsspec.AbstractFileSystem)
        and isinstance(obj.path, str)
    )


def urlpathlike_to_string(urlpath: UrlpathLike) -> str:
    """convert an urlpath-like object and stringify it"""
    if is_fsspec_open_file_like(urlpath):
        fs: fsspec.AbstractFileSystem = urlpath.fs
        path: str = urlpath.path
        return json.dumps({
            "fs": fs.to_json(),
            "path": path
        })

    if isinstance(urlpath, os.PathLike):
        urlpath = os.fspath(urlpath)

    if '~' in urlpath:
        urlpath = os.path.expanduser(urlpath)

    if isinstance(urlpath, bytes):
        return urlpath.decode()
    elif isinstance(urlpath, str):
        return urlpath
    else:
        raise TypeError(f"can't stringify: {urlpath!r} of type {type(urlpath)!r}")


def urlpathlike_to_fsspec(obj: UrlpathLike, *, mode='rb') -> OpenFileLike:
    """use an urlpath-like object and return an fsspec.core.OpenFile"""
    if is_fsspec_open_file_like(obj):
        return obj

    try:
        json_obj = json.loads(obj)  # type: ignore
    except (json.JSONDecodeError, TypeError):
        if isinstance(obj, os.PathLike):
            obj = os.fspath(obj)
        if not isinstance(obj, str):
            raise TypeError(f"got {obj!r} of type {type(obj)!r}")
        return fsspec.open(obj, mode=mode)
    else:
        if not isinstance(json_obj, dict):
            raise TypeError(f"got json {json_obj!r} of type {type(json_obj)!r}")
        fs = fsspec.AbstractFileSystem.from_json(json_obj["fs"])
        return fs.open(json_obj["path"], mode=mode)
