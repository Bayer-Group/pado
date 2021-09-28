"""file utility functions"""
import gzip
import json
import lzma
import os
import pickle
import sys
import tarfile
import typing
import zipfile
from ast import literal_eval
from contextlib import ExitStack
from contextlib import contextmanager
from pathlib import PurePath
from typing import Any
from typing import BinaryIO
from typing import ContextManager
from typing import Iterable
from typing import Iterator
from typing import NamedTuple
from typing import Tuple
from typing import Union

import fsspec
from fsspec import AbstractFileSystem
from fsspec import get_filesystem_class
from fsspec.core import OpenFile
from fsspec.core import strip_protocol

from pado.types import FsspecIOMode
from pado.types import OpenFileLike
from pado.types import UrlpathLike

if sys.version_info[:2] >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard


_PADO_FSSPEC_PICKLE_PROTOCOL = 4


class _OpenFileAndParts(NamedTuple):
    file: OpenFile
    parts: Tuple[str, ...]


def find_files(urlpath: UrlpathLike, *, glob: str = "**/*") -> Iterable[_OpenFileAndParts]:
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

    return [
        _OpenFileAndParts(
            file=OpenFile(fs=fs, path=opth),
            parts=split_into_parts(pth, opth),
        )
        for opth in fs.glob(os.path.join(pth, glob))
    ]


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
        try:
            serialized_fs = fs.to_json()
        except NotImplementedError:
            serialized_fs = repr(pickle.dumps(fs, protocol=_PADO_FSSPEC_PICKLE_PROTOCOL))
        return json.dumps({
            "fs": serialized_fs,
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


def urlpathlike_to_fsspec(obj: UrlpathLike, *, mode: FsspecIOMode = 'rb') -> OpenFileLike:
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
        try:
            fs = fsspec.AbstractFileSystem.from_json(json_obj["fs"])
        except json.JSONDecodeError:
            fs = pickle.loads(literal_eval(json_obj["fs"]))
        return fsopen(fs, json_obj["path"], mode=mode)


def urlpathlike_to_fs_and_path(obj: UrlpathLike) -> Tuple[AbstractFileSystem, str]:
    """use an urlpath-like object and return an fsspec.AbstractFileSystem and a path"""
    if is_fsspec_open_file_like(obj):
        return obj.fs, obj.path

    try:
        json_obj = json.loads(obj)  # type: ignore
    except (json.JSONDecodeError, TypeError):
        if isinstance(obj, os.PathLike):
            obj = os.fspath(obj)
        if not isinstance(obj, str):
            raise TypeError(f"got {obj!r} of type {type(obj)!r}")
        fs, _, (path,) = fsspec.get_fs_token_paths(obj)
        return fs, path
    else:
        if not isinstance(json_obj, dict):
            raise TypeError(f"got json {json_obj!r} of type {type(json_obj)!r}")
        try:
            fs = fsspec.AbstractFileSystem.from_json(json_obj["fs"])
        except json.JSONDecodeError:
            fs = pickle.loads(literal_eval(json_obj["fs"]))
        return fs, json_obj["path"]


def urlpathlike_to_path_parts(obj: UrlpathLike) -> Tuple[str, ...]:
    """take an urlpathlike object and return the path parts

    this does not instantiate the fsspec.AbstractFilesystem class.
    (does not open connections, etc on instantiation)
    """
    if is_fsspec_open_file_like(obj):
        path = obj.path
    else:
        try:
            json_obj = json.loads(obj)  # type: ignore
        except (json.JSONDecodeError, TypeError):
            if isinstance(obj, os.PathLike):
                obj = os.fspath(obj)
            if not isinstance(obj, str):
                raise TypeError(f"got {obj!r} of type {type(obj)!r}")
            path = strip_protocol(obj)
        else:
            if not isinstance(json_obj, dict):
                raise TypeError(f"got json {json_obj!r} of type {type(json_obj)!r}")
            path = json_obj['path']
    return PurePath(path).parts


def fsopen(
    fs: AbstractFileSystem,
    path: [str, os.PathLike],
    *,
    mode: FsspecIOMode = 'rb',
) -> OpenFileLike:
    """small helper to support mode 'x' for fsspec filesystems"""
    if mode not in typing.get_args(FsspecIOMode):
        raise ValueError("fsspec only supports a subset of IOModes")
    if 'x' in mode:
        if fs.exists(path):
            raise FileExistsError(f"{path!r} at {fs!r}")
        else:
            mode = mode.replace('x', 'w')
    return OpenFile(fs, path, mode=mode)


@contextmanager
def uncompressed(file: Union[BinaryIO, ContextManager[BinaryIO]]) -> Iterator[BinaryIO]:
    """contextmanager for reading nested compressed files

    supported formats: GZIP, LZMA, ZIP, TAR

    """
    GZIP = {b"\x1F\x8B"}
    LZMA = {b"\xFD\x37\x7A\x58\x5A\x00"}
    ZIP = {b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"}

    def is_tar(fileobj):
        _pos = fileobj.tell()
        try:
            t = tarfile.open(fileobj=fileobj)  # rewinds to 0 on failure
            t.close()
            return True
        except tarfile.TarError:
            return False
        finally:
            fileobj.seek(_pos)

    with ExitStack() as stack:
        if not (hasattr(file, 'read') and hasattr(file, 'seek')):
            file = stack.enter_context(file)

        pos = file.tell()
        magic = file.read(8)
        file.seek(pos)

        if magic[:2] in GZIP:
            fgz = stack.enter_context(gzip.open(file))
            yield stack.enter_context(uncompressed(fgz))

        elif magic[:6] in LZMA:
            fxz = stack.enter_context(lzma.open(file))
            yield stack.enter_context(uncompressed(fxz))

        elif magic[:4] in ZIP:
            fzip = stack.enter_context(zipfile.ZipFile(file))
            paths = fzip.namelist()
            if len(paths) != 1:
                raise RuntimeError(
                    "zip must contain exactly one file: won't auto uncompress"
                )
            fzipped = stack.enter_context(fzip.open(paths[0], mode="r"))
            yield stack.enter_context(uncompressed(fzipped))

        elif is_tar(file):
            ftar = stack.enter_context(tarfile.open(fileobj=file))
            members = ftar.getmembers()
            if len(members) != 1:
                raise RuntimeError(
                    f"tar must contain exactly one file: won't auto uncompress"
                )
            ftarred = stack.enter_context(ftar.extractfile(members[0]))
            yield stack.enter_context(uncompressed(ftarred))

        else:
            yield file
