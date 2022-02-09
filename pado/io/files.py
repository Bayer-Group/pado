"""file utility functions"""
from __future__ import annotations

import gzip
import inspect
import json
import lzma
import os
import pickle
import sys
import tarfile
import typing
import warnings
import zipfile
from ast import literal_eval
from contextlib import ExitStack
from contextlib import contextmanager
from pathlib import PurePath
from typing import Any
from typing import AnyStr
from typing import BinaryIO
from typing import Collection
from typing import ContextManager
from typing import Iterable
from typing import Iterator
from typing import NamedTuple
from typing import Tuple
from typing import Union

import fsspec
from fsspec import AbstractFileSystem
from fsspec.core import OpenFile
from fsspec.core import strip_protocol

from pado.types import FsspecIOMode
from pado.types import OpenFileLike
from pado.types import UrlpathLike

if sys.version_info[:2] >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

__all__ = [
    "find_files",
    "is_fsspec_open_file_like",
    "urlpathlike_to_string",
    "urlpathlike_to_fsspec",
    "urlpathlike_to_fs_and_path",
    "urlpathlike_to_path_parts",
    "urlpathlike_to_localpath",
    "urlpathlike_to_uri",
    "fsopen",
    "uncompressed",
]


_PADO_FSSPEC_PICKLE_PROTOCOL = 4


class _OpenFileAndParts(NamedTuple):
    file: OpenFile
    parts: Tuple[str, ...]


def find_files(
    urlpath: UrlpathLike,
    *,
    glob: str = "**/*",
    storage_options: dict[str, Any] | None = None,
) -> Iterable[_OpenFileAndParts]:
    """iterate over the files with matching at all paths"""
    if storage_options is None:
        storage_options = {}

    ofile = urlpathlike_to_fsspec(urlpath, **storage_options)
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


def _get_constructor_default_options(cls: type) -> dict[str, Any]:
    """return the default options of the class constructor"""
    arg_spec = inspect.getfullargspec(cls.__init__)
    default_options = {}
    for name, value in zip(
        reversed(arg_spec.args),
        reversed(arg_spec.defaults or ()),
    ):
        default_options[name] = value
    if arg_spec.kwonlydefaults:
        default_options.update(arg_spec.kwonlydefaults)
    return default_options


def _remove_duplicate_items(d0: dict[str, Any], d1: dict[str, Any]) -> dict[str, Any]:
    """performs d0 - d1 for all items only if d0[k] == d1[k]"""
    return {k: v for k, v in d0.items() if not (k in d1 and d1[k] == v)}


def _pathlike_to_string(pathlike: os.PathLike[AnyStr]) -> str:
    """stringify a pathlike object"""
    if isinstance(pathlike, os.PathLike):
        pathlike = os.fspath(pathlike)

    if "~" in pathlike:
        pathlike = os.path.expanduser(pathlike)

    if isinstance(pathlike, bytes):
        return pathlike.decode()
    elif isinstance(pathlike, str):
        return pathlike
    else:
        raise TypeError(f"can't stringify: {pathlike!r} of type {type(pathlike)!r}")


def urlpathlike_to_uri(
    urlpath: UrlpathLike,
    *,
    repr_fallback: bool = False,
    ignore_options: Collection[str] = (),
) -> str:
    """convert an urlpath-like object to an fsspec URI

    Parameters
    ----------
    urlpath :
        an urlpath-like object
    repr_fallback :
        allow falling back to a repr like representation if encoding to a uri
        is impossible due to `storage_args` or `storage_options`
    ignore_options :
        a set of options that should be omitted from the uri representation

    Returns
    -------
    uri :
        a URI string representation of the urlpath-like

    """
    if is_fsspec_open_file_like(urlpath):
        fs = urlpath.fs
        pth = urlpath.path

        cls = type(fs)
        custom_args = fs.storage_args
        default_options = _get_constructor_default_options(cls)
        custom_options = _remove_duplicate_items(fs.storage_options, default_options)

        if ignore_options:
            for key in ignore_options:
                if key in custom_options:
                    del custom_options[key]

        if not custom_args and not custom_options:
            proto = fs.protocol
            if isinstance(proto, (list, tuple)):
                proto = proto[0]
            return f"{proto}://{pth}"

        elif not custom_args:
            # todo: ... can prettify some file systems
            pass

        if not repr_fallback:
            raise ValueError("can't serialize urlpath to a pure uri")

        # provide a repr like string
        cls_name = f"{cls.__module__}.{cls.__name__}"
        fs_params = [
            *custom_args,
            *(f"{k}={v!r}" for k, v in custom_options.items()),
        ]
        return f"{cls_name}({', '.join(fs_params)}).open({pth!r})"

    else:
        return _pathlike_to_string(urlpath)


def urlpathlike_to_string(
    urlpath: UrlpathLike,
    *,
    ignore_options: Collection[str] = (),
) -> str:
    """convert an urlpath-like object and stringify it"""
    if is_fsspec_open_file_like(urlpath):
        fs: fsspec.AbstractFileSystem = urlpath.fs
        path: str = urlpath.path
        try:
            serialized_fs = fs.to_json()
        except NotImplementedError:
            if ignore_options:
                warnings.warn(
                    "ignore_options are not handled for FS that don't support to_json",
                    stacklevel=2,
                )
            serialized_fs = repr(
                pickle.dumps(fs, protocol=_PADO_FSSPEC_PICKLE_PROTOCOL)
            )
        else:
            if ignore_options:
                d = json.loads(serialized_fs)
                for opt in ignore_options:
                    d.pop(opt, None)
                serialized_fs = json.dumps(d)

        return json.dumps({"fs": serialized_fs, "path": path})

    else:
        if ignore_options:
            warnings.warn(
                "ignore_options are not handled for stringified UrlpathLike",
                stacklevel=2,
            )
        return _pathlike_to_string(urlpath)


def urlpathlike_to_fsspec(
    obj: UrlpathLike,
    *,
    mode: FsspecIOMode = "rb",
    storage_options: dict[str, Any] | None = None,
) -> OpenFileLike:
    """use an urlpath-like object and return an fsspec.core.OpenFile"""
    if is_fsspec_open_file_like(obj):
        return obj

    if storage_options is None:
        storage_options = {}

    try:
        json_obj = json.loads(obj)  # type: ignore
    except (json.JSONDecodeError, TypeError):
        if isinstance(obj, os.PathLike):
            obj = os.fspath(obj)
        if not isinstance(obj, str):
            raise TypeError(f"got {obj!r} of type {type(obj)!r}")
        return fsspec.open(obj, mode=mode, **storage_options)
    else:
        if not isinstance(json_obj, dict):
            raise TypeError(f"got json {json_obj!r} of type {type(json_obj)!r}")
        try:
            _fs = json.loads(json_obj["fs"])
        except json.JSONDecodeError:
            # json_obj["fs"] is not json ...
            if storage_options:
                raise NotImplementedError(
                    "pickled filesystems can't change storage_options"
                )
            fs = pickle.loads(literal_eval(json_obj["fs"]))
        else:
            # json_obj["fs"] is json
            if not isinstance(_fs, dict):
                raise TypeError(f"expected dict, got {_fs!r}")
            _fs.update(**storage_options)
            fs = fsspec.AbstractFileSystem.from_json(json.dumps(_fs))
        return fsopen(fs, json_obj["path"], mode=mode)


def urlpathlike_to_fs_and_path(
    obj: UrlpathLike,
    *,
    storage_options: dict[str, Any] | None = None,
) -> Tuple[AbstractFileSystem, str]:
    """use an urlpath-like object and return an fsspec.AbstractFileSystem and a path"""
    if is_fsspec_open_file_like(obj):
        return obj.fs, obj.path

    if storage_options is None:
        storage_options = {}

    try:
        json_obj = json.loads(obj)  # type: ignore
    except (json.JSONDecodeError, TypeError):
        if isinstance(obj, os.PathLike):
            obj = os.fspath(obj)
        if not isinstance(obj, str):
            raise TypeError(f"got {obj!r} of type {type(obj)!r}")
        fs, _, (path,) = fsspec.get_fs_token_paths(obj, storage_options=storage_options)
        return fs, path
    else:
        if not isinstance(json_obj, dict):
            raise TypeError(f"got json {json_obj!r} of type {type(json_obj)!r}")
        try:
            _fs = json.loads(json_obj["fs"])
        except json.JSONDecodeError:
            # json_obj["fs"] is not json ...
            if storage_options:
                raise NotImplementedError(
                    "pickled filesystems can't change storage_options"
                )
            fs = pickle.loads(literal_eval(json_obj["fs"]))
        else:
            # json_obj["fs"] is json
            if not isinstance(_fs, dict):
                raise TypeError(f"expected dict, got {_fs!r}")
            _fs.update(**storage_options)
            fs = fsspec.AbstractFileSystem.from_json(json.dumps(_fs))
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
            path = json_obj["path"]
    return PurePath(path).parts


def urlpathlike_to_localpath(
    obj: UrlpathLike,
    *,
    mode: FsspecIOMode = "rb",
    storage_options: dict[str, Any] | None = None,
) -> str:
    """take an urlpathlike object and return a local path"""
    if "r" not in mode:
        raise ValueError("urlpathlike_to_localpath only works for read modes")
    of = urlpathlike_to_fsspec(obj, mode=mode, storage_options=storage_options)
    if not getattr(of.fs, "local_file", False):
        raise ValueError("FileSystem does not have attribute .local_file=True")
    with of as f:
        return f.name


def fsopen(
    fs: AbstractFileSystem,
    path: [str, os.PathLike],
    *,
    mode: FsspecIOMode = "rb",
) -> OpenFileLike:
    """small helper to support mode 'x' for fsspec filesystems"""
    if mode not in typing.get_args(FsspecIOMode):
        raise ValueError("fsspec only supports a subset of IOModes")
    if "x" in mode:
        if fs.exists(path):
            raise FileExistsError(f"{path!r} at {fs!r}")
        else:
            mode = mode.replace("x", "w")
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
        if not (hasattr(file, "read") and hasattr(file, "seek")):
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
                    "tar must contain exactly one file: won't auto uncompress"
                )
            ftarred = stack.enter_context(ftar.extractfile(members[0]))
            yield stack.enter_context(uncompressed(ftarred))

        else:
            yield file
