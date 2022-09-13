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
import warnings
import zipfile
from ast import literal_eval
from contextlib import ExitStack
from contextlib import contextmanager
from typing import Any
from typing import AnyStr
from typing import BinaryIO
from typing import Collection
from typing import ContextManager
from typing import Iterator
from typing import NamedTuple
from typing import Tuple
from typing import Union

import fsspec
from fsspec import AbstractFileSystem
from fsspec.core import OpenFile
from fsspec.core import strip_protocol
from fsspec.implementations.local import LocalFileSystem
from fsspec.registry import _import_class
from fsspec.registry import get_filesystem_class
from fsspec.utils import get_protocol
from fsspec.utils import infer_storage_options

from pado.types import FsspecIOMode
from pado.types import OpenFileLike
from pado.types import UrlpathLike

if sys.version_info[:2] >= (3, 10):
    from typing import Literal
    from typing import TypeGuard
    from typing import get_args
else:
    from typing_extensions import Literal
    from typing_extensions import TypeGuard
    from typing_extensions import get_args

__all__ = [
    "find_files",
    "is_fsspec_open_file_like",
    "update_fs_storage_options",
    "urlpathlike_local_via_fs",
    "urlpathlike_get_fs_cls",
    "urlpathlike_get_path",
    "urlpathlike_get_storage_args_options",
    "urlpathlike_is_localfile",
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
_PADO_ALLOW_PICKLED_URLPATHS = None


def _pado_pickle_load(obj: Any):
    """load a pickled urlpath fs"""
    global _PADO_ALLOW_PICKLED_URLPATHS

    if _PADO_ALLOW_PICKLED_URLPATHS is None:
        from pado.settings import settings

        _PADO_ALLOW_PICKLED_URLPATHS = settings.allow_pickled_urlpaths

    if not _PADO_ALLOW_PICKLED_URLPATHS:
        raise Exception(
            "Loading pickled urlpaths is disabled. "
            "Use PADO_ALLOW_PICKLED_URLPATHS=1 to enable, or set in your"
            ".pado.toml config file."
        )
    else:
        return pickle.loads(literal_eval(obj))  # nosec B301


def _os_path_parts(pth: str) -> tuple[str, ...]:
    """os.path version of pathlib.Path().parts"""
    remaining, part = os.path.split(pth)
    if remaining == pth:
        return (remaining,)
    elif part == pth:
        return (pth,)
    else:
        return (*_os_path_parts(remaining), part)


class _OpenFileAndParts(NamedTuple):
    file: OpenFile
    parts: Tuple[str, ...]


def find_files(
    urlpath: UrlpathLike,
    *,
    glob: str = "**/*",
    storage_options: dict[str, Any] | None = None,
) -> list[_OpenFileAndParts]:
    """iterate over the files with matching at all paths"""
    if storage_options is None:
        storage_options = {}

    ofile = urlpathlike_to_fsspec(urlpath, storage_options=storage_options)
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


def _deserialize_fsspec_json(obj: Any) -> dict[str, Any] | None:
    """try to deserialize"""
    try:
        json_obj = json.loads(obj)  # type: ignore
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(json_obj, dict):
        # this is unexpected, so its better to raise instead of returning None
        raise TypeError(f"got json {json_obj!r} of type {type(json_obj)!r}")

    return json_obj


def _get_fsspec_cls_from_serialized_fs(fs_obj: str) -> type[AbstractFileSystem]:
    """try to return the filesystem cls from a json string or pickle"""
    try:
        fs_dct = json.loads(fs_obj)
    except json.JSONDecodeError:
        # json_obj["fs"] is not json ...
        fs = _pado_pickle_load(fs_obj)
        return type(fs)
    else:
        # json_obj["fs"] is json
        if not isinstance(fs_dct, dict):
            raise TypeError(f"expected dict, got {fs_dct!r}")

        protocol = fs_dct.pop("protocol")
        try:
            cls = _import_class(fs_dct.pop("cls"))
        except (ImportError, ValueError, RuntimeError, KeyError):
            cls = get_filesystem_class(protocol)
        return cls


def _get_fsspec_storage_args_options_from_serialized_fs(
    fs_obj: str,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """try to return the filesystem storage args and options from a json string or pickle"""
    try:
        fs_dct = json.loads(fs_obj)
    except json.JSONDecodeError:
        # json_obj["fs"] is not json ...
        fs = _pado_pickle_load(fs_obj)
        return fs.storage_args, fs.storage_options
    else:
        # json_obj["fs"] is json
        if not isinstance(fs_dct, dict):
            raise TypeError(f"expected dict, got {fs_dct!r}")

        _ = fs_dct.pop("protocol")
        _ = fs_dct.pop("cls")
        storage_args = fs_dct.pop("args")
        return storage_args, fs_dct


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


def _build_uri(
    cls_name,
    protocol,
    path,
    custom_args,
    custom_options,
    *,
    ignore_options,
    repr_fallback,
):
    if ignore_options is True:
        custom_options.clear()
    elif ignore_options:
        for key in ignore_options:
            if key in custom_options:
                del custom_options[key]

    if not custom_args and not custom_options:
        return f"{protocol}://{path}"

    elif not custom_args:
        # todo: ... can prettify some file systems
        pass

    if not repr_fallback:
        raise ValueError("can't serialize urlpath to a pure uri")

    # provide a repr like string
    fs_params = [
        *custom_args,
        *(f"{k}={v!r}" for k, v in custom_options.items()),
    ]
    return f"{cls_name}({', '.join(fs_params)}).open({path!r})"


def urlpathlike_to_uri(
    urlpath: UrlpathLike,
    *,
    repr_fallback: bool = False,
    ignore_options: Collection[str] | Literal[True] = (),
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
    if isinstance(urlpath, str) and urlpath[0] == "{" and urlpath[-1] == "}":
        obj = _deserialize_fsspec_json(urlpath)
        if obj:
            _fs = obj["fs"]
            if isinstance(_fs, str):
                _fs = json.loads(_fs)
            cls_name = _fs["cls"]
            custom_args = _fs.pop("args")
            custom_options = _fs
            path = obj["path"]
            protocol = _fs.pop("protocol")
            return _build_uri(
                cls_name,
                protocol,
                path,
                custom_args,
                custom_options,
                ignore_options=ignore_options,
                repr_fallback=repr_fallback,
            )

    if is_fsspec_open_file_like(urlpath):
        fs = urlpath.fs
        path = urlpath.path

        cls = type(fs)
        cls_name = f"{cls.__module__}.{cls.__name__}"
        custom_args = fs.storage_args
        default_options = _get_constructor_default_options(cls)
        custom_options = _remove_duplicate_items(fs.storage_options, default_options)

        protocol = fs.protocol
        if isinstance(protocol, (list, tuple)):
            protocol = protocol[0]

        return _build_uri(
            cls_name,
            protocol,
            path,
            custom_args,
            custom_options,
            ignore_options=ignore_options,
            repr_fallback=repr_fallback,
        )

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
            fs = _pado_pickle_load(json_obj["fs"])
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
            fs = _pado_pickle_load(json_obj["fs"])
        else:
            # json_obj["fs"] is json
            if not isinstance(_fs, dict):
                raise TypeError(f"expected dict, got {_fs!r}")
            _fs.update(**storage_options)
            fs = fsspec.AbstractFileSystem.from_json(json.dumps(_fs))
        return fs, json_obj["path"]


def urlpathlike_get_fs_cls(obj: UrlpathLike) -> type[AbstractFileSystem]:
    """get the urlpathlike filesystem class"""
    if is_fsspec_open_file_like(obj):
        return type(obj.fs)

    obj_json = _deserialize_fsspec_json(obj)
    if obj_json:
        return _get_fsspec_cls_from_serialized_fs(obj_json["fs"])

    if isinstance(obj, os.PathLike):
        obj = os.fspath(obj)

    if not isinstance(obj, str):
        raise TypeError(f"got {obj!r} of type {type(obj)!r}")

    return get_filesystem_class(get_protocol(obj))


def urlpathlike_get_path(
    obj: UrlpathLike, *, fs_cls: type[AbstractFileSystem] | None = None
) -> str:
    """get the path from the urlpath"""
    if is_fsspec_open_file_like(obj):
        return obj.path

    obj_json = _deserialize_fsspec_json(obj)
    if obj_json:
        return obj_json["path"]

    if isinstance(obj, os.PathLike):
        obj = os.fspath(obj)

    if not isinstance(obj, str):
        raise TypeError(f"got {obj!r} of type {type(obj)!r}")

    if fs_cls is None:
        fs_cls = get_filesystem_class(get_protocol(obj))

    # noinspection PyProtectedMember
    return fs_cls._strip_protocol(obj)


def urlpathlike_get_storage_args_options(
    obj: UrlpathLike,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """get the path from the urlpath"""
    if is_fsspec_open_file_like(obj):
        fs = obj.fs
        return fs.storage_args, fs.storage_options

    obj_json = _deserialize_fsspec_json(obj)
    if obj_json:
        fs_str = obj_json["fs"]
        return _get_fsspec_storage_args_options_from_serialized_fs(fs_str)

    if isinstance(obj, os.PathLike):
        obj = os.fspath(obj)

    if not isinstance(obj, str):
        raise TypeError(f"got {obj!r} of type {type(obj)!r}")

    return (), infer_storage_options(obj)


def urlpathlike_local_via_fs(
    obj: UrlpathLike,
    fs: AbstractFileSystem,
) -> UrlpathLike:
    """take an urlpath and access it via another fs"""
    fs_cls = urlpathlike_get_fs_cls(obj)
    if issubclass(fs_cls, LocalFileSystem):
        path = urlpathlike_get_path(obj, fs_cls=fs_cls)
        return fsopen(fs, path)
    else:
        return obj


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
    return _os_path_parts(path)


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


def update_fs_storage_options(
    fs: AbstractFileSystem, *, storage_options: dict[str, Any] | None
) -> AbstractFileSystem:
    """update the storage_options of an existing filesystem"""
    if not storage_options:
        return fs
    make_instance, (cls, fs_args, fs_so) = fs.__reduce__()
    fs_so.update(storage_options)
    return make_instance(cls, fs_args, fs_so)


def fsopen(
    fs: AbstractFileSystem,
    path: [str, os.PathLike],
    *,
    mode: FsspecIOMode = "rb",
) -> OpenFile:
    """small helper to support mode 'x' for fsspec filesystems"""
    if mode not in get_args(FsspecIOMode):
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


def urlpathlike_is_localfile(urlpath: UrlpathLike, must_exist: bool = True) -> bool:
    """Check whether an urlpath corresponds to a file on a local filesystem.

    Parameters
    ----------
    urlpath : UrlpathLike
        URL to be checked
    must_exist : bool, optional
        If True, the file must exist on the local machine's filesystem. Default is True.

    Returns
    -------
    bool
        True if `urlpath` corresponds to a local file, False otherwise
    """
    fs, pth = urlpathlike_to_fs_and_path(urlpath)
    return isinstance(fs, LocalFileSystem) and (fs.exists(pth) or not must_exist)
