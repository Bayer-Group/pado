"""file utility functions"""
import hashlib
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Iterator
from typing import NamedTuple
from typing import Tuple
from zipfile import ZipFile

from fsspec.core import OpenFile

if sys.version_info[:2] >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

import fsspec

from pado.types import OpenFileLike
from pado.types import UrlpathLike

_logger = logging.getLogger(__name__)


def hash_file(path, hasher=hashlib.sha256) -> str:
    """calculate the hash of a file"""
    hasher = hasher()
    with open(path, "rb") as reader:
        for chunk in iter(lambda: reader.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_str(string: str, hasher=hashlib.sha256) -> str:
    """calculate the hash of a string"""
    return hasher(string.encode()).hexdigest()


def hash_zip(path, hasher=hashlib.sha256) -> str:
    """calculate the hash of a zip file

    Uses full paths and file contents to hash a zip file.
    That is, builds a hash based on paths within zip and content of each decompressed file.
    Ignores everything else, for example, file date.
    """
    hasher = hasher()
    with ZipFile(path) as reader:
        for file_info in sorted(reader.infolist(), key=lambda x: x.filename):
            hasher.update(file_info.filename.encode("utf-8"))
            with reader.open(file_info.filename, mode="r") as file_reader:
                for chunk in iter(lambda: file_reader.read(8192), b""):
                    hasher.update(chunk)
    return hasher.hexdigest()


class FilesDupeFinder:
    def __init__(self, hasher=hash_zip, paths=None):
        """find duplicate files according to hash implementation"""
        super().__init__()
        self.dupes = defaultdict(set)  # a map hash -> files
        self.hasher = hasher
        self.update(*(paths or ()))

    def update(self, *paths):
        for path in paths:
            if Path(path).is_file():
                self.dupes[self.hasher(path)].add(str(path))

    def dupes(self):
        return sorted(group for group in self.dupes.values() if len(group) > 1)


def zip_inplace(dest_dir, path, delete_path=False, file_format="zip"):
    """create an archive of the directory at `path` and store at dest_dir"""
    dest_dir, path = Path(dest_dir), Path(path)
    if not path.is_dir():
        raise ValueError("path {} must be a directory".format(path))
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_zip = dest_dir / path.stem
    shutil.make_archive(
        base_name=dest_zip, format=file_format, root_dir=path.parent, base_dir=path.name
    )
    if delete_path:
        shutil.rmtree(path)


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
