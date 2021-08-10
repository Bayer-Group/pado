"""helpers for dealing with fsspec OpenFile"""
import fnmatch
import json
import os.path
from pado.types import UrlpathLike
from pado.util.files import urlpathlike_to_fsspec


def get_root_dir(urlpath: UrlpathLike, *, allow_file: str = "*.toml") -> UrlpathLike:
    """return the root dir from a urlpath-like path to a dir or file"""
    ofile = urlpathlike_to_fsspec(urlpath)
    fs = ofile.fs
    pth = ofile.path
    root, file = os.path.split(pth)
    if fnmatch.fnmatch(file, allow_file):
        return json.dumps({
            "path": root,
            "fs": fs.to_json(),
        })
    else:
        return urlpath


