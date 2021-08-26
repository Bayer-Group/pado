"""helpers for dealing with fsspec OpenFile"""
import fnmatch
import json
import os.path
from operator import itemgetter
from pathlib import PurePath
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple

from tqdm import tqdm

from pado.io.files import urlpathlike_to_fs_and_path
from pado.io.files import urlpathlike_to_fsspec
from pado.types import OpenFileLike
from pado.types import UrlpathLike


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


def match_partial_paths_reversed(
    current_urlpaths: Sequence[UrlpathLike],
    new_urlpaths: Sequence[OpenFileLike],
    *,
    ignore_ambiguous: bool = False,
    progress: bool = False,
) -> Sequence[OpenFileLike]:
    """match paths for re-assigning files

    returns a new sequence of paths replacing current with new ones
    raises ValueError in case match is ambiguous

    """
    current_path_parts = {}
    for _idx, urlpathlike in enumerate(current_urlpaths):
        fs, path = urlpathlike_to_fs_and_path(urlpathlike)
        parts = PurePath(path).parts
        current_path_parts[parts] = {
            'index': _idx,
            'cur': urlpathlike,
            'new': None,
        }
    parts = set(current_path_parts)

    def match(x: Tuple[str, ...], s: Set[Tuple[str, ...]], idx: int) -> Optional[Tuple[str, ...]]:
        try:
            xi = x[idx]  # raises index error when out of parts to match
        except IndexError:
            if ignore_ambiguous:
                return None
            else:
                raise ValueError(f"ambiguous: {x!r} -> {s!r}")
        sj = set(sx for sx in s if sx[idx] == xi or xi is None)
        if len(sj) == 1:
            return sj.pop()
        elif len(sj) == 0:
            return None
        else:
            return match(x, sj, idx - 1)

    if progress:
        new_urlpaths = tqdm(new_urlpaths, desc="matching new files")

    for of in new_urlpaths:
        m = match(PurePath(of.path).parts, parts, -1)
        if m:
            current_path_parts[m]['new'] = of

    return [
        x['new'] or x['cur']
        for x in sorted(current_path_parts.values(), key=itemgetter('index'))
    ]
