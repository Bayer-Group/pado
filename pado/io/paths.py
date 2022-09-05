"""helpers for dealing with fsspec OpenFile"""
from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple

from fsspec import AbstractFileSystem
from fsspec.core import OpenFile
from tqdm import tqdm

from pado.io.files import fsopen
from pado.io.files import urlpathlike_to_path_parts
from pado.types import FsspecIOMode
from pado.types import UrlpathLike

if TYPE_CHECKING:
    from pado.dataset import PadoDataset


__all__ = [
    "match_partial_paths_reversed",
    "search_dataset",
    "get_dataset_fs",
]


def match_partial_paths_reversed(
    current_urlpaths: Sequence[UrlpathLike],
    new_urlpaths: Sequence[UrlpathLike],
    *,
    ignore_ambiguous: bool = False,
    progress: bool = False,
) -> Sequence[UrlpathLike]:
    """match paths for re-assigning files

    returns a new sequence of paths replacing current with new ones
    raises ValueError in case match is ambiguous

    """
    current_path_parts = {}
    for _idx, urlpathlike in enumerate(current_urlpaths):
        parts = urlpathlike_to_path_parts(urlpathlike)
        current_path_parts[parts] = {
            "index": _idx,
            "cur": urlpathlike,
            "new": None,
        }
    parts = set(current_path_parts)

    def match(
        x: Tuple[str, ...], s: Set[Tuple[str, ...]], idx: int
    ) -> Optional[Tuple[str, ...]]:
        try:
            xi = x[idx]  # raises index error when out of parts to match
        except IndexError:
            if ignore_ambiguous:
                return None
            else:
                raise ValueError(f"ambiguous: {x!r} -> {s!r}")
        sj = {sx for sx in s if sx[idx] == xi or xi is None}
        if len(sj) == 1:
            return sj.pop()
        elif len(sj) == 0:
            return None
        else:
            return match(x, sj, idx - 1)

    if progress:
        new_urlpaths = tqdm(new_urlpaths, desc="trying to match new files")

    for new_up in new_urlpaths:
        new_parts = urlpathlike_to_path_parts(new_up)
        m = match(new_parts, parts, -1)
        if m:
            current_path_parts[m]["new"] = new_up

    return [
        x["new"] or x["cur"]
        for x in sorted(current_path_parts.values(), key=itemgetter("index"))
    ]


def search_dataset(
    ds: PadoDataset, glob: str, *, mode: FsspecIOMode = "rb"
) -> list[OpenFile]:
    """search for files in a pado dataset root path"""
    # noinspection PyProtectedMember
    fs, get_fspath = ds._fs, ds._get_fspath
    return [fsopen(fs, p, mode=mode) for p in fs.glob(get_fspath(glob))]


def get_dataset_fs(ds: PadoDataset) -> AbstractFileSystem:
    """return the dataset's filesystem"""
    # noinspection PyProtectedMember
    try:
        return ds._fs
    except AttributeError:
        raise TypeError(f"not a PadoDataset, got {type(ds).__name__}")
