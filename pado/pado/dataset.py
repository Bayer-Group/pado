from __future__ import annotations

import os
import pathlib
import re
import shutil
from pathlib import Path
from typing import Literal, Union

from pado.structure import File, Group

PathOrStr = Union[str, os.PathLike]


def is_pado_dataset(path: PathOrStr) -> bool:
    """check if the given path is a valid pado dataset"""
    # fixme: skeleton implementation
    # todo:
    #   - verify folder and file structure
    #
    path = Path(path)
    if not (path / "pado.dataset.toml").is_file():
        return False

    return True


def verify_pado_dataset_integrity(path: PathOrStr) -> bool:
    """verify file integrity of a pado dataset"""
    # fixme: skeleton implementation
    # todo:
    #   - verify file hashes
    #   - add specific file or store ?in pado.dataset.toml?
    #
    path = Path(path)
    if not is_pado_dataset(path):
        raise ValueError("provided Path is not a pado dataset")

    dataset_dir = path.parent
    for file in dataset_dir.glob("**/*"):
        # todo: check integrity
        pass

    return True


def _key_to_path(root, key):
    p = pathlib.PurePath(key)
    if p.is_absolute():
        p = p.relative_to("/")
    key_path = root / p
    if Path(os.path.commonpath([root.parts, key_path.parts])) != root:
        raise ValueError("can't break out of PadoDataset")
    return key_path


def _create_missing_group_dirs(root, key):
    p = _key_to_path(root, key)
    p.parent.mkdir(parents=True, exist_ok=True)


DatasetIOMode = Union[
    Literal["r"],
    Literal["r+"],
    Literal["w"],
    Literal["w+"],
    Literal["a"],
    Literal["a+"],
    Literal["x"],
    Literal["x+"],
]


class PadoDataset:
    def __init__(self, path: Union[str, pathlib.Path], mode: DatasetIOMode = "r"):
        """open or create a new PadoDataset

        Parameters
        ----------
        path:
            path to `pado.dataset.toml` file, or its parent directory
        mode:
            'r' --> readonly, error if not there
            'r+' --> read/write, error if not there
            'a' = 'a+' --> read/write, create if not there, append if there
            'w' = 'w+' --> read/write, create if not there, truncate if there
            'x' = 'x+' --> read/write, create if not there, error if there

        """
        self._config = pathlib.Path(path)
        self._mode = str(mode)

        # guarantee p points to `pado.dataset.toml` file (allow directory)
        if not self._config.suffix:
            self._config /= "pado.dataset.toml"
        elif self._config.suffix != ".toml":
            raise ValueError("dataset file requires '.toml' suffix")

        if not re.match(r"^[rawx][+]?$", mode):
            raise ValueError(f"unsupported mode '{mode}'")

        p = self._config.expanduser().absolute()
        _exists = p.is_file()

        self._readonly = mode == "r"
        self._path = self._config.parent

        if mode in {"r", "r+"} and not _exists:
            raise FileNotFoundError(p)
        elif mode in {"x", "x+"} and _exists:
            raise FileExistsError(p)
        elif mode in {"w", "w+"} and _exists:
            shutil.rmtree(self._path, ignore_errors=True)
            _exists = False

        if _exists:
            if not verify_pado_dataset_integrity(path):
                raise RuntimeError("dataset integrity degraded")
        else:
            pass

    @property
    def path(self):
        return self._path

    def __getitem__(self, key: str):
        path = _key_to_path(self._path, key)
        if path.is_dir():
            return Group(path, _root=self)
        elif path.is_file():
            return File(path, _root=self)
        else:
            raise KeyError(key)

    def __delitem__(self, key: str):
        path = _key_to_path(self._path, key)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()
        else:
            raise KeyError(key)

    def __contains__(self, key: str):
        path = _key_to_path(self._path, key)
        if path.is_dir():
            return True
        elif path.is_file():
            return True
        else:
            return False

    def __setitem__(self, key: str, item: Union[Group, File]):
        path = _key_to_path(self._path, key)
        _create_missing_group_dirs(self._path, key)
        # fixme
        pass


    def query(self, *query_args, **query_kwargs) -> PadoDatasetView:
        raise NotImplementedError("todo")


class PadoDatasetView:
    """a container for accessing the various data in filtered form"""

    pass
