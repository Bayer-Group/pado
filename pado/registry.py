from __future__ import annotations

import json
import os
import warnings
from contextlib import AbstractContextManager
from contextlib import ExitStack
from typing import TYPE_CHECKING
from typing import Any
from typing import MutableMapping
from typing import Optional

import fsspec
from fsspec.core import OpenFile

from pado.settings import pado_config_path
from pado.settings import settings
from pado.types import FsspecIOMode
from pado.types import IOMode
from pado.types import UrlpathWithStorageOptions

if TYPE_CHECKING:
    from pado import PadoDataset

__all__ = [
    "dataset_registry",
    "list_registries",
    "open_registered_dataset",
]


class _DatasetRegistry(MutableMapping[str, UrlpathWithStorageOptions]):
    """a simple json file based key value store"""

    FILENAME = ".pado_dataset_registry.json"

    def __init__(self, name: str | None):
        self._name: str | None = name
        self._cm: Optional[ExitStack] = None
        self._data: Optional[dict] = None

    @property
    def is_default(self):
        return self._name is None

    def __enter__(self):
        # load the data
        try:
            with self._open(mode="r") as f:
                contents = f.read()
        except FileNotFoundError:
            self._data = {}
            return self

        try:
            self._data = json.loads(contents)
        except json.JSONDecodeError:
            of = self._open(mode="w")
            fn = of.path
            fs = of.fs
            warnings.warn(f"registry corrupted: moving to {fn}.corrupted")
            fs.move(fn, f"{fn}.corrupted")
            self._data = {}
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self._open(mode="w") as f:
            json.dump(self._data, f, indent=2)
        self._data = None

    def _open(self, mode: FsspecIOMode) -> OpenFile:
        # return a fsspec OpenFile instance for the registry json
        if self._name is None:
            urlpath = os.path.join(
                pado_config_path(ensure_dir=True),
                self.FILENAME,
            )
            storage_options = {}
        else:
            dct = settings.registry[self._name]
            urlpath = dct.pop("urlpath")
            storage_options = dct
        return fsspec.open(urlpath, mode=mode, **storage_options)

    def __contains__(self, name: str):
        if self._data is None:
            raise RuntimeError(f"{self!r} has to be used in a with statement")
        return name in self._data

    def __iter__(self):
        if self._data is None:
            raise RuntimeError(f"{self!r} has to be used in a with statement")
        return iter(self._data)

    def __len__(self):
        if self._data is None:
            raise RuntimeError(f"{self!r} has to be used in a with statement")
        return len(self._data)

    def __getitem__(self, name: str) -> UrlpathWithStorageOptions:
        if self._data is None:
            raise RuntimeError(f"{self!r} has to be used in a with statement")
        if not isinstance(name, str):
            raise TypeError(
                f"name must be a string, got {name!r} of {type(name).__name__}"
            )
        value = self._data[name]
        if isinstance(value, str):
            return UrlpathWithStorageOptions(value)
        else:
            return UrlpathWithStorageOptions(value["urlpath"], value["storage_options"])

    def __setitem__(
        self, name: str, path: str | UrlpathWithStorageOptions | dict[str, Any]
    ):
        if self._data is None:
            raise RuntimeError(f"{self!r} has to be used in a with statement")
        if not isinstance(name, str):
            raise TypeError(
                f"name must be a string, got {name!r} of {type(name).__name__}"
            )
        if isinstance(path, str):
            self._data[name] = path
        elif isinstance(path, UrlpathWithStorageOptions):
            self._data[name] = path._asdict()
        elif isinstance(path, dict):
            self._data[name] = {
                "urlpath": path["urlpath"],
                "storage_options": path.get("storage_options", None),
            }
        else:
            raise ValueError(
                f"unsupported value: {path!r} of type {type(path).__name__!r}"
            )

    def __delitem__(self, name: str):
        if self._data is None:
            raise RuntimeError(f"{self!r} has to be used in a with statement")
        del self._data[name]

    def items(self):
        for name in self:
            yield name, self[name]


def dataset_registry(
    name: str | None = None,
) -> AbstractContextManager[_DatasetRegistry]:
    """return the dataset registry instance"""
    return _DatasetRegistry(name)


def list_registries() -> dict[str, _DatasetRegistry]:
    """return additional registry instances

    Note: the default registry is not included in this output
    """
    return {key: _DatasetRegistry(key) for key in settings.registry}


def open_registered_dataset(
    name: str,
    *,
    mode: IOMode = "r",
    storage_options: dict[str, Any] | None = None,
) -> PadoDataset:
    """helper function to open a registered PadoDataset"""
    from pado.dataset import PadoDataset

    with dataset_registry() as dct:
        up, so = dct[name]
        if storage_options:
            so = so or {}
            so.update(storage_options)
        return PadoDataset(up, mode=mode, storage_options=so)
