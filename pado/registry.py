from __future__ import annotations

import json
import os
import shutil
import warnings
from contextlib import AbstractContextManager
from contextlib import ExitStack
from typing import Any
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional

from pado import PadoDataset
from pado.settings import pado_config_path
from pado.types import IOMode
from pado.types import UrlpathLike

__all__ = [
    "dataset_registry",
    "open_registered_dataset",
]


class _DatasetRegistryItem(NamedTuple):
    urlpath: UrlpathLike
    storage_options: dict[str, str | int | float] | None = None


class _DatasetRegistry(MutableMapping[str, _DatasetRegistryItem]):
    """a simple json file based key value store"""

    FILENAME = ".pado_dataset_registry.json"

    def __init__(self):
        self._cm: Optional[ExitStack] = None
        self._data: Optional[dict] = None

    def __enter__(self):
        fn = os.path.join(
            pado_config_path(ensure_dir=True),
            self.FILENAME,
        )
        try:
            with open(fn) as f:
                self._data = json.load(f)
        except FileNotFoundError:
            self._data = {}
        except json.JSONDecodeError:
            shutil.move(fn, f"{fn}.corrupted")
            warnings.warn(f"registry corrupted and moved to {fn}.corrupted")
            self._data = {}
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        fn = os.path.join(
            pado_config_path(ensure_dir=True),
            self.FILENAME,
        )
        with open(fn, "w") as f:
            json.dump(self._data, f, indent=2)
        self._data = None

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

    def __getitem__(self, name: str) -> _DatasetRegistryItem:
        if self._data is None:
            raise RuntimeError(f"{self!r} has to be used in a with statement")
        if not isinstance(name, str):
            raise TypeError(
                f"name must be a string, got {name!r} of {type(name).__name__}"
            )
        value = self._data[name]
        if isinstance(value, str):
            return _DatasetRegistryItem(value)
        else:
            return _DatasetRegistryItem(value["urlpath"], value["storage_options"])

    def __setitem__(self, name: str, path: str | _DatasetRegistryItem | dict[str, Any]):
        if self._data is None:
            raise RuntimeError(f"{self!r} has to be used in a with statement")
        if not isinstance(name, str):
            raise TypeError(
                f"name must be a string, got {name!r} of {type(name).__name__}"
            )
        if isinstance(path, str):
            self._data[name] = path
        elif isinstance(path, _DatasetRegistryItem):
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


def dataset_registry() -> AbstractContextManager[_DatasetRegistry]:
    """return the dataset registry instance"""
    return _DatasetRegistry()


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
