from __future__ import annotations

import base64
import json
import os
import warnings
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterator
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
    "has_secrets",
    "list_secrets",
    "set_secret",
]


class _JsonFileData(AbstractContextManager):
    """load a dict from json in a context manager"""

    FILENAME: str  # define in subclasses

    def __init__(self, name: str | None):
        super().__init__()
        self._name: str | None = name
        self._data: Optional[dict] = None

    @property
    def data(self):
        if self._data is None:
            raise RuntimeError(f"{self!r} has to be used in a with statement")
        return self._data

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


class _UserInputSecretStore(MutableMapping[str, str], _JsonFileData):
    """a simple json file based secret store for the registry

    note: this is not encrypted!
    """

    FILENAME = ".pado_secrets.json"

    def __init__(self):
        super().__init__(None)

    @staticmethod
    def make_key(name, dataset_name, secret_name):
        if not name.isidentifier():
            raise ValueError(f"{name!r} should have been a valid identifier")
        if not dataset_name.isidentifier():
            raise ValueError(f"{dataset_name!r} should have been a valid identifier")
        if not secret_name.isidentifier():
            raise ValueError(f"{secret_name!r} should have been a valid identifier")
        return f"@SECRET:{name}:{dataset_name}:{secret_name}"

    @staticmethod
    def parse_secret(key):
        if not isinstance(key, str):
            raise TypeError(f"key must be of type str, got {type(key).__name__}")
        if not key.startswith("@SECRET:"):
            return None
        try:
            _, name, dataset_name, secret_name = key.split(":")
        except ValueError:
            return None
        if not name.isidentifier():
            raise RuntimeError(f"{name!r} should have been a valid identifier")
        if not dataset_name.isidentifier():
            raise RuntimeError(f"{dataset_name!r} should have been a valid identifier")
        if not secret_name.isidentifier():
            raise RuntimeError(f"{secret_name!r} should have been a valid identifier")
        return name, dataset_name, secret_name

    def set(
        self, registry_name: str | None, dataset_name: str, secret_name: str, value: str
    ):
        key = self.make_key(registry_name, dataset_name, secret_name)
        self[key] = value
        return key

    def __setitem__(self, k: str, v: str) -> None:
        if self.parse_secret(k) is None:
            raise KeyError("prefer using .set(...)")
        v = json.dumps(v)
        self.data[k] = base64.urlsafe_b64encode(v.encode()).decode()

    def __delitem__(self, k: str) -> None:
        if self.parse_secret(k) is None:
            raise KeyError(k)
        del self.data[k]

    def __getitem__(self, k: str) -> str:
        if self.parse_secret(k) is None:
            raise KeyError(k)
        v = self.data[k]
        v = base64.urlsafe_b64decode(v.encode()).decode()
        return json.loads(v)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)


secret_stores = {"user_input": _UserInputSecretStore()}


def is_secret(secret_name: str) -> bool:
    """return if secret is a secret"""
    return _UserInputSecretStore.parse_secret(secret_name) is not None


_NO_DEFAULT = object()


def get_secret(value: str, *, default: Any = _NO_DEFAULT) -> str:
    """return the secret from stores"""
    if not is_secret(value):
        return value

    for store in secret_stores.values():
        try:
            with store:
                return store[value]
        except KeyError:
            pass
    if default is _NO_DEFAULT:
        raise KeyError(f"{value} not in stores")
    return default


def has_secrets(value: str | UrlpathWithStorageOptions) -> bool:
    return len(list_secrets(value)) > 0


def set_secret(
    secret_name: str,
    value: str,
    *,
    registry_name: str | None = ...,
    dataset_name: str | None = None,
):
    """set a secret"""
    store = secret_stores["user_input"]
    if is_secret(secret_name):
        registry_name, dataset_name, secret_name = store.parse_secret(secret_name)
    else:
        if registry_name is ...:
            raise ValueError(
                "must provide registry_name if secret is a python identifier"
            )
        elif registry_name is not None:
            raise NotImplementedError("todo: named registries")
        else:
            registry_name = "__default__"
        if dataset_name is None:
            raise ValueError(
                "must provide dataset_name if secret is a python identifier"
            )
    with store:
        return store.set(registry_name, dataset_name, secret_name, value)


def list_secrets(value: str | UrlpathWithStorageOptions) -> list[str]:
    if isinstance(value, str):
        check = [value]
    elif isinstance(value, UrlpathWithStorageOptions):
        so = value.storage_options or {}
        check = [value.urlpath, *so.values()]
    else:
        raise TypeError("must provide str or UrlpathWithStorageOptions")
    return [x for x in check if is_secret(x)]


class _DatasetRegistry(MutableMapping[str, UrlpathWithStorageOptions], _JsonFileData):
    """a simple json file based key value store"""

    FILENAME = ".pado_dataset_registry.json"

    def __init__(self, name: str | None):
        super().__init__(name)

    def __contains__(self, name: str):
        return name in self.data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, name: str) -> UrlpathWithStorageOptions:
        if not isinstance(name, str):
            raise TypeError(
                f"name must be a string, got {name!r} of {type(name).__name__}"
            )
        value = self.data[name]
        if isinstance(value, str):
            value = get_secret(value, default=value)
            return UrlpathWithStorageOptions(value)
        else:
            urlpath = get_secret(value["urlpath"], default=value["urlpath"])
            if value["storage_options"] is None:
                so = None
            else:
                so = {
                    k: get_secret(v, default=v)
                    for k, v in value["storage_options"].items()
                }
            return UrlpathWithStorageOptions(urlpath, so)

    def __setitem__(
        self, name: str, path: str | UrlpathWithStorageOptions | dict[str, Any]
    ):
        if not isinstance(name, str):
            raise TypeError(
                f"name must be a string, got {name!r} of {type(name).__name__}"
            )
        if not name.isidentifier():
            raise ValueError("name must be a valid python identifier")
        if isinstance(path, str):
            self.data[name] = path
        elif isinstance(path, UrlpathWithStorageOptions):
            self.data[name] = path._asdict()
        elif isinstance(path, dict):
            self.data[name] = {
                "urlpath": path["urlpath"],
                "storage_options": path.get("storage_options", None),
            }
        else:
            raise ValueError(
                f"unsupported value: {path!r} of type {type(path).__name__!r}"
            )

    def __delitem__(self, name: str):
        del self.data[name]

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
