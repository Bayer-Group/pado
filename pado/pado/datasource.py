from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TypeVar, Union

import requests


class DataSourceException(Exception):
    pass


class DataSourceRegistry:
    def __init__(self, identifier: str):
        self.identifier = identifier

    def register_datasource(self, data_source: "DataSource"):
        pass


class DataSource(ABC):
    """DataSource base class

    All data sources should go through this abstraction to
    allow channelling them into the same output format.

    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def inputs(self) -> Iterable["DataSource"]:
        ...

    @abstractmethod
    def cache(self) -> Optional[Path]:
        ...

    @abstractmethod
    def outputs(self) -> Iterable[Any]:
        ...

    def create(self, *args, **kwargs) -> None:
        raise DataSourceException(f"{self.__class__.__name__}.create is not supported")

    def get(self, *args, **kwargs) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__}.get is not implemented")

    @abstractmethod
    def is_creatable(self) -> bool:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    _T = TypeVar("_T")

    @classmethod
    @abstractmethod
    def pass_through_register(
        cls, registry: DataSourceRegistry, name: str, obj: _T, **kwargs
    ) -> _T:
        ...


class File(DataSource):
    def __init__(self, name: str, filename: Path):
        super().__init__(name)
        self._path = Path(filename)

    def inputs(self) -> Iterable[DataSource]:
        return ()

    def cache(self):
        return None

    def outputs(self) -> Iterable[Path]:
        return (self._path,)

    def get(self):
        return self._path

    def is_creatable(self) -> bool:
        return self._path.is_file()

    def is_cached(self) -> bool:
        return self._path.is_file()

    def is_available(self) -> bool:
        return self._path.is_file()

    @classmethod
    def pass_through_register(
        cls,
        registry: DataSourceRegistry,
        name: str,
        obj: Union[Path, Iterable[Path]],
        **__,
    ) -> Path:
        paths = [obj] if isinstance(obj, Path) else obj
        for o in paths:
            registry.register_datasource(cls(name, o))
        return obj


class DownloadableFile(File):
    def __init__(self, name: str, filename: Path, url: str):
        super().__init__(name, filename)
        self._url = url

    def is_creatable(self) -> bool:
        r = requests.head(self._url)
        return r.status_code == 200

    def create(self, force=False):
        mode = "xb" if not force else "wb"
        r = requests.get(self._url, stream=True)
        r.raise_for_status()
        with open(self._path, mode) as f:
            for chunk in r.iter_content(512):
                f.write(chunk)


class Function(DataSource):
    def __init__(
        self, name, inputs: Iterable[DataSource] = (), cache: Optional[Path] = None
    ):
        super().__init__(name)
        self._inputs = inputs
        self._cache = cache

    def inputs(self) -> Iterable[DataSource]:
        return tuple(self._inputs)

    def cache(self) -> Optional[Path]:
        return self._cache

    def outputs(self) -> Iterable[Any]:
        return ()

    def is_creatable(self) -> bool:
        if all(inp.is_available() or inp.is_creatable() for inp in self._inputs):
            return True

    def is_cached(self) -> bool:
        if self._cache is None:
            return False
        return self._cache.is_file()

    def is_available(self) -> bool:
        return False

    _C = TypeVar("_C", bound=Callable)

    @classmethod
    def pass_through_register(
        cls,
        registry: DataSourceRegistry,
        name: str,
        obj: _C,
        inputs: Iterable[DataSource] = (),
        cache: Optional[Path] = None,
    ) -> _C:
        registry.register_datasource(cls(name, inputs=inputs, cache=cache))
        return obj

    @classmethod
    def register(
        cls,
        registry: DataSourceRegistry,
        name: str,
        inputs: Iterable[DataSource] = (),
        cache: Optional[Path] = None,
    ) -> _C:
        def decorator(func):
            cls.pass_through_register(registry, name, func, inputs, cache)
            return func

        return decorator
