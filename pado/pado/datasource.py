from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TypeVar, Union


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
    def inputs(self) -> Iterable[str]:
        ...

    @abstractmethod
    def cache(self) -> Optional[Path]:
        ...

    @abstractmethod
    def outputs(self) -> Iterable[Any]:
        ...

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


class FileDataSource(DataSource):
    def __init__(self, name: str, filename: Path):
        super().__init__(name)
        self._path = Path(filename)

    def inputs(self) -> Iterable[str]:
        return ()

    def cache(self):
        return None

    def outputs(self) -> Iterable[Path]:
        return (self._path,)

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
        **__
    ) -> Path:
        paths = [obj] if isinstance(obj, Path) else obj
        for o in paths:
            registry.register_datasource(cls(name, o))
        return obj


class FunctionDataSource(DataSource):
    def __init__(
        self, name, inputs: Iterable[DataSource] = (), cache: Optional[Path] = None
    ):
        super().__init__(name)
        self._inputs = inputs
        self._cache = cache

    def inputs(self) -> Iterable[str]:
        return tuple(inp.name for inp in self._inputs)

    def cache(self) -> Optional[Path]:
        return self._cache

    def outputs(self) -> Iterable[Any]:
        return ()

    def is_creatable(self) -> bool:
        if all(inp.is_creatable() for inp in self._inputs):
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
