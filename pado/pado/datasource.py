from abc import ABC, abstractmethod
from typing import Any, Hashable, Iterable

import pandas as pd


class ImageResource(ABC):
    @property
    @abstractmethod
    def id(self) -> Hashable:
        ...

    @property
    @abstractmethod
    def path(self) -> Any:
        ...


class DataSource(ABC):
    """DataSource base class

    All data sources should go through this abstraction to
    allow channelling them into the same output format.

    """

    identifier: str

    @property
    @abstractmethod
    def metadata(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def images(self) -> Iterable[ImageResource]:
        ...

    def acquire(self, raise_if_missing: bool = True):
        pass

    def release(self):
        pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
