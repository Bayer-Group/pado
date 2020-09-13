from abc import ABC, abstractmethod

import pandas as pd

import pado.resource


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

    @property
    @abstractmethod
    def images(self) -> pado.resource.ImageResourcesProvider:
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
