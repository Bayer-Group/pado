from abc import ABC, abstractmethod
from typing import List

import pandas as pd

import pado.resource
from pado.structure import PadoColumn


class DataSource(ABC):
    """DataSource base class

    All data sources should go through this abstraction to
    allow channelling them into the same output format.

    """

    identifier: str
    image_id_columns: List[str] = [PadoColumn.IMAGE]

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
