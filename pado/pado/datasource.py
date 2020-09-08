from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from pado.structure import PadoColumn


class ImageResource(ABC):
    @property
    @abstractmethod
    def id(self) -> Tuple[str, ...]:
        ...

    @property
    @abstractmethod
    def path(self) -> Path:
        """this should point to the image resource"""
        ...


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
