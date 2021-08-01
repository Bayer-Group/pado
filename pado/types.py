"""pado.types (and ABCs)

A collection of useful types and abstract base classes in pado
"""
from __future__ import annotations

import os
import sys
from abc import ABC
from abc import abstractmethod
from typing import IO
from typing import Mapping
from typing import TYPE_CHECKING
from typing import Union

import pandas as pd

if sys.version_info >= (3, 10):
    from typing import Literal  # 3.8+
    from typing import TypeAlias  # 3.10+
else:
    from typing_extensions import Literal
    from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from pado.annotations import Annotations
    from pado.images import ImageId
    from pado.img import Image


# --- types ---

UrlpathLike: TypeAlias = Union[str, bytes, os.PathLike, IO[str], IO[bytes]]
IOMode: TypeAlias = Literal['r', 'r+', 'w', 'a', 'x']


# --- abcs ---

class DatasetABC(ABC):

    @property
    @abstractmethod
    def metadata(self) -> pd.DataFrame:
        raise NotImplementedError("implement in subclass")

    @property
    @abstractmethod
    def images(self) -> Mapping[ImageId, Image]:
        raise NotImplementedError("implement in subclass")

    @property
    @abstractmethod
    def annotations(self) -> Mapping[ImageId, Annotations]:
        raise NotImplementedError("implement in subclass")
